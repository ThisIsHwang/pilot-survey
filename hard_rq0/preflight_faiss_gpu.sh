#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || {
  echo "Missing .venv-pilot; run bash scripts/bootstrap.sh first." >&2
  exit 1
}
E5_GPU=${E5_GPU:-7}
E5_MIN_FREE_MEMORY_MIB=${E5_MIN_FREE_MEMORY_MIB:-40960}
FAISS_SMOKE_TEMP_MEMORY_MIB=${FAISS_SMOKE_TEMP_MEMORY_MIB:-64}

for name in E5_GPU E5_MIN_FREE_MEMORY_MIB FAISS_SMOKE_TEMP_MEMORY_MIB; do
  value=${!name}
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$name must be a non-negative integer; got '$value'." >&2
    exit 2
  }
done
if [[ "$E5_MIN_FREE_MEMORY_MIB" == 0 || "$FAISS_SMOKE_TEMP_MEMORY_MIB" == 0 ]]; then
  echo "E5_MIN_FREE_MEMORY_MIB and FAISS_SMOKE_TEMP_MEMORY_MIB must be positive." >&2
  exit 2
fi
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required for the FAISS GPU preflight." >&2
  exit 1
}

free_mib=$(nvidia-smi --id="$E5_GPU" --query-gpu=memory.free \
  --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$free_mib" =~ ^[0-9]+$ ]] || {
  echo "Unable to read free memory for E5 GPU $E5_GPU." >&2
  exit 1
}
if (( free_mib < E5_MIN_FREE_MEMORY_MIB )); then
  echo "E5 GPU $E5_GPU has ${free_mib} MiB free; the paged 30.06 GiB FP16 index requires at least ${E5_MIN_FREE_MEMORY_MIB} MiB headroom." >&2
  echo "Stop the stale GPU process before loading the Hard-RQ0 E5 index." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$E5_GPU" \
FAISS_SMOKE_TEMP_MEMORY_MIB="$FAISS_SMOKE_TEMP_MEMORY_MIB" \
  "$PILOT_PYTHON" - <<'PY'
import os

import faiss
import numpy as np

from stackpilot.faiss_gpu import paged_flat_gpu_loader

if faiss.get_num_gpus() != 1:
    raise SystemExit(
        f"Paged FAISS smoke test requires one visible GPU; found {faiss.get_num_gpus()}"
    )
vectors = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
cpu_index = faiss.IndexFlatIP(vectors.shape[1])
cpu_index.add(vectors)
options = faiss.GpuMultipleClonerOptions()
options.useFloat16 = True
options.shard = True
scratch = int(os.environ["FAISS_SMOKE_TEMP_MEMORY_MIB"])
with paged_flat_gpu_loader(faiss, temp_memory_mib=scratch) as state:
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=options)
    scores, indices = gpu_index.search(vectors[:1], 1)
    if (
        gpu_index.ntotal != len(vectors)
        or state.documents != len(vectors)
        or int(indices[0, 0]) != 0
        or not np.isfinite(scores).all()
    ):
        raise SystemExit(
            "Paged FAISS GPU add/search smoke test returned an invalid result: "
            f"ntotal={gpu_index.ntotal}, state={state}, "
            f"indices={indices.tolist()}, scores={scores.tolist()}"
        )
print(
    "Paged FAISS GPU add/search smoke test passed on "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}."
)
PY

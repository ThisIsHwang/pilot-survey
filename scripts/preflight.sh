#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Required platform: Linux x86_64; found $(uname -s) $(uname -m)." >&2
  exit 1
fi
for command_name in git curl nvidia-smi nvcc g++; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 1
  fi
done
if ! nvcc --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "ColBERT requires the CUDA 12.9 toolkit (nvcc), not only a CUDA driver." >&2
  nvcc --version >&2 || true
  exit 1
fi
if [[ ! -x "$ROOT/.venv-pilot/bin/python" ]]; then
  echo "Missing .venv-pilot. Run: bash scripts/bootstrap.sh" >&2
  exit 1
fi
if [[ ! -x "$ROOT/.venv-vllm/bin/python" ]]; then
  echo "Missing .venv-vllm. Run: bash scripts/bootstrap_vllm.sh" >&2
  exit 1
fi

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
VLLM_PYTHON=$ROOT/.venv-vllm/bin/python
TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_extensions
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "== GPU allocation =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,mig.mode.current \
  --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ $GPU_COUNT -ne 8 ]]; then
  echo "Expected exactly 8 H100 GPUs on this node; nvidia-smi reports $GPU_COUNT." >&2
  exit 1
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv 'H100'; then
  echo "Every GPU must be an H100." >&2
  exit 1
fi
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
  grep -Eqv '^[[:space:]]*Disabled[[:space:]]*$'; then
  echo "MIG must be disabled on all eight H100 GPUs for this run." >&2
  exit 1
fi
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-61440}
if ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
  awk -v minimum="$MIN_FREE_GPU_MIB" '$1 < minimum { exit 1 }'; then
  echo "Each GPU needs at least ${MIN_FREE_GPU_MIB} MiB free for this run." >&2
  echo "Stop stale jobs or lower MIN_FREE_GPU_MIB only if the allocation is intentionally shared." >&2
  exit 1
fi

echo "== Pilot CUDA/Python stack =="
"$PILOT_PYTHON" - <<'PY'
import sys
import re
import subprocess
import sysconfig
from pathlib import Path

import faiss
import torch
from torch.utils.cpp_extension import CUDA_HOME

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"pilot Python must be 3.12, found {sys.version.split()[0]}")
python_header = Path(sysconfig.get_paths()["include"]) / "Python.h"
if not python_header.is_file():
    raise SystemExit(f"ColBERT requires Python 3.12 development headers; missing {python_header}")
if torch.version.cuda != "12.9" or not torch.cuda.is_available():
    raise SystemExit(f"pilot torch CUDA mismatch: torch={torch.__version__}, CUDA={torch.version.cuda}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"pilot torch sees {torch.cuda.device_count()} GPUs, expected 8")
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    if "H100" not in props.name or props.total_memory < 70 * 1024**3:
        raise SystemExit(f"GPU {index} is not a full H100: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
torch.ones(1, device="cuda").add_(1)
if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() != 8:
    raise SystemExit(f"FAISS GPU validation failed; visible GPUs={faiss.get_num_gpus()}")
if CUDA_HOME is None:
    raise SystemExit("PyTorch cannot locate CUDA_HOME; load/export the CUDA 12.9 toolkit")
cuda_nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
if not cuda_nvcc.is_file():
    raise SystemExit(f"CUDA_HOME does not contain nvcc: {cuda_nvcc}")
nvcc_version = subprocess.run(
    [str(cuda_nvcc), "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout
if re.search(r"release 12\.9(?:[, ]|$)", nvcc_version) is None:
    raise SystemExit(f"CUDA_HOME nvcc is not CUDA 12.9: {cuda_nvcc}\n{nvcc_version}")
print(f"Python {sys.version.split()[0]}; torch {torch.__version__}; FAISS {faiss.__version__}; CUDA_HOME={CUDA_HOME}")
PY

echo "== ColBERT CUDA extension smoke test =="
COLBERT_GPU=${COLBERT_GPU:-4}
PATH="$ROOT/.venv-pilot/bin:$PATH" CUDA_VISIBLE_DEVICES="$COLBERT_GPU" \
  "$PILOT_PYTHON" - <<'PY'
import torch
from colbert.indexing.codecs.residual import ResidualCodec

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        "ColBERT extension smoke test requires exactly one visible CUDA GPU; "
        f"found {torch.cuda.device_count()}"
    )
ResidualCodec.try_load_torch_extensions(True)
if not getattr(ResidualCodec, "loaded_extensions", False):
    raise SystemExit("ColBERT CUDA extensions did not report a successful load")
print(f"ColBERT CUDA extensions loaded on {torch.cuda.get_device_name(0)}")
PY

echo "== vLLM CUDA/Python stack =="
"$VLLM_PYTHON" - <<'PY'
import sys

import torch
import vllm
import vllm._C  # noqa: F401
from vllm import envs

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"vLLM Python must be 3.12, found {sys.version.split()[0]}")
if vllm.__version__ != "0.19.0" or envs.VLLM_MAIN_CUDA_VERSION != "12.9":
    raise SystemExit(
        f"vLLM build mismatch: version={vllm.__version__}, CUDA={envs.VLLM_MAIN_CUDA_VERSION}"
    )
if torch.version.cuda != "12.9" or not torch.cuda.is_available():
    raise SystemExit(f"vLLM torch CUDA mismatch: torch={torch.__version__}, CUDA={torch.version.cuda}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"vLLM torch sees {torch.cuda.device_count()} GPUs, expected 8")
torch.ones(1, device="cuda").mul_(2)
print(f"Python {sys.version.split()[0]}; vLLM {vllm.__version__}; torch {torch.__version__}")
PY

echo "== Java / retriever imports =="
source "$ROOT/scripts/lib/bootstrap_java.sh"
ensure_java "$ROOT"
"$PILOT_PYTHON" - <<'PY'
import sys
from pathlib import Path

from stackpilot.ragatouille_compat import install_langchain_retriever_compat

install_langchain_retriever_compat()
import psutil  # noqa: F401
from fast_pytorch_kmeans import KMeans  # noqa: F401
from colbert import Indexer, Searcher  # noqa: F401
from pyserini.index.lucene import IndexReader  # noqa: F401
from pyserini.search.lucene import LuceneSearcher  # noqa: F401
from ragatouille import RAGPretrainedModel  # noqa: F401

sys.path.insert(0, str(Path.cwd() / "upstream" / "Search-R1"))
from search_r1.search import index_builder, retrieval_server  # noqa: E402,F401

print(
    "Pyserini, psutil, fast-pytorch-kmeans, ColBERT, RAGatouille, "
    "and Search-R1 imports passed"
)
PY

EXPECTED_SEARCH_R1=598e61bd1d36895726d28a8d06b3a15bed19f5d3
if [[ ! -d "$ROOT/upstream/Search-R1/.git" ]]; then
  echo "Missing upstream/Search-R1. Run: bash scripts/bootstrap.sh" >&2
  exit 1
fi
ACTUAL_SEARCH_R1=$(git -C "$ROOT/upstream/Search-R1" rev-parse HEAD)
if [[ "$ACTUAL_SEARCH_R1" != "$EXPECTED_SEARCH_R1" ]]; then
  echo "Search-R1 commit mismatch: $ACTUAL_SEARCH_R1" >&2
  exit 1
fi
echo "Search-R1: $ACTUAL_SEARCH_R1"

SHM_KIB=$(df -Pk /dev/shm | awk 'NR == 2 { print $4 }')
if [[ -z "$SHM_KIB" || "$SHM_KIB" -lt 1048576 ]]; then
  echo "/dev/shm needs at least 1 GiB free for multi-GPU vLLM; found ${SHM_KIB:-unknown} KiB." >&2
  exit 1
fi
echo "/dev/shm free: $((SHM_KIB / 1024)) MiB"

for port in 8001 8002 8003 9000; do
  if port_is_open "$PILOT_PYTHON" "$port"; then
    if [[ ${ALLOW_USED_PORTS:-0} == 1 ]]; then
      echo "Port $port is already in use (allowed)."
    else
      echo "Port $port is already in use. Run: bash scripts/stop_servers.sh" >&2
      exit 1
    fi
  else
    echo "Port $port is free."
  fi
done

MODEL_PATH_WAS_SET=0
if [[ -n "${MODEL_PATH+x}" ]]; then
  MODEL_PATH_WAS_SET=1
fi
MODEL_PATH=${MODEL_PATH:-${MODEL:-Qwen/Qwen2.5-7B-Instruct}}
if [[ $MODEL_PATH_WAS_SET -eq 1 || "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || -e "$MODEL_PATH" ]]; then
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "MODEL_PATH is not a valid local Hugging Face model directory: $MODEL_PATH" >&2
    exit 1
  fi
  echo "Local Qwen model: $MODEL_PATH"
else
  echo "Qwen will be resolved from Hugging Face: $MODEL_PATH"
fi

echo "Preflight passed for Linux / CUDA 12.9 / 8x H100."

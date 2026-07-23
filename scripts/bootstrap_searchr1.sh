#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p upstream work logs

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
SEARCH_R1=$ROOT/upstream/Search-R1
RUNTIME_PATCH=$ROOT/searchr1_stage2/searchr1-runtime.patch

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Search-R1 bootstrap requires Linux x86_64." >&2
  exit 1
fi
for command_name in git nvcc g++; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Required command is missing: $command_name" >&2
    exit 1
  }
done
if ! command -v "$PYTHON_REQUEST" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_REQUEST" >&2
  exit 1
fi
if ! nvcc --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "The host CUDA toolkit must be 12.9; nvcc reports:" >&2
  nvcc --version >&2 || true
  exit 1
fi
NVCC_REAL=$(readlink -f "$(command -v nvcc)")
DETECTED_CUDA_HOME=$(cd "$(dirname "$NVCC_REAL")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-$DETECTED_CUDA_HOME}
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]] || \
  ! "$CUDA_HOME/bin/nvcc" --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "CUDA_HOME must point at the CUDA 12.9 toolkit; got: $CUDA_HOME" >&2
  exit 1
fi
export CUDA_HOME
echo "CUDA_HOME=$CUDA_HOME"

PYTHON_BASE=$("$PYTHON_REQUEST" -c \
  'import os,sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')
"$PYTHON_BASE" - <<'PY'
import sys
import sysconfig
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required; found {sys.version.split()[0]}")
header = Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.is_file():
    raise SystemExit(f"Python development headers are missing: {header}")
PY

if [[ ! -d "$SEARCH_R1/.git" ]]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git "$SEARCH_R1"
fi
if ! git -C "$SEARCH_R1" cat-file -e "$SEARCH_R1_COMMIT^{commit}" 2>/dev/null; then
  git -C "$SEARCH_R1" fetch origin "$SEARCH_R1_COMMIT"
fi
if [[ "$(git -C "$SEARCH_R1" rev-parse HEAD)" != "$SEARCH_R1_COMMIT" ]]; then
  git -C "$SEARCH_R1" checkout --detach "$SEARCH_R1_COMMIT"
fi

# The pinned trainer issued unbounded localhost retrieval calls. Apply the
# small idempotent runtime patch without resetting any unrelated user edits.
if git -C "$SEARCH_R1" apply --unidiff-zero --reverse --check "$RUNTIME_PATCH" >/dev/null 2>&1; then
  echo "Search-R1 retrieval-timeout patch is already applied."
elif git -C "$SEARCH_R1" apply --unidiff-zero --check "$RUNTIME_PATCH" >/dev/null 2>&1; then
  git -C "$SEARCH_R1" apply --unidiff-zero "$RUNTIME_PATCH"
  echo "Applied Search-R1 retrieval-timeout patch."
else
  echo "Unable to apply $RUNTIME_PATCH cleanly to pinned Search-R1." >&2
  echo "Preserving the upstream checkout; inspect its local changes." >&2
  exit 1
fi

# Apply the hard-RQ0 rollout-seed hook before any Stage-2 signature is computed.
# With RQ0_SEED unset it preserves Stage-2's original seed=0 behavior, while
# keeping the upstream dirty-tree fingerprint stable across full-pipeline reruns.
"$PYTHON_BASE" "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  --search-r1-root "$SEARCH_R1"

source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
"$UV_BIN" venv --clear --no-project --python "$PYTHON_BASE" .venv-searchr1
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python

# Search-R1's embedded veRL 0.1 is API-coupled to vLLM 0.6.3, whose wheel
# requires torch 2.4/cu121. Keep this compatibility runtime isolated: the host
# driver/toolkit remains CUDA 12.9 and the pilot/vLLM serving envs remain cu129.
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_TORCH_BACKEND=cu121 \
UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
    --python "$SEARCH_R1_PYTHON" \
    -r requirements-searchr1.txt

# flash-attn must see the installed torch and the host CUDA compiler, so it is
# intentionally a second, non-isolated source build.
MAX_JOBS=${FLASH_ATTN_MAX_JOBS:-4} \
FLASH_ATTN_CUDA_ARCHS=90 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
    --python "$SEARCH_R1_PYTHON" \
    --no-build-isolation \
    flash-attn==2.8.3

UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
    --python "$SEARCH_R1_PYTHON" \
    --no-deps \
    -e "$SEARCH_R1"

"$UV_BIN" pip check --python "$SEARCH_R1_PYTHON"
"$SEARCH_R1_PYTHON" - <<'PY'
import importlib.metadata

import flash_attn
import ray
import tensordict
import torch
import transformers
import vllm
from flash_attn import flash_attn_func
from search_r1.llm_agent.generation import LLMGenerationManager  # noqa: F401
from verl.trainer.main_ppo import RewardManager  # noqa: F401

expected = {
    "torch": "2.4.0",
    "vllm": "0.6.3",
    "transformers": "4.47.1",
    "tensordict": "0.5.0",
    "ray": "2.42.1",
    "flash-attn": "2.8.3",
}
for package, version in expected.items():
    actual = importlib.metadata.version(package).split("+")[0]
    if actual != version:
        raise SystemExit(f"Expected {package} {version}, found {actual}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"Search-R1 compatibility torch must be cu121; found {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected eight CUDA GPUs, found {torch.cuda.device_count()}")
for index in range(8):
    if "H100" not in torch.cuda.get_device_name(index):
        raise SystemExit(f"GPU {index} is not an H100: {torch.cuda.get_device_name(index)}")

q = torch.randn(
    1, 16, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
out = flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
if out.shape != q.shape or not torch.isfinite(out).all():
    raise SystemExit("flash-attn H100 warm-up failed")
out.float().sum().backward()
if q.grad is None or not torch.isfinite(q.grad).all():
    raise SystemExit("flash-attn H100 backward warm-up failed")
print(
    f"Search-R1 runtime ready: torch {torch.__version__}, vLLM {vllm.__version__}, "
    f"flash-attn {flash_attn.__version__}, transformers {transformers.__version__}, "
    f"Ray {ray.__version__}; host is expected to remain CUDA 12.9."
)
PY

cat <<MSG
Search-R1 bootstrap complete.
Training interpreter: $SEARCH_R1_PYTHON
Pinned upstream: $SEARCH_R1_COMMIT
MSG

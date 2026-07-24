#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p upstream work logs
command -v flock >/dev/null 2>&1 || {
  echo "flock is required to serialize the Search-R1 environment bootstrap." >&2
  exit 1
}
exec {SEARCHR1_BOOTSTRAP_LOCK_FD}>"$ROOT/work/.bootstrap-searchr1.lock"
echo "Acquiring Search-R1 bootstrap lock."
flock "$SEARCHR1_BOOTSTRAP_LOCK_FD"
source "$ROOT/scripts/lib/bootstrap_env.sh"
validate_bootstrap_flag
SEARCHR1_DEFER_GPU_PROBE=${SEARCHR1_DEFER_GPU_PROBE:-0}
if [[ "$SEARCHR1_DEFER_GPU_PROBE" != 0 && "$SEARCHR1_DEFER_GPU_PROBE" != 1 ]]; then
  echo "SEARCHR1_DEFER_GPU_PROBE must be 0 or 1; got '$SEARCHR1_DEFER_GPU_PROBE'." >&2
  exit 2
fi

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}

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

if [[ ! -e "$SEARCH_R1/.git" ]]; then
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
bash "$ROOT/scripts/apply_searchr1_runtime_patch.sh"

# Apply the hard-RQ0 rollout-seed hook before any Stage-2 signature is computed.
# With RQ0_SEED unset it preserves Stage-2's original seed=0 behavior, while
# keeping the upstream dirty-tree fingerprint stable across full-pipeline reruns.
"$PYTHON_BASE" "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  --search-r1-root "$SEARCH_R1"
"$PYTHON_BASE" "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" \
  --search-r1-root "$SEARCH_R1"
"$PYTHON_BASE" "$ROOT/hard_rq0/patch_searchr1_validation.py" \
  --search-r1-root "$SEARCH_R1"
"$PYTHON_BASE" "$ROOT/hard_rq0/patch_searchr1_experiment_env.py" \
  --search-r1-root "$SEARCH_R1"

source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
CORE_ENV=searchr1
CORE_MARKER=$ROOT/.venv-searchr1/.stackpilot-bootstrap.json
CORE_SIGNATURE=$("$PYTHON_BASE" -m stackpilot.bootstrap_cache signature \
  --root "$ROOT" --environment "$CORE_ENV" --python "$PYTHON_BASE" \
  --input "$ROOT/requirements-searchr1.txt" \
  --input "$SEARCH_R1/pyproject.toml" \
  --input "$SEARCH_R1/verl/version/version" \
  --input "$ROOT/scripts/bootstrap_searchr1.sh" \
  --input "$ROOT/scripts/apply_searchr1_runtime_patch.sh" \
  --input "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  --input "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" \
  --input "$ROOT/hard_rq0/patch_searchr1_validation.py" \
  --input "$ROOT/hard_rq0/patch_searchr1_experiment_env.py" \
  --input "$ROOT/scripts/lib/bootstrap_env.sh" \
  --input "$ROOT/scripts/lib/bootstrap_uv.sh" \
  --value "uv=0.11.30" --value "torch_backend=cu121" \
  --value "search_r1_commit=$SEARCH_R1_COMMIT")
FLASH_ENV=searchr1-flash-attn
FLASH_MARKER=$ROOT/.venv-searchr1/.stackpilot-flash-attn.json
NVCC_SIGNATURE=$("$CUDA_HOME/bin/nvcc" --version | tr '\n' ' ')
GXX_SIGNATURE=$(g++ -dumpfullversion -dumpversion)
FLASH_SIGNATURE=$("$PYTHON_BASE" -m stackpilot.bootstrap_cache signature \
  --root "$ROOT" --environment "$FLASH_ENV" --python "$PYTHON_BASE" \
  --input "$ROOT/requirements-searchr1.txt" \
  --input "$ROOT/scripts/bootstrap_searchr1.sh" \
  --input "$ROOT/scripts/lib/bootstrap_env.sh" \
  --value "flash_attn=2.8.3" --value "cuda_arch=90" \
  --value "cuda_home=$CUDA_HOME" --value "nvcc=$NVCC_SIGNATURE" \
  --value "gxx=$GXX_SIGNATURE" --value "torch_backend=cu121")

install_searchr1_core() {
  # Search-R1's embedded veRL is API-coupled to vLLM 0.6.3/torch cu121.
  # Reuse local wheels first and download only a missing dependency delta.
  UV_DEFAULT_INDEX=https://pypi.org/simple \
  UV_TORCH_BACKEND=cu121 \
  UV_LINK_MODE=copy \
    uv_pip_install_cached_first "$UV_BIN" \
    --python "$SEARCH_R1_PYTHON" \
    -r requirements-searchr1.txt

  UV_LINK_MODE=copy \
    uv_pip_install_cached_first "$UV_BIN" \
      --python "$SEARCH_R1_PYTHON" --no-deps -e "$SEARCH_R1"
}

install_searchr1_flash() {
  # Rebuild only when explicitly forced or when the existing native extension
  # cannot be imported; a compatible build is adopted with the current
  # torch/CUDA/compiler signature.
  MAX_JOBS=${FLASH_ATTN_MAX_JOBS:-4} \
  FLASH_ATTN_CUDA_ARCHS=90 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  UV_LINK_MODE=copy \
    uv_pip_install_cached_first "$UV_BIN" \
    --python "$SEARCH_R1_PYTHON" \
    --no-build-isolation \
    --reinstall-package flash-attn \
    flash-attn==2.8.3
}

searchr1_core_is_valid() {
  [[ -x "$SEARCH_R1_PYTHON" ]] || return 1
  "$SEARCH_R1_PYTHON" "$ROOT/stackpilot/bootstrap_cache.py" \
    verify-requirements --requirements "$ROOT/requirements-searchr1.txt" \
    --require 'verl==0.1' --editable "verl=$SEARCH_R1" >/dev/null || return 1
  "$UV_BIN" pip check --python "$SEARCH_R1_PYTHON" || return 1
  "$SEARCH_R1_PYTHON" - <<'PY' || return 1
import importlib.metadata

import ray
import tensordict
import torch
import transformers
import vllm
import vllm._C  # noqa: F401

expected = {
    "torch": "2.4.0",
    "vllm": "0.6.3",
    "transformers": "4.47.1",
    "tensordict": "0.5.0",
    "ray": "2.42.1",
}
actual_versions = {}
for package, version in expected.items():
    actual = importlib.metadata.version(package).split("+")[0]
    actual_versions[package] = actual
    if actual != version:
        raise SystemExit(f"Expected {package} {version}, found {actual}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"Search-R1 compatibility torch must be cu121; found {torch.version.cuda}")
print(
    f"Search-R1 core imports passed: torch {torch.__version__}, "
    f"vLLM {vllm.__version__}, transformers {transformers.__version__}, "
    f"Ray {ray.__version__}, tensordict {actual_versions['tensordict']}"
)
PY
}

searchr1_flash_is_valid() {
  [[ -x "$SEARCH_R1_PYTHON" ]] || return 1
  "$SEARCH_R1_PYTHON" - <<'PY' || return 1
import importlib.metadata

import flash_attn
from flash_attn import flash_attn_func  # noqa: F401

actual = importlib.metadata.version("flash-attn").split("+")[0]
if actual != "2.8.3":
    raise SystemExit(f"Expected flash-attn 2.8.3, found {actual}")
print(f"flash-attn native import passed: {flash_attn.__version__}")
PY
}

searchr1_combined_imports_are_valid() {
  "$SEARCH_R1_PYTHON" - <<'PY' || return 1
from search_r1.llm_agent.generation import LLMGenerationManager  # noqa: F401
from verl.trainer.main_ppo import RewardManager  # noqa: F401

print("Search-R1/veRL training imports passed.")
PY
}

core_hit=0
if [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && \
   bootstrap_marker_matches \
     "$PYTHON_BASE" "$CORE_MARKER" "$CORE_ENV" "$CORE_SIGNATURE" && \
   searchr1_core_is_valid; then
  core_hit=1
  echo "Reusing verified Search-R1 core environment."
elif [[ ${FORCE_BOOTSTRAP:-0} != 1 && -x "$SEARCH_R1_PYTHON" ]] && \
     bootstrap_interpreter_compatible \
       "$PYTHON_BASE" "$SEARCH_R1_PYTHON" "$PYTHON_BASE" && \
     searchr1_core_is_valid; then
  core_hit=1
  write_bootstrap_marker \
    "$SEARCH_R1_PYTHON" "$CORE_MARKER" "$CORE_ENV" "$CORE_SIGNATURE"
  echo "Adopted the existing verified Search-R1 core without reinstalling packages."
fi

if [[ $core_hit -ne 1 ]]; then
  rm -f -- "$CORE_MARKER"
  prepare_cached_venv \
    "$PYTHON_BASE" "$PYTHON_BASE" "$SEARCH_R1_PYTHON" \
    "$ROOT/.venv-searchr1" "$UV_BIN"
  install_searchr1_core
  if ! searchr1_core_is_valid; then
    echo "Incremental Search-R1 repair did not validate; rebuilding it once." >&2
    "$UV_BIN" venv --clear --no-project --python "$PYTHON_BASE" \
      "$ROOT/.venv-searchr1"
    install_searchr1_core
    searchr1_core_is_valid
  fi
  write_bootstrap_marker \
    "$SEARCH_R1_PYTHON" "$CORE_MARKER" "$CORE_ENV" "$CORE_SIGNATURE"
fi

flash_hit=0
if [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && \
   bootstrap_marker_matches \
     "$PYTHON_BASE" "$FLASH_MARKER" "$FLASH_ENV" "$FLASH_SIGNATURE" && \
   searchr1_flash_is_valid; then
  flash_hit=1
  echo "Reusing verified flash-attn build."
elif [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && \
     searchr1_flash_is_valid; then
  flash_hit=1
  write_bootstrap_marker \
    "$SEARCH_R1_PYTHON" "$FLASH_MARKER" "$FLASH_ENV" "$FLASH_SIGNATURE"
  echo "Adopted the existing compatible flash-attn build without recompiling it."
fi

if [[ $flash_hit -ne 1 ]]; then
  rm -f -- "$FLASH_MARKER"
  install_searchr1_flash
  searchr1_flash_is_valid
  write_bootstrap_marker \
    "$SEARCH_R1_PYTHON" "$FLASH_MARKER" "$FLASH_ENV" "$FLASH_SIGNATURE"
fi

searchr1_combined_imports_are_valid

"$UV_BIN" pip check --python "$SEARCH_R1_PYTHON"
if [[ "$SEARCHR1_DEFER_GPU_PROBE" == 1 ]]; then
  echo "Deferring the H100 flash-attn warm-up to the Stage-2 preflight."
else
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
fi

cat <<MSG
Search-R1 bootstrap complete.
Training interpreter: $SEARCH_R1_PYTHON
Pinned upstream: $SEARCH_R1_COMMIT
MSG

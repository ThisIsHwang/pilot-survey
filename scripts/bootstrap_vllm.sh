#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/bootstrap_env.sh"
validate_bootstrap_flag

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
# vLLM 0.19.0 is intentionally pinned: its PyPI wheel is natively built for
# CUDA 12.9. vLLM 0.20+ switched the default PyPI binary to CUDA 13, whose
# explicit cu129 alternative is hosted on GitHub release assets blocked by the
# target cluster's egress proxy.
CUDA_BACKEND=cu129

if ! command -v "$PYTHON_REQUEST" >/dev/null 2>&1; then
  echo "Python 3.12 executable not found: $PYTHON_REQUEST" >&2
  echo "Install Python 3.12 or set PYTHON_BIN to a Python 3.12 executable." >&2
  exit 1
fi
PYTHON_BIN=$("$PYTHON_REQUEST" -c \
  'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Resolved base Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap targets Linux x86_64; found $(uname -s) $(uname -m)." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 is required; found {sys.version.split()[0]}. "
        "Set PYTHON_BIN to a Python 3.12 interpreter."
    )
libc_name, libc_version = platform.libc_ver()
if libc_name != "glibc":
    raise SystemExit(f"glibc is required; found {libc_name or 'unknown'} {libc_version}")
parts = tuple(int(part) for part in libc_version.split(".")[:2])
if parts < (2, 31):
    raise SystemExit(f"vLLM's PyPI wheel requires glibc 2.31+; found {libc_version}.")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; run bootstrap on the allocated 8x H100 node." >&2
  exit 1
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne 8 ]]; then
  echo "Exactly 8 visible H100 GPUs are required; found ${#GPU_NAMES[@]}." >&2
  exit 1
fi
for GPU_NAME in "${GPU_NAMES[@]}"; do
  if [[ "$GPU_NAME" != *H100* ]]; then
    echo "Every visible GPU must be an H100; found: $GPU_NAME" >&2
    exit 1
  fi
done

source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
BOOTSTRAP_ENV=vllm
BOOTSTRAP_MARKER=$ROOT/.venv-vllm/.stackpilot-bootstrap.json
BOOTSTRAP_SIGNATURE=$("$PYTHON_BIN" -m stackpilot.bootstrap_cache signature \
  --root "$ROOT" --environment "$BOOTSTRAP_ENV" --python "$PYTHON_BIN" \
  --input "$ROOT/requirements-vllm.txt" \
  --input "$ROOT/scripts/bootstrap_vllm.sh" \
  --input "$ROOT/scripts/lib/bootstrap_env.sh" \
  --input "$ROOT/scripts/lib/bootstrap_uv.sh" \
  --value "uv=0.11.30" --value "torch_backend=$CUDA_BACKEND")
VENV_PYTHON=$ROOT/.venv-vllm/bin/python

install_vllm_environment() {
  # uv reuses compatible artifacts in the persistent cache and downloads only
  # packages absent from it.
  UV_DEFAULT_INDEX=https://pypi.org/simple \
  UV_LINK_MODE=copy \
    uv_pip_install_cached_first "$UV_BIN" \
    --python "$VENV_PYTHON" \
    --torch-backend "$CUDA_BACKEND" \
    -r requirements-vllm.txt
}

vllm_install_is_valid() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" "$ROOT/stackpilot/bootstrap_cache.py" verify-requirements \
    --requirements "$ROOT/requirements-vllm.txt" >/dev/null || return 1
  "$UV_BIN" pip check --python "$VENV_PYTHON" || return 1
  "$VENV_PYTHON" - <<'PY' || return 1
import torch
import vllm
import vllm._C  # noqa: F401
from vllm import envs

if vllm.__version__ != "0.19.0":
    raise SystemExit(f"Expected vLLM 0.19.0, found {vllm.__version__}.")
if torch.version.cuda != "12.9":
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch wheel, found torch {torch.__version__} "
        f"with CUDA {torch.version.cuda!r}."
    )
if envs.VLLM_MAIN_CUDA_VERSION != "12.9":
    raise SystemExit(f"vLLM binary targets CUDA {envs.VLLM_MAIN_CUDA_VERSION}, not 12.9.")
print(
    f"vLLM native imports passed: vLLM {vllm.__version__}; "
    f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}"
)
PY
}

vllm_hardware_is_valid() {
  "$VENV_PYTHON" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("vLLM's PyTorch cannot initialize CUDA on this node.")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected 8 visible H100s, found {torch.cuda.device_count()} GPUs.")
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    if "H100" not in props.name:
        raise SystemExit(f"GPU {index} is not an H100: {props.name}")
    if props.total_memory < 70 * 1024**3:
        raise SystemExit(f"GPU {index} appears to be a MIG slice ({props.total_memory / 1024**3:.1f} GiB).")
torch.ones(1, device="cuda").mul_(2)
print(
    f"vLLM {vllm.__version__}; PyTorch {torch.__version__}; "
    f"wheel CUDA {torch.version.cuda}; visible H100s: {torch.cuda.device_count()}"
)
PY
}

cache_hit=0
if [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && \
   bootstrap_marker_matches \
     "$PYTHON_BIN" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE" && \
   vllm_install_is_valid; then
  cache_hit=1
  echo "Reusing verified vLLM environment: $ROOT/.venv-vllm"
elif [[ ${FORCE_BOOTSTRAP:-0} != 1 && -x "$VENV_PYTHON" ]] && \
     bootstrap_interpreter_compatible "$PYTHON_BIN" "$VENV_PYTHON" "$PYTHON_BIN" && \
     vllm_install_is_valid; then
  cache_hit=1
  write_bootstrap_marker \
    "$VENV_PYTHON" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE"
  echo "Adopted the existing verified vLLM environment without reinstalling packages."
fi

if [[ $cache_hit -ne 1 ]]; then
  rm -f -- "$BOOTSTRAP_MARKER"
  prepare_cached_venv \
    "$PYTHON_BIN" "$PYTHON_BIN" "$VENV_PYTHON" "$ROOT/.venv-vllm" "$UV_BIN"
  install_vllm_environment
  if ! vllm_install_is_valid; then
    echo "Incremental vLLM repair did not validate; rebuilding it once." >&2
    "$UV_BIN" venv --clear --no-project --python "$PYTHON_BIN" "$ROOT/.venv-vllm"
    install_vllm_environment
    vllm_install_is_valid
  fi
  write_bootstrap_marker \
    "$VENV_PYTHON" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE"
fi

# A driver/GPU problem is reported without deleting the verified environment.
vllm_hardware_is_valid

echo "vLLM environment ready: source $ROOT/.venv-vllm/bin/activate"

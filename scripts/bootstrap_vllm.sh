#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_VERSION=0.25.1
CUDA_BACKEND=cu129

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Install Python 3.10-3.14 or set PYTHON_BIN (for example, PYTHON_BIN=python3.12)." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit(
        f"Python 3.10-3.14 is required; found {sys.version.split()[0]}. "
        "Set PYTHON_BIN to a supported interpreter."
    )
PY

case "$(uname -m)" in
  x86_64) WHEEL_ARCH=x86_64 ;;
  aarch64|arm64) WHEEL_ARCH=aarch64 ;;
  *)
    echo "Unsupported architecture for the prebuilt vLLM CUDA wheel: $(uname -m)" >&2
    exit 1
    ;;
esac

# vLLM's CUDA 12.9 release wheels require glibc 2.28 or newer.
"$PYTHON_BIN" - <<'PY'
import platform

libc_name, libc_version = platform.libc_ver()
if libc_name == "glibc":
    parts = tuple(int(part) for part in libc_version.split(".")[:2])
    if parts < (2, 28):
        raise SystemExit(f"glibc 2.28+ is required; found {libc_version}.")
PY

# Build the environment with uv, bypassing stdlib venv/ensurepip entirely.
source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
"$UV_BIN" venv \
  --clear \
  --no-project \
  --python "$PYTHON_BIN" \
  .venv-vllm
VENV_PYTHON=$ROOT/.venv-vllm/bin/python

VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_BACKEND}-cp38-abi3-manylinux_2_28_${WHEEL_ARCH}.whl"

# Install the CUDA-matched vLLM wheel and let it select its exact compatible
# torch/torchvision versions in the same resolver transaction.
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_TORCH_BACKEND=$CUDA_BACKEND \
  "$UV_BIN" pip install \
    --python "$VENV_PYTHON" \
    "$VLLM_WHEEL"

"$UV_BIN" pip check --python "$VENV_PYTHON"
"$VENV_PYTHON" - <<'PY'
import torch
import vllm

if torch.version.cuda != "12.9":
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch wheel, found torch {torch.__version__} "
        f"with CUDA {torch.version.cuda!r}."
    )
print(f"vLLM {vllm.__version__}; PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")
PY

echo "vLLM environment ready: source $ROOT/.venv-vllm/bin/activate"

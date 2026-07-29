#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
command -v "$PYTHON_REQUEST" >/dev/null 2>&1 || {
  echo "Python 3.12 is required (set PYTHON_BIN if needed)." >&2
  exit 1
}
"$PYTHON_REQUEST" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"TRACE requires Python 3.12; found {sys.version.split()[0]}")
PY
command -v nvcc >/dev/null 2>&1 || { echo "nvcc is required." >&2; exit 1; }
nvcc --version | grep -Eq 'release 12\.9([, ]|$)' || {
  echo "TRACE requires the CUDA 12.9 toolkit." >&2
  nvcc --version >&2 || true
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 8 ]] || {
  echo "TRACE is configured for exactly 8 visible GPUs; found ${#GPU_NAMES[@]}." >&2
  exit 1
}
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == *H100* ]] || { echo "Expected H100, found: $name" >&2; exit 1; }
done
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
  grep -Eqv '^[[:space:]]*Disabled[[:space:]]*$'; then
  echo "MIG must be disabled on all eight H100 GPUs." >&2
  exit 1
fi

source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
VENV=$ROOT/.venv-trace
VENV_PYTHON=$VENV/bin/python
BASE_PYTHON=$("$PYTHON_REQUEST" -c 'import os,sys; print(os.path.realpath(getattr(sys,"_base_executable",sys.executable)))')

rebuild=0
if [[ ${FORCE_BOOTSTRAP:-0} == 1 || ! -x "$VENV_PYTHON" ]]; then
  rebuild=1
elif ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  rebuild=1
fi
if [[ $rebuild == 1 ]]; then
  "$UV_BIN" venv --clear --no-project --python "$BASE_PYTHON" "$VENV"
fi
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_TORCH_BACKEND=cu129 \
UV_LINK_MODE=copy \
  "$UV_BIN" pip install --python "$VENV_PYTHON" \
    -r requirements-trace.txt -e .

"$VENV_PYTHON" - <<'PY'
import accelerate
import peft
import torch
import transformers
if torch.version.cuda != "12.9":
    raise SystemExit(f"Expected CUDA 12.9 torch wheel, found {torch.version.cuda!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected 8 visible CUDA GPUs, found {torch.cuda.device_count()}")
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    if "H100" not in properties.name:
        raise SystemExit(f"GPU {index} is not an H100: {properties.name}")
    if properties.total_memory < 70 * 1024**3:
        raise SystemExit(
            f"GPU {index} appears to be a MIG slice: "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )
print(
    f"TRACE environment ready: torch={torch.__version__} cuda={torch.version.cuda} "
    f"transformers={transformers.__version__} peft={peft.__version__} "
    f"accelerate={accelerate.__version__}"
)
PY

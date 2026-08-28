#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/bootstrap_uv.sh"

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
command -v "$PYTHON_REQUEST" >/dev/null 2>&1 || {
  echo "Python 3.12 is required. Set PYTHON_BIN when it is not on PATH." >&2
  exit 1
}
PYTHON_BASE=$("$PYTHON_REQUEST" -c 'import os,sys; print(os.path.realpath(getattr(sys,"_base_executable",sys.executable)))')
"$PYTHON_BASE" -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
ensure_uv "$ROOT"

VENV=$ROOT/.venv-qwen35
VENV_PYTHON=$VENV/bin/python

qwen35_environment_is_valid() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib
import torch
import transformers
import peft
import accelerate

assert transformers.__version__ == "5.16.1", transformers.__version__
assert peft.__version__ == "0.20.0", peft.__version__
assert accelerate.__version__ == "1.14.0", accelerate.__version__
assert torch.__version__.split("+")[0] == "2.11.0", torch.__version__
module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
assert hasattr(module, "Qwen3_5ForCausalLM")
PY
}

if [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && qwen35_environment_is_valid; then
  echo "Reusing verified Qwen3.5 environment: $VENV"
  exit 0
fi

"$UV_BIN" venv --clear --no-project --python "$PYTHON_BASE" "$VENV"
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_TORCH_BACKEND=cu129 \
UV_LINK_MODE=copy \
  uv_pip_install_cached_first "$UV_BIN" \
  --python "$VENV_PYTHON" \
  -r "$ROOT/requirements-qwen35.txt"
"$UV_BIN" pip install --python "$VENV_PYTHON" --no-deps -e "$ROOT"

qwen35_environment_is_valid || {
  echo "The Qwen3.5 environment failed validation." >&2
  exit 1
}

STACKPILOT_QWEN35_NO_THINK=1 \
PYTHONPATH="$ROOT/query_credit/qwen35_site:$ROOT" \
  "$VENV_PYTHON" - <<'PY'
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

assert getattr(
    PreTrainedTokenizerBase.apply_chat_template,
    "_stackpilot_no_think",
    False,
), "sitecustomize did not install the no-thinking template guard"
print("Qwen3.5 text-only environment ready; local chat templates force enable_thinking=False.")
PY

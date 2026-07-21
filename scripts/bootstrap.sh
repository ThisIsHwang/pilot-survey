#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p upstream work logs

PYTHON_BIN=${PYTHON_BIN:-python}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
CUDA_BACKEND=cu129

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Install Python 3.10-3.14 or set PYTHON_BIN (for example, PYTHON_BIN=python3.12)." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The prebuilt faiss-gpu wheel requires Linux x86_64; found $(uname -s) $(uname -m)." >&2
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

SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
if [[ ! -d upstream/Search-R1/.git ]]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git upstream/Search-R1
fi
git -C upstream/Search-R1 fetch --all --tags
git -C upstream/Search-R1 checkout "$SEARCH_R1_COMMIT"

# Build the environment with uv, bypassing stdlib venv/ensurepip entirely.
source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
"$UV_BIN" venv \
  --clear \
  --no-project \
  --python "$PYTHON_BIN" \
  .venv-pilot
VENV_PYTHON=$ROOT/.venv-pilot/bin/python

# Resolve torch and all pilot dependencies together. uv routes PyTorch packages only
# to the CUDA 12.9 index, while normal packages continue to come from PyPI.
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_TORCH_BACKEND=$CUDA_BACKEND \
  "$UV_BIN" pip install \
    --python "$VENV_PYTHON" \
    "torch==$TORCH_VERSION" \
    -r requirements-pilot.txt \
    -e compat/faiss-cpu-gpu-shim \
    -e .

"$UV_BIN" pip check --python "$VENV_PYTHON"
"$VENV_PYTHON" - <<'PY'
import faiss
import torch

if torch.version.cuda != "12.9":
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch wheel, found torch {torch.__version__} "
        f"with CUDA {torch.version.cuda!r}."
    )
print(f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}; GPU available: {torch.cuda.is_available()}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("The installed FAISS module has no GPU support.")
print(f"FAISS {faiss.__version__}; GPU resources visible: {faiss.get_num_gpus()}")
PY

if ! command -v java >/dev/null 2>&1; then
  echo "WARNING: Java is missing. Install OpenJDK 21 before building the BM25 index." >&2
else
  java -version 2>&1 | head -1
fi

cat <<MSG
Bootstrap complete.
Activate with: source $ROOT/.venv-pilot/bin/activate
Search-R1 pinned at: $SEARCH_R1_COMMIT

For Search-R1's native GRPO stage, create its official separate environment later;
the zero-shot pilot deliberately keeps the dependency surface smaller.
MSG

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p upstream work logs
source "$ROOT/scripts/lib/bootstrap_env.sh"
validate_bootstrap_flag

PYTHON_REQUEST=${PYTHON_BIN:-python3.12}
CUDA_BACKEND=cu129

if ! command -v "$PYTHON_REQUEST" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_REQUEST" >&2
  echo "Install Python 3.12 or set PYTHON_BIN to a Python 3.12 executable." >&2
  exit 1
fi
# If the caller already activated .venv-pilot, do not ask uv to clear the
# environment that contains its own source interpreter.
PYTHON_BIN=$("$PYTHON_REQUEST" -c \
  'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Resolved base Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The prebuilt faiss-gpu wheel requires Linux x86_64; found $(uname -s) $(uname -m)." >&2
  exit 1
fi
for command_name in git nvcc g++; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    echo "Load the CUDA 12.9 development toolkit before bootstrapping." >&2
    exit 1
  fi
done
if ! nvcc --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "Expected the CUDA 12.9 toolkit; nvcc reports:" >&2
  nvcc --version >&2 || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys
import sysconfig
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 is required; found {sys.version.split()[0]}. "
        "Set PYTHON_BIN to a Python 3.12 interpreter."
    )
libc_name, libc_version = platform.libc_ver()
if libc_name != "glibc":
    raise SystemExit(f"glibc is required; found {libc_name or 'unknown'} {libc_version}")
try:
    libc_parts = tuple(int(part) for part in libc_version.split(".")[:2])
except ValueError as error:
    raise SystemExit(f"Unable to parse glibc version: {libc_version!r}") from error
if libc_parts < (2, 28):
    raise SystemExit(f"The pilot binary wheels require glibc 2.28+; found {libc_version}.")
python_header = Path(sysconfig.get_paths()["include"]) / "Python.h"
if not python_header.is_file():
    raise SystemExit(
        f"Python 3.12 development headers are required for ColBERT; missing {python_header}. "
        "Install python3.12-dev (or the equivalent package for this distribution)."
    )
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; run bootstrap on the allocated 8x H100 node." >&2
  exit 1
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne 8 ]]; then
  echo "Exactly 8 visible H100 GPUs are required; found ${#GPU_NAMES[@]}." >&2
  echo "Check the job allocation and CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
for GPU_NAME in "${GPU_NAMES[@]}"; do
  if [[ "$GPU_NAME" != *H100* ]]; then
    echo "Every visible GPU must be an H100; found: $GPU_NAME" >&2
    exit 1
  fi
done
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | \
  grep -Eqv '^[[:space:]]*Disabled[[:space:]]*$'; then
  echo "MIG must be disabled on all eight H100 GPUs." >&2
  exit 1
fi

SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
if [[ ! -d upstream/Search-R1/.git ]]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git upstream/Search-R1
fi
if ! git -C upstream/Search-R1 cat-file -e "$SEARCH_R1_COMMIT^{commit}" 2>/dev/null; then
  git -C upstream/Search-R1 fetch origin "$SEARCH_R1_COMMIT"
fi
git -C upstream/Search-R1 checkout --detach "$SEARCH_R1_COMMIT"

# Build the environment with uv, bypassing stdlib venv/ensurepip entirely.
source "$ROOT/scripts/lib/bootstrap_uv.sh"
ensure_uv "$ROOT"
BOOTSTRAP_ENV=pilot
BOOTSTRAP_MARKER=$ROOT/.venv-pilot/.stackpilot-bootstrap.json
BOOTSTRAP_SIGNATURE=$("$PYTHON_BIN" -m stackpilot.bootstrap_cache signature \
  --root "$ROOT" --environment "$BOOTSTRAP_ENV" --python "$PYTHON_BIN" \
  --input "$ROOT/requirements-pilot.txt" \
  --input "$ROOT/pyproject.toml" \
  --input "$ROOT/compat/faiss-cpu-gpu-shim/pyproject.toml" \
  --input "$ROOT/compat/llama-index-core-shim/pyproject.toml" \
  --input "$ROOT/scripts/bootstrap.sh" \
  --input "$ROOT/scripts/lib/bootstrap_env.sh" \
  --input "$ROOT/scripts/lib/bootstrap_java.sh" \
  --input "$ROOT/scripts/lib/bootstrap_uv.sh" \
  --value "uv=0.11.30" --value "torch_backend=$CUDA_BACKEND")
VENV_PYTHON=$ROOT/.venv-pilot/bin/python

install_pilot_environment() {
  # Resolve torch and all pilot dependencies together. uv reuses compatible
  # artifacts in the persistent cache and fetches only cache misses.
  UV_DEFAULT_INDEX=https://pypi.org/simple \
  UV_TORCH_BACKEND=$CUDA_BACKEND \
  UV_LINK_MODE=copy \
    uv_pip_install_cached_first "$UV_BIN" \
    --python "$VENV_PYTHON" \
    -r requirements-pilot.txt \
    -e compat/faiss-cpu-gpu-shim \
    -e compat/llama-index-core-shim \
    -e .
}

pilot_install_is_valid() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" "$ROOT/stackpilot/bootstrap_cache.py" verify-requirements \
    --requirements "$ROOT/requirements-pilot.txt" \
    --require 'faiss-cpu==1.14.3' \
    --require 'llama-index==0.14.23' \
    --require 'stack-adapt-pilot==0.1.0' \
    --editable "faiss-cpu=$ROOT/compat/faiss-cpu-gpu-shim" \
    --editable "llama-index=$ROOT/compat/llama-index-core-shim" \
    --editable "stack-adapt-pilot=$ROOT" >/dev/null || return 1
  "$UV_BIN" pip check --python "$VENV_PYTHON" || return 1
  "$VENV_PYTHON" - <<'PY' || return 1
import faiss
import torch

if torch.version.cuda != "12.9":
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch wheel, found torch {torch.__version__} "
        f"with CUDA {torch.version.cuda!r}."
    )
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("The installed FAISS module has no GPU support.")
print(f"Pilot native imports passed: PyTorch {torch.__version__}; FAISS {faiss.__version__}")
PY

  source "$ROOT/scripts/lib/bootstrap_java.sh"
  ensure_java "$ROOT" || return 1

  "$VENV_PYTHON" - <<'PY' || return 1
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
from stackpilot import (  # noqa: F401
    hard_assets,
    hard_policy_eval,
    hard_query_analysis,
    hard_query_report,
    hard_rq0_report,
    policy_eval,
    prepare_hard_rq0,
    query_stats,
    rq0_report,
    validate_hard_results,
)

sys.path.insert(0, str(Path.cwd() / "upstream" / "Search-R1"))
from search_r1.search import index_builder, retrieval_server  # noqa: E402,F401

print(
    "Pyserini, psutil, fast-pytorch-kmeans, ColBERT, RAGatouille, "
    "policy/RQ0/hard-RQ0 modules, and Search-R1 imports passed."
)
PY
}

pilot_hardware_is_valid() {
  "$VENV_PYTHON" - <<'PY'
import faiss
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize CUDA on this node.")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected 8 visible H100s, found {torch.cuda.device_count()} GPUs.")
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    if "H100" not in props.name:
        raise SystemExit(f"GPU {index} is not an H100: {props.name}")
    if props.total_memory < 70 * 1024**3:
        raise SystemExit(
            f"GPU {index} appears to be a MIG slice "
            f"({props.total_memory / 1024**3:.1f} GiB)."
        )
torch.ones(1, device="cuda").add_(1)
if faiss.get_num_gpus() != 8:
    raise SystemExit(f"FAISS sees {faiss.get_num_gpus()} GPUs; expected exactly 8.")
print(
    f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}; "
    f"visible H100s: {torch.cuda.device_count()}"
)
print(f"FAISS {faiss.__version__}; GPU resources visible: {faiss.get_num_gpus()}")
PY
}

cache_hit=0
if [[ ${FORCE_BOOTSTRAP:-0} != 1 ]] && \
   bootstrap_marker_matches \
     "$PYTHON_BIN" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE" && \
   pilot_install_is_valid; then
  cache_hit=1
  echo "Reusing verified pilot environment: $ROOT/.venv-pilot"
elif [[ ${FORCE_BOOTSTRAP:-0} != 1 && -x "$VENV_PYTHON" ]] && \
     bootstrap_interpreter_compatible "$PYTHON_BIN" "$VENV_PYTHON" "$PYTHON_BIN" && \
     pilot_install_is_valid; then
  cache_hit=1
  write_bootstrap_marker \
    "$VENV_PYTHON" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE"
  echo "Adopted the existing verified pilot environment without reinstalling packages."
fi

if [[ $cache_hit -ne 1 ]]; then
  rm -f -- "$BOOTSTRAP_MARKER"
  prepare_cached_venv \
    "$PYTHON_BIN" "$PYTHON_BIN" "$VENV_PYTHON" "$ROOT/.venv-pilot" "$UV_BIN"
  install_pilot_environment
  if ! pilot_install_is_valid; then
    echo "Incremental pilot-environment repair did not validate; rebuilding it once." >&2
    "$UV_BIN" venv --clear --no-project --python "$PYTHON_BIN" "$ROOT/.venv-pilot"
    install_pilot_environment
    pilot_install_is_valid
  fi
  write_bootstrap_marker \
    "$VENV_PYTHON" "$BOOTSTRAP_MARKER" "$BOOTSTRAP_ENV" "$BOOTSTRAP_SIGNATURE"
fi

# Hardware failures never invalidate or rebuild a verified package cache.
pilot_hardware_is_valid

cat <<MSG
Bootstrap complete.
Activate with: source $ROOT/.venv-pilot/bin/activate
Search-R1 pinned at: $SEARCH_R1_COMMIT

For Search-R1's native GRPO stage, run: bash scripts/bootstrap_searchr1.sh
The zero-shot pilot deliberately keeps that legacy training stack isolated.
MSG

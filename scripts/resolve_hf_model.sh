#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_REF=${1:?Usage: resolve_hf_model.sh MODEL_REF [REVISION] [PYTHON]}
REVISION=${2:-main}
PYTHON_BIN=${3:-$ROOT/.venv-pilot/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -z "$REVISION" ]]; then
  echo "Hugging Face revision must not be empty." >&2
  exit 2
fi
if [[ "$MODEL_REF" == \~/* ]]; then
  MODEL_REF="$HOME/${MODEL_REF#\~/}"
fi

MODEL_KIND=hub
if [[ -d "$MODEL_REF" ]]; then
  MODEL_KIND=local
elif [[ "$MODEL_REF" == /* || "$MODEL_REF" == ./* || "$MODEL_REF" == ../* || -e "$MODEL_REF" ]]; then
  echo "Local model path does not exist or is not a directory: $MODEL_REF" >&2
  exit 2
fi

if [[ "$MODEL_KIND" == hub ]]; then
  echo "Resolving Hugging Face model $MODEL_REF at revision $REVISION ..." >&2
fi
"$PYTHON_BIN" "$ROOT/stackpilot/hf_model_cache.py" \
  "$MODEL_REF" "$REVISION" "$MODEL_KIND"

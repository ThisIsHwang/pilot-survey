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
"$PYTHON_BIN" - "$MODEL_REF" "$REVISION" "$MODEL_KIND" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_ref, revision, model_kind = sys.argv[1:]
if model_kind == "hub":
    snapshot = Path(snapshot_download(repo_id=model_ref, revision=revision)).resolve()
else:
    snapshot = Path(model_ref).expanduser().resolve()

config_path = snapshot / "config.json"
try:
    json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid model config in {snapshot}: {exc}") from exc

index_files = sorted(snapshot.glob("*.safetensors.index.json")) or sorted(
    snapshot.glob("*.bin.index.json")
)
if len(index_files) > 1:
    raise SystemExit(f"Multiple model weight indexes found in {snapshot}: {index_files}")
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [snapshot / name for name in sorted(set(index["weight_map"].values()))]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid model weight index {index_files[0]}: {exc}") from exc
else:
    weights = sorted(snapshot.glob("*.safetensors")) or sorted(snapshot.glob("*.bin"))
bad_weights = [path for path in weights if not path.is_file() or path.stat().st_size == 0]
if not weights or bad_weights:
    raise SystemExit(
        f"Missing or incomplete model weights in {snapshot}: {bad_weights or 'none found'}"
    )
if not (snapshot / "tokenizer.json").is_file() and not (
    snapshot / "tokenizer.model"
).is_file():
    raise SystemExit(f"Tokenizer files are missing from {snapshot}")

print(snapshot)
if model_kind == "hub":
    print(f"Pinned {model_ref}@{revision} to snapshot {snapshot.name}", file=sys.stderr)
PY

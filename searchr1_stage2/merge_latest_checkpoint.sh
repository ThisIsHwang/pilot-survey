#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXP=${EXP:?Set EXP to the Search-R1 experiment name}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/work/checkpoints/$EXP}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/work/merged/$EXP}
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python

[[ -x "$SEARCH_R1_PYTHON" ]] || {
  echo "Missing .venv-searchr1; run scripts/bootstrap_searchr1.sh" >&2
  exit 1
}
if [[ ${ALLOW_INCOMPLETE_CHECKPOINT:-0} != 1 && ! -f "$CHECKPOINT_ROOT/.complete.json" ]]; then
  echo "Training is not marked complete: $CHECKPOINT_ROOT/.complete.json" >&2
  exit 1
fi

# The pinned Search-R1 FSDP worker already writes a complete Hugging Face model
# at actor/global_step_N. Prefer the exact checkpoint recorded atomically after
# training; only the explicit incomplete override falls back to a directory scan.
if [[ -f "$CHECKPOINT_ROOT/.complete.json" ]]; then
  LATEST=$("$SEARCH_R1_PYTHON" - \
    "$CHECKPOINT_ROOT/.complete.json" "$CHECKPOINT_ROOT" "$EXP" <<'PY'
import json
import sys
from pathlib import Path

marker_path, checkpoint_root, expected_experiment = sys.argv[1:]
payload = json.loads(Path(marker_path).read_text(encoding="utf-8"))
if payload.get("experiment") != expected_experiment:
    raise SystemExit(
        f"Completion marker experiment {payload.get('experiment')!r} does not match "
        f"{expected_experiment!r}"
    )
updates = int(payload["total_updates"])
if updates < 1:
    raise SystemExit(f"Invalid total_updates in {marker_path}: {updates}")
expected = (Path(checkpoint_root) / "actor" / f"global_step_{updates}").resolve()
recorded = Path(payload["checkpoint"]).resolve()
if recorded != expected:
    raise SystemExit(
        f"Completion marker points to {recorded}, expected exact final checkpoint {expected}"
    )
print(expected)
PY
  )
else
  LATEST=$(find "$CHECKPOINT_ROOT/actor" -mindepth 1 -maxdepth 1 \
    -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1 || true)
fi
[[ -n "$LATEST" ]] || {
  echo "No actor/global_step_* checkpoint under $CHECKPOINT_ROOT" >&2
  exit 1
}

"$SEARCH_R1_PYTHON" - "$LATEST" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = root / "config.json"
if not config.is_file():
    raise SystemExit(f"Checkpoint has no config.json: {root}")
json.loads(config.read_text(encoding="utf-8"))
index_files = sorted(root.glob("*.safetensors.index.json"))
if len(index_files) > 1:
    raise SystemExit(f"Checkpoint has multiple safetensors indexes: {root}")
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [root / name for name in sorted(set(index["weight_map"].values()))]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid checkpoint weight index: {index_files[0]}") from exc
else:
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
if not weights or any(
    not path.is_file() or path.stat().st_size < 1024 * 1024 for path in weights
):
    raise SystemExit(f"Checkpoint weights are missing or incomplete: {root}")
if not (root / "tokenizer.json").is_file() and not (root / "tokenizer.model").is_file():
    raise SystemExit(f"Checkpoint tokenizer is missing: {root}")
PY

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ -L "$OUTPUT_DIR" ]]; then
  current_target=$(readlink -f "$OUTPUT_DIR" || true)
  if [[ "$current_target" == "$(readlink -f "$LATEST")" ]]; then
    echo "Reusing merged-model link: $OUTPUT_DIR -> $LATEST"
    exit 0
  fi
  rm -- "$OUTPUT_DIR"
elif [[ -d "$OUTPUT_DIR" ]]; then
  if [[ -z "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rmdir -- "$OUTPUT_DIR"
  elif [[ ${FORCE_MODEL_LINK:-0} == 1 && "$OUTPUT_DIR" == "$ROOT/work/merged/"* ]]; then
    backup="${OUTPUT_DIR}.previous.$(date +%Y%m%d-%H%M%S)"
    mv -- "$OUTPUT_DIR" "$backup"
    echo "Moved previous merged output to: $backup"
  else
    echo "Refusing to replace existing output directory: $OUTPUT_DIR" >&2
    exit 1
  fi
elif [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to replace non-directory output: $OUTPUT_DIR" >&2
  exit 1
fi

ln -s "$LATEST" "$OUTPUT_DIR"
echo "$OUTPUT_DIR -> $LATEST"

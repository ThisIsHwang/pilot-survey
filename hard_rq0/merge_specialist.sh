#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
SEED=${SEED:?Set SEED}
PROFILE=${PROFILE:-pilot}

case "$BACKEND" in
  bm25|e5) ;;
  *) echo "BACKEND must be bm25 or e5" >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer; got '$SEED'." >&2
  exit 2
fi
case "$PROFILE" in
  smoke|pilot|full) ;;
  *) echo "PROFILE must be smoke, pilot, or full" >&2; exit 2 ;;
esac
[[ -x "$SEARCH_R1_PYTHON" ]] || {
  echo "Missing .venv-searchr1; run bash scripts/bootstrap_searchr1.sh" >&2
  exit 1
}

EXP=${EXP:-hard-rq0-${BACKEND}-seed${SEED}-${PROFILE}}
if [[ ! "$EXP" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXP may contain only letters, digits, dot, underscore, and dash: $EXP" >&2
  exit 2
fi
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/work/hard_rq0/checkpoints/$EXP}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/work/hard_rq0/merged/$EXP}
CHECKPOINT_ROOT=$("$SEARCH_R1_PYTHON" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$CHECKPOINT_ROOT")
OUTPUT_DIR=$("$SEARCH_R1_PYTHON" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$OUTPUT_DIR")
COMPLETE_MARKER=$CHECKPOINT_ROOT/.complete.json

[[ -f "$COMPLETE_MARKER" ]] || {
  echo "Training is not complete: $COMPLETE_MARKER" >&2
  exit 1
}
EXPECTED_CHECKPOINT=$("$SEARCH_R1_PYTHON" - \
  "$COMPLETE_MARKER" "$CHECKPOINT_ROOT" "$EXP" "$BACKEND" "$SEED" "$PROFILE" <<'PY'
import json
import sys
from pathlib import Path

marker_path, checkpoint_root, experiment, backend, seed, profile = sys.argv[1:]
try:
    payload = json.loads(Path(marker_path).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid completion marker {marker_path}: {exc}") from exc
if payload.get("schema") != 1:
    raise SystemExit(f"Unsupported completion marker schema: {payload.get('schema')!r}")
expected = {
    "experiment": experiment,
    "backend": backend,
    "seed": int(seed),
    "profile": profile,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"Completion marker {key}={payload.get(key)!r}, expected {value!r}"
        )
if not payload.get("training_signature"):
    raise SystemExit("Completion marker has no training signature")
updates = int(payload.get("total_updates", 0))
if updates < 1 or int(payload.get("trainer_stop_step", 0)) != updates + 1:
    raise SystemExit("Completion marker does not describe an exact N-update run")
expected_checkpoint = (Path(checkpoint_root) / "actor" / f"global_step_{updates}").resolve()
recorded_checkpoint = Path(payload.get("checkpoint", "")).resolve()
if recorded_checkpoint != expected_checkpoint:
    raise SystemExit(
        f"Completion marker points to {recorded_checkpoint}, expected {expected_checkpoint}"
    )
config = expected_checkpoint / "config.json"
try:
    json.loads(config.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid checkpoint config {config}: {exc}") from exc
index_files = sorted(expected_checkpoint.glob("*.safetensors.index.json")) or sorted(
    expected_checkpoint.glob("*.bin.index.json")
)
if len(index_files) > 1:
    raise SystemExit(f"Checkpoint has multiple weight indexes: {expected_checkpoint}")
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [
            expected_checkpoint / name
            for name in sorted(set(index["weight_map"].values()))
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid checkpoint weight index: {index_files[0]}") from exc
else:
    weights = sorted(expected_checkpoint.glob("*.safetensors")) or sorted(
        expected_checkpoint.glob("*.bin")
    )
if not weights or any(
    not path.is_file() or path.stat().st_size < 1024 * 1024 for path in weights
):
    raise SystemExit(f"Checkpoint weights are missing or incomplete: {expected_checkpoint}")
if not (expected_checkpoint / "tokenizer.json").is_file() and not (
    expected_checkpoint / "tokenizer.model"
).is_file():
    raise SystemExit(f"Checkpoint tokenizer is missing: {expected_checkpoint}")
print(expected_checkpoint)
PY
)

# The shared merger already validates the model config, tokenizer, and every
# weight shard before publishing the symlink. Allow an explicit replacement of
# a legacy real directory only inside the hard-RQ0 merged-model root.
if [[ -d "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" && \
      ${FORCE_MODEL_LINK:-0} == 1 ]]; then
  case "$OUTPUT_DIR/" in
    "$ROOT/work/hard_rq0/merged/"*) ;;
    *)
      echo "Refusing to rotate nonstandard OUTPUT_DIR: $OUTPUT_DIR" >&2
      exit 1
      ;;
  esac
  backup="${OUTPUT_DIR}.previous.$(date +%Y%m%d-%H%M%S).$$"
  mv -- "$OUTPUT_DIR" "$backup"
  echo "Moved previous merged output to: $backup"
fi

EXP="$EXP" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"

[[ -L "$OUTPUT_DIR" ]] || {
  echo "Merged output is not a symlink: $OUTPUT_DIR" >&2
  exit 1
}
if [[ "$(readlink -f "$OUTPUT_DIR")" != "$EXPECTED_CHECKPOINT" ]]; then
  echo "Merged output does not point at the validated final checkpoint: $OUTPUT_DIR" >&2
  exit 1
fi
echo "Validated hard-RQ0 model link: $OUTPUT_DIR -> $EXPECTED_CHECKPOINT"

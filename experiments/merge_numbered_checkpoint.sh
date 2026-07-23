#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EXPERIMENT_ID=${EXPERIMENT_ID:?Set EXPERIMENT_ID}
SEED=${SEED:?Set SEED}
PROFILE=${PROFILE:-pilot}
VARIANT=${VARIANT:?Set VARIANT}
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
RUN_ID=${RUN_ID:-$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)}
CHECKPOINT_ROOT=$ROOT/work/experiments/$EXPERIMENT_ID/checkpoints/$RUN_ID
OUTPUT_DIR=$ROOT/work/experiments/$EXPERIMENT_ID/merged/$RUN_ID
EXP=$RUN_ID CHECKPOINT_ROOT=$CHECKPOINT_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  bash "$ROOT/searchr1_stage2/merge_latest_checkpoint.sh"
echo "$OUTPUT_DIR"

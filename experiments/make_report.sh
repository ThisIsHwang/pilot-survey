#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
EXP002_ROOT=${EXP002_ROOT:-$ROOT/work/hard_rq0}
EXP002_RESULT_SET=${EXP002_RESULT_SET:-$PROFILE}
HARD_RESULTS=${HARD_RESULTS:-$EXP002_ROOT/runs/$EXP002_RESULT_SET/results/policies}
EXP003_RESULTS=${EXP003_RESULTS:-$ROOT/work/experiments/EXP-003/results}
EXP004_RESULTS=${EXP004_RESULTS:-$ROOT/work/experiments/EXP-004/results}
EXP005_RESULTS=${EXP005_RESULTS:-$ROOT/work/experiments/EXP-005/results}
EXP006_RESULTS=${EXP006_RESULTS:-$ROOT/work/experiments/EXP-006/results}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/work/experiments/reports/$PROFILE}

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
for path in "$HARD_RESULTS" "$EXP003_RESULTS" "$EXP004_RESULTS"; do
  [[ -d "$path" ]] || { echo "Missing required result directory: $path" >&2; exit 1; }
done

args=(
  --profile "$PROFILE"
  --hard-results "$HARD_RESULTS"
  --exp003-results "$EXP003_RESULTS"
  --exp004-results "$EXP004_RESULTS"
  --output-dir "$OUTPUT_DIR"
)
if [[ -d "$EXP005_RESULTS" ]] && find "$EXP005_RESULTS" -type f -name '*.jsonl' -print -quit | grep -q .; then
  args+=(--exp005-results "$EXP005_RESULTS")
fi
if [[ -d "$EXP006_RESULTS" ]] && find "$EXP006_RESULTS" -type f -name '*.jsonl' -print -quit | grep -q .; then
  args+=(--exp006-results "$EXP006_RESULTS")
fi

"$PILOT_PYTHON" -m stackpilot.numbered_experiment_report "${args[@]}"
echo "Report: $OUTPUT_DIR/NUMBERED_EXPERIMENT_REPORT.md"

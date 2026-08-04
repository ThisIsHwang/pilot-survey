#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
JOBS=$ROOT/work/multipositive_generalization/interactive_plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing $JOBS; run interactive_plan.sh." >&2; exit 1; }
GPUS=${MULTIPOSITIVE_INTERACTIVE_GPUS:-"0 1 2 3 4 5 6"}
WORKERS=${MULTIPOSITIVE_INTERACTIVE_WORKERS:-7}
# shellcheck disable=SC2206
GPU_ARGS=($GPUS)
"$PYTHON" -m stackpilot.query_attribution_scheduler --jobs "$JOBS" --python "$PYTHON" --gpus "${GPU_ARGS[@]}" --workers "$WORKERS" "$@"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
JOBS=$ROOT/work/multipositive_generalization/plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing $JOBS; run plan.sh." >&2; exit 1; }
"$PYTHON" -m stackpilot.trace_model_contract --jobs "$JOBS"
GPUS=${MULTIPOSITIVE_GPUS:-"0 1 2 3 4 5 6 7"}
WORKERS=${MULTIPOSITIVE_WORKERS:-8}
STAGGER=${MULTIPOSITIVE_LAUNCH_STAGGER:-2}
# shellcheck disable=SC2206
GPU_ARGS=($GPUS)
"$PYTHON" -m stackpilot.query_attribution_scheduler --jobs "$JOBS" --python "$PYTHON" --gpus "${GPU_ARGS[@]}" --workers "$WORKERS" --launch-stagger "$STAGGER" "$@"

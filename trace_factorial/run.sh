#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
WORK_DIR=${TRACE_FACTORIAL_WORK_DIR:-$ROOT/work/trace_factorial}
JOBS=$WORK_DIR/plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing $JOBS; run trace_factorial/plan.sh." >&2; exit 1; }
TRACE_GPUS=${TRACE_GPUS:-"0 1 2 3 4 5 6 7"}
TRACE_WORKERS=${TRACE_WORKERS:-8}
TRACE_LAUNCH_STAGGER=${TRACE_LAUNCH_STAGGER:-2}
"$PYTHON" -m stackpilot.trace_model_contract --jobs "$JOBS"
# shellcheck disable=SC2206
GPU_ARGS=($TRACE_GPUS)
"$PYTHON" -m stackpilot.trace_factorial_scheduler \
  --jobs "$JOBS" \
  --python "$PYTHON" \
  --gpus "${GPU_ARGS[@]}" \
  --workers "$TRACE_WORKERS" \
  --launch-stagger "$TRACE_LAUNCH_STAGGER" \
  "$@"

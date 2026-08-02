#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
JOBS=$ROOT/work/query_attribution/plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing $JOBS; run query_attribution/plan.sh." >&2; exit 1; }
"$PYTHON" -m stackpilot.trace_model_contract --jobs "$JOBS"
QUERY_ATTRIBUTION_GPUS=${QUERY_ATTRIBUTION_GPUS:-"0 1 2 3 4 5 6 7"}
QUERY_ATTRIBUTION_WORKERS=${QUERY_ATTRIBUTION_WORKERS:-8}
QUERY_ATTRIBUTION_LAUNCH_STAGGER=${QUERY_ATTRIBUTION_LAUNCH_STAGGER:-2}
# shellcheck disable=SC2206
GPU_ARGS=($QUERY_ATTRIBUTION_GPUS)
"$PYTHON" -m stackpilot.query_attribution_scheduler \
  --jobs "$JOBS" \
  --python "$PYTHON" \
  --gpus "${GPU_ARGS[@]}" \
  --workers "$QUERY_ATTRIBUTION_WORKERS" \
  --launch-stagger "$QUERY_ATTRIBUTION_LAUNCH_STAGGER" \
  "$@"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
JOBS=$ROOT/work/query_attribution/interactive_plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing interactive plan; run interactive_plan.sh." >&2; exit 1; }
QUERY_ATTR_INTERACTIVE_GPUS=${QUERY_ATTR_INTERACTIVE_GPUS:-"0 1 2 3 4 5 6"}
# shellcheck disable=SC2206
GPU_ARGS=($QUERY_ATTR_INTERACTIVE_GPUS)
"$PYTHON" -m stackpilot.query_attribution_scheduler --jobs "$JOBS" --python "$PYTHON" --gpus "${GPU_ARGS[@]}" --workers "${QUERY_ATTR_INTERACTIVE_WORKERS:-7}" --launch-stagger "${QUERY_ATTR_INTERACTIVE_LAUNCH_STAGGER:-3}" "$@"

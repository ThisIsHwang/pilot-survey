#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run query_equivalence/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
QUERY_EQUIVALENCE_GPUS=${QUERY_EQUIVALENCE_GPUS:-"0 1 2 3 4 5 6 7"}
QUERY_EQUIVALENCE_WORKERS=${QUERY_EQUIVALENCE_WORKERS:-8}
QUERY_EQUIVALENCE_LAUNCH_STAGGER=${QUERY_EQUIVALENCE_LAUNCH_STAGGER:-2}
JOBS=$ROOT/work/query_equivalence/plans/$PROFILE/jobs.jsonl
[[ -s "$JOBS" ]] || { echo "Missing $JOBS; run query_equivalence/plan.sh." >&2; exit 1; }
"$PYTHON" -m stackpilot.trace_model_contract --jobs "$JOBS"
# shellcheck disable=SC2206
GPU_ARGS=($QUERY_EQUIVALENCE_GPUS)
"$PYTHON" -m stackpilot.query_equivalence_scheduler \
  --jobs "$JOBS" \
  --python "$PYTHON" \
  --gpus "${GPU_ARGS[@]}" \
  --workers "$QUERY_EQUIVALENCE_WORKERS" \
  --launch-stagger "$QUERY_EQUIVALENCE_LAUNCH_STAGGER" \
  "$@"

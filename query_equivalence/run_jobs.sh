#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
JOBS=$ROOT/work/query_equivalence/plans/$PROFILE/jobs.jsonl
[[ -f "$JOBS" ]] || { echo "Missing plan: $JOBS" >&2; exit 1; }
"$PYTHON" -m stackpilot.trace_model_contract --jobs "$JOBS"
read -r -a GPUS <<< "${QUERY_EQUIVALENCE_GPUS:-0 1 2 3 4 5 6 7}"
"$PYTHON" -m stackpilot.query_equivalence_scheduler \
  --jobs "$JOBS" --python "$PYTHON" --gpus "${GPUS[@]}" \
  --workers "${QUERY_EQUIVALENCE_WORKERS:-8}" "$@"

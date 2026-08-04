#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
STARTED=0
cleanup(){ status=$?; if [[ $STARTED == 1 && ${KEEP_SERVICES:-0} != 1 ]]; then bash hard_rq0/stop_retrievers.sh || true; fi; exit $status; }
trap cleanup EXIT INT TERM
if [[ ${SKIP_SERVICES:-0} != 1 ]]; then E5_GPU=${E5_GPU:-7} bash hard_rq0/launch_retrievers.sh; STARTED=1; fi
ARGS=()
if [[ -n ${BEHAVIOR_FEEDBACK_INPUTS:-} ]]; then IFS=':' read -r -a VALUES <<< "$BEHAVIOR_FEEDBACK_INPUTS"; for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done; fi
"$PYTHON" -m stackpilot.paired_retriever_grid --config configs/behavior_feedback.yaml --profile "$PROFILE" "${ARGS[@]}"

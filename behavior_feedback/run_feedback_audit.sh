#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ! -x "$ROOT/.venv-trace/bin/python" ]]; then
  bash trace_go/bootstrap.sh
fi
PYTHON=${FEEDBACK_PYTHON:-$ROOT/.venv-trace/bin/python}
STARTED=0
cleanup() {
  status=$?
  if [[ $STARTED == 1 && ${KEEP_SERVICES:-0} != 1 ]]; then
    bash hard_rq0/stop_retrievers.sh || true
  fi
  exit $status
}
trap cleanup EXIT INT TERM
if [[ ${SKIP_SERVICES:-0} != 1 ]]; then
  E5_GPU=${E5_GPU:-7} bash hard_rq0/launch_retrievers.sh
  STARTED=1
fi
export CUDA_VISIBLE_DEVICES=${FEEDBACK_GPU:-0}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
ARGS=()
if [[ -n ${BEHAVIOR_FEEDBACK_INPUTS:-} ]]; then
  IFS=':' read -r -a VALUES <<< "$BEHAVIOR_FEEDBACK_INPUTS"
  for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done
fi
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.response_feedback_audit \
  --config configs/behavior_feedback.yaml --profile "$PROFILE" "${ARGS[@]}"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
[[ -x "$ROOT/.venv-trace/bin/python" ]] || bash trace_go/bootstrap.sh
[[ -x "$ROOT/.venv-vllm/bin/python" ]] || bash scripts/bootstrap_vllm.sh
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
STARTED=0
cleanup(){ status=$?; if [[ $STARTED == 1 && ${KEEP_SERVICES:-0} != 1 ]]; then bash causal_query_audit/stop_services.sh || true; fi; exit $status; }
trap cleanup EXIT INT TERM
if [[ ${SKIP_SERVICES:-0} != 1 ]]; then
  CAUSAL_QUERY_BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  E5_GPU=${E5_GPU:-7} bash causal_query_audit/launch_services.sh
  STARTED=1
fi
if [[ ${SKIP_DOCUMENT_CTU:-0} != 1 ]]; then
  if [[ -n ${BEHAVIOR_FEEDBACK_INPUTS:-} ]]; then export INTERFACE_CAUSAL_INPUTS="$BEHAVIOR_FEEDBACK_INPUTS"; fi
  PROFILE="$PROFILE" bash interface_causality/run_document_ctu.sh
fi
ARGS=()
if [[ -n ${BEHAVIOR_FEEDBACK_INPUTS:-} ]]; then IFS=':' read -r -a VALUES <<< "$BEHAVIOR_FEEDBACK_INPUTS"; for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done; fi
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.document_action_ctu_closure \
  --config configs/behavior_feedback.yaml --profile "$PROFILE" "${ARGS[@]}"

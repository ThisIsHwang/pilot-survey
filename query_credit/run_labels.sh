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
  CAUSAL_QUERY_BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} E5_GPU=${E5_GPU:-7} \
    bash causal_query_audit/launch_services.sh
  STARTED=1
fi
ARGS=()
if [[ -n ${QUERY_CREDIT_INPUTS:-} ]]; then
  IFS=':' read -r -a VALUES <<< "$QUERY_CREDIT_INPUTS"
  for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done
fi
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.query_credit_labels \
  --config configs/query_credit.yaml --profile "$PROFILE" "${ARGS[@]}"

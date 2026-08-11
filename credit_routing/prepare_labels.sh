#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
[[ -x "$ROOT/.venv-vllm/bin/python" ]] || bash scripts/bootstrap_vllm.sh
STARTED=0
cleanup(){ status=$?; if [[ $STARTED == 1 && ${KEEP_SERVICES:-0} != 1 ]]; then bash causal_query_audit/stop_services.sh || true; fi; exit $status; }
trap cleanup EXIT INT TERM
if [[ ${SKIP_SERVICES:-0} != 1 ]]; then
  CAUSAL_QUERY_BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
  E5_GPU=${E5_GPU:-7} bash causal_query_audit/launch_services.sh
  STARTED=1
fi
ARGS=()
if [[ -n ${CREDIT_ROUTING_INPUTS:-} ]]; then
  IFS=':' read -r -a VALUES <<< "$CREDIT_ROUTING_INPUTS"
  for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done
fi
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.credit_routing_labels \
  --config configs/credit_routing.yaml \
  --causal-config configs/causal_query_audit.yaml \
  --profile "$PROFILE" "${ARGS[@]}"

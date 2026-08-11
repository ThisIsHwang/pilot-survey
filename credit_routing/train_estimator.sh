#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
ARGS=()
if [[ -n ${CREDIT_ROUTING_LABELS:-} ]]; then
  IFS=':' read -r -a VALUES <<< "$CREDIT_ROUTING_LABELS"
  for value in "${VALUES[@]}"; do ARGS+=(--input "$value"); done
fi
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -m stackpilot.credit_routing_model \
  --config configs/credit_routing.yaml --profile "$PROFILE" "${ARGS[@]}"

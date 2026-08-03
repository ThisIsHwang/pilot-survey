#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
ARGS=()
if [[ -n ${INTERFACE_CAUSAL_INPUTS:-} ]]; then
  IFS=':' read -r -a INPUTS <<< "$INTERFACE_CAUSAL_INPUTS"
  ARGS+=(--inputs "${INPUTS[@]}")
fi
[[ -n ${DOCUMENT_CTU_FILE:-} ]] && ARGS+=(--document-ctu "$DOCUMENT_CTU_FILE")
"$PYTHON" -m stackpilot.interface_credit_granularity \
  --config configs/interface_causality.yaml \
  --profile "$PROFILE" "${ARGS[@]}"

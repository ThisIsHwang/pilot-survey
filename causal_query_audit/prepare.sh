#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first." >&2; exit 1; }
CONFIG=${CAUSAL_QUERY_CONFIG:-configs/causal_query_audit.yaml}
PROFILE=${PROFILE:-pilot}
ARGS=(--config "$CONFIG" --profile "$PROFILE")
if [[ -n ${CAUSAL_QUERY_INPUTS:-${TRACE_INPUTS:-}} ]]; then
  value=${CAUSAL_QUERY_INPUTS:-$TRACE_INPUTS}
  normalized=${value//$'\n'/:}
  IFS=':' read -r -a patterns <<< "$normalized"
  ARGS+=(--inputs "${patterns[@]}")
fi
"$PYTHON" -m stackpilot.causal_query_prepare "${ARGS[@]}" "$@"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-trace/bin/python
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
CONFIG=${TRACE_CONFIG:-configs/trace_go.yaml}
ARGS=(--config "$CONFIG")
if [[ -n ${TRACE_INPUTS:-} ]]; then
  # TRACE_INPUTS is a newline- or colon-separated list so glob patterns remain intact.
  normalized=${TRACE_INPUTS//$'\n'/:}
  IFS=':' read -r -a patterns <<< "$normalized"
  ARGS+=(--inputs "${patterns[@]}")
fi
"$PYTHON" -m stackpilot.trace_bank "${ARGS[@]}"

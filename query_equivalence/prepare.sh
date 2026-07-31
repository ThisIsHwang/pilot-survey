#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first" >&2; exit 1; }
args=()
if [[ -n ${QUERY_EQUIVALENCE_INPUTS:-} ]]; then
  IFS=':' read -r -a patterns <<< "$QUERY_EQUIVALENCE_INPUTS"
  args+=(--results "${patterns[@]}")
fi
"$PYTHON" -m stackpilot.query_equivalence_prepare \
  --config "${QUERY_EQUIVALENCE_CONFIG:-configs/query_equivalence.yaml}" \
  "${args[@]}"

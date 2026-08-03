#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
if [[ ${REFRESH_EQUIVALENCE_PREPARE:-0} == 1 ]]; then
  if [[ -n ${QUERY_ATTRIBUTION_INPUTS:-} ]]; then
    export QUERY_EQUIVALENCE_INPUTS="$QUERY_ATTRIBUTION_INPUTS"
  fi
  PROFILE="$PROFILE" bash query_equivalence/prepare.sh
fi
"$PYTHON" -m stackpilot.query_attribution_prepare --config configs/query_attribution.yaml --profile "$PROFILE"

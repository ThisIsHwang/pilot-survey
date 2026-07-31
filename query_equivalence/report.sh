#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
"$PYTHON" -m stackpilot.query_equivalence_report \
  --config "${QUERY_EQUIVALENCE_CONFIG:-configs/query_equivalence.yaml}" \
  --profile "$PROFILE" "$@"

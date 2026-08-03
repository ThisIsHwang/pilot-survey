#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
PROFILE=${PROFILE:-pilot}
"$PYTHON" -m stackpilot.query_attribution_interactive_plan --config configs/query_attribution.yaml --profile "$PROFILE"

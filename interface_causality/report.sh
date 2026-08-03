#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${TRACE_PYTHON:-$ROOT/.venv-trace/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run trace_go/bootstrap.sh first." >&2; exit 1; }
PROFILE=${PROFILE:-pilot}
"$PYTHON" -m stackpilot.interface_suite_report \
  --config configs/interface_causality.yaml \
  --profile "$PROFILE"
cat "work/interface_causality/reports/$PROFILE/suite/INTERFACE_CAUSALITY_REPORT.md"

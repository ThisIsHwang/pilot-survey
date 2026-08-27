#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
INPUT=${CREDIT_DEPTH_META_INPUT:-$ROOT/credit_depth/meta_audit_template.csv}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -m stackpilot.credit_depth_meta_audit \
  --input "$INPUT" --output "$ROOT/work/credit_depth/reports/$PROFILE/EXP-061"

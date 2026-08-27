#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -m stackpilot.credit_depth_gate \
  --census "$ROOT/work/credit_depth/reports/$PROFILE/EXP-063/$BACKEND" \
  --labels "$ROOT/work/credit_depth/reports/$PROFILE/EXP-064/$BACKEND" \
  --output "$ROOT/work/credit_depth/reports/$PROFILE/prerequisite-gate/$BACKEND"

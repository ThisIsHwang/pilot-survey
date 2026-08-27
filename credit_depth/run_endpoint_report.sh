#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -m stackpilot.credit_depth_endpoint_report \
  --config "$CONFIG" --profile "$PROFILE" \
  --input "$ROOT/work/credit_depth/endpoint/$PROFILE/*/episodes-*.csv" \
  --output "$ROOT/work/credit_depth/reports/$PROFILE/EXP-067"

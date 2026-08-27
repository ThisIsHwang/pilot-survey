#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
OUTPUT=$ROOT/work/credit_depth/benchmark/$PROFILE
"$PYTHON" -m stackpilot.credit_depth_builder \
  --config "$CONFIG" --profile "$PROFILE" --output "$OUTPUT"

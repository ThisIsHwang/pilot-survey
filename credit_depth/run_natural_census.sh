#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
INPUT=${CREDIT_DEPTH_NATURAL_INPUT:?Set CREDIT_DEPTH_NATURAL_INPUT to candidate_metrics.csv/jsonl}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -m stackpilot.credit_depth_census \
  --config "$CONFIG" --profile "$PROFILE" --natural-input "$INPUT" \
  --output "$ROOT/work/credit_depth/reports/$PROFILE/EXP-062"

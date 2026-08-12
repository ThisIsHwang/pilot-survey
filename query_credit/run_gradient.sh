#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
CUDA_VISIBLE_DEVICES=${GRADIENT_GPU:-0} "$PYTHON" -m stackpilot.query_credit_gradient \
  --config configs/query_credit.yaml --profile "$PROFILE"

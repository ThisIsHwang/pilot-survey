#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
PORT=$("$PYTHON" - "$CONFIG" "$BACKEND" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(c['retrieval'][f'{sys.argv[2]}_port'])
PY
)
PROFILE="$PROFILE" BACKEND="$BACKEND" bash credit_depth/launch_retriever.sh
"$PYTHON" -m stackpilot.credit_depth_census \
  --config "$CONFIG" --profile "$PROFILE" \
  --controlled-states "$ROOT/work/credit_depth/benchmark/$PROFILE/audit_states.jsonl" \
  --retriever-url "http://127.0.0.1:${PORT}/retrieve" \
  --output "$ROOT/work/credit_depth/reports/$PROFILE/EXP-063/$BACKEND"

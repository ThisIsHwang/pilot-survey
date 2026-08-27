#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
METHOD=${METHOD:?Set METHOD}
TRAINING_SEED=${TRAINING_SEED:?Set TRAINING_SEED}
MODEL=${MODEL:?Set MODEL to a served validation-best model name}
API_BASE=${API_BASE:-http://127.0.0.1:9000/v1}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
PORT=$($PYTHON - "$CONFIG" "$BACKEND" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(c['retrieval'][f'{sys.argv[2]}_port'])
PY
)
LIMIT=$($PYTHON - "$CONFIG" "$PROFILE" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(c['profiles'][sys.argv[2]]['eval_limit'])
PY
)
PROFILE="$PROFILE" BACKEND="$BACKEND" bash credit_depth/launch_retriever.sh
"$PYTHON" -m stackpilot.credit_depth_eval \
  --families "$ROOT/work/credit_depth/benchmark/$PROFILE/families.jsonl" \
  --output "$ROOT/work/credit_depth/endpoint/$PROFILE/$BACKEND" \
  --model "$MODEL" --api-base "$API_BASE" \
  --retriever-url "http://127.0.0.1:${PORT}/retrieve" \
  --method "$METHOD" --training-seed "$TRAINING_SEED" --limit "$LIMIT"

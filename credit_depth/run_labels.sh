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
COLLECTION=$ROOT/work/credit_depth/labels/$PROFILE/$BACKEND
if [[ ${CREDIT_DEPTH_SYNTHETIC_LABELS:-0} == 1 || "$PROFILE" == smoke && ${CREDIT_DEPTH_LIVE_SMOKE:-0} != 1 ]]; then
  REPEATS=$("$PYTHON" - "$CONFIG" "$PROFILE" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(','.join(map(str,c['profiles'][sys.argv[2]]['label_repeats'])))
PY
)
  "$PYTHON" -m stackpilot.credit_depth_synthetic_labels \
    --candidate-metrics "$ROOT/work/credit_depth/reports/$PROFILE/EXP-063/$BACKEND/candidate_metrics.csv" \
    --repeats "$REPEATS" --output "$COLLECTION"
else
  "$PYTHON" -m stackpilot.credit_depth_rollout \
    --config "$CONFIG" --profile "$PROFILE" \
    --states "$ROOT/work/credit_depth/benchmark/$PROFILE/audit_states.jsonl" \
    --retriever-url "http://127.0.0.1:${PORT}/retrieve" --output "$COLLECTION"
fi
"$PYTHON" -m stackpilot.credit_depth_label_audit \
  --config "$CONFIG" --input "$COLLECTION/candidate_replicates.csv" \
  --output "$ROOT/work/credit_depth/reports/$PROFILE/EXP-064/$BACKEND"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
SEEDS=${SEEDS:-$("$PYTHON" - configs/credit_routing.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
METHODS=${METHODS:-"outcome-only action-route observation-route both"}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
for backend in $BACKENDS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      KEEP_ROUTING_STACK=0 PROFILE="$PROFILE" BACKEND="$backend" METHOD="$method" SEED="$seed" \
        bash credit_routing/train_grpo.sh
    done
  done
done

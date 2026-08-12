#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY=$ROOT/experiments/registry.query_credit.json
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
SEEDS=${SEEDS:-$("$PYTHON" - configs/query_credit.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
METHODS=${METHODS:-"outcome doc-to-action alias-normalized shuffled-doc"}
for backend in $BACKENDS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      PROFILE="$PROFILE" BACKEND="$backend" METHOD="$method" SEED="$seed" \
        BQ_SETUP_READY=${BQ_SETUP_READY:-0} bash query_credit/train_grpo.sh
    done
  done
done

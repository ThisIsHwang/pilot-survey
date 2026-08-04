#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ! -x "$ROOT/.venv-pilot/bin/python" ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_searchr1.sh
fi
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
SEEDS=${SEEDS:-$("$PYTHON" - configs/behavior_quotient.yaml "$PROFILE" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
BACKENDS=${BACKENDS:-"bm25 e5"}
METHODS=${METHODS:-"random-surface balanced-surface random-quotient balanced-quotient"}
for backend in $BACKENDS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      BQ_SETUP_READY=${BQ_SETUP_READY:-0} \
        EXPERIMENT_ID=EXP-027 PROFILE="$PROFILE" SEED="$seed" \
        BACKEND="$backend" METHOD="$method" \
        bash behavior_quotient/train_grpo.sh
      export BQ_SETUP_READY=1
    done
  done
done

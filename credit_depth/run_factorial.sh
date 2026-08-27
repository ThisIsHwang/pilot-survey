#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKENDS=${BACKENDS:-"bm25 e5"}
METHODS=${METHODS:-"outcome immediate verifiable"}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
SEEDS=$($PYTHON - "$CONFIG" "$PROFILE" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(' '.join(map(str,c['profiles'][sys.argv[2]]['seeds'])))
PY
)
for backend in $BACKENDS; do
  PROFILE="$PROFILE" BACKEND="$backend" bash credit_depth/launch_retriever.sh
  for method in $METHODS; do
    for seed in $SEEDS; do
      EXPERIMENT_ID=EXP-065 PROFILE="$PROFILE" BACKEND="$backend" METHOD="$method" SEED="$seed" \
        bash credit_depth/train_grpo.sh
    done
  done
done

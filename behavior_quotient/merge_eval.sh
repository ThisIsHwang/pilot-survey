#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ! -x "$ROOT/.venv-pilot/bin/python" ]]; then
  bash scripts/bootstrap.sh
fi
if [[ ! -x "$ROOT/.venv-vllm/bin/python" ]]; then
  bash scripts/bootstrap_vllm.sh
fi
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
SEEDS=${SEEDS:-$("$PYTHON" - configs/behavior_quotient.yaml "$PROFILE" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
LIMIT=${LIMIT:-$("$PYTHON" - configs/behavior_quotient.yaml "$PROFILE" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
print(cfg['profiles'][sys.argv[2]]['eval_limit'])
PY
)}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
METHODS=${METHODS:-"standard random-surface balanced-surface random-quotient balanced-quotient"}
for source_backend in $BACKENDS; do
  for method in $METHODS; do
    variant="${source_backend}-${method}"
    for seed in $SEEDS; do
      if [[ "$method" == standard ]]; then
        training_experiment=EXP-024
      else
        training_experiment=EXP-027
      fi
      EXPERIMENT_ID="$training_experiment" SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
        bash experiments/merge_numbered_checkpoint.sh
      run_id=$("$PYTHON" -m stackpilot.experiment_registry run-id "$training_experiment" \
        --seed "$seed" --profile "$PROFILE" --variant "$variant")
      model_ref=$ROOT/work/experiments/$training_experiment/merged/$run_id
      EXPERIMENT_ID="$training_experiment" TAG="bq-${source_backend}-${method}" SEED="$seed" PROFILE="$PROFILE" \
        VARIANT="$variant" MODEL_REF="$model_ref" LIMIT="$LIMIT" \
        BACKENDS="bm25 e5 hybrid" TOPKS="3" \
        bash experiments/eval_numbered_policy.sh
    done
  done
done

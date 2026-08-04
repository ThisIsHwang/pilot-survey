#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
[[ -x "$ROOT/.venv-vllm/bin/python" ]] || bash scripts/bootstrap_vllm.sh
SEEDS=${SEEDS:-$("$PYTHON" - configs/behavior_feedback.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(' '.join(map(str,cfg['profiles'][sys.argv[2]]['seeds'])))
PY
)}
LIMIT=${LIMIT:-$("$PYTHON" - configs/behavior_feedback.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(cfg['profiles'][sys.argv[2]]['eval_limit'])
PY
)}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
METHODS=${METHODS:-"standard iid-surface posthoc-surface feedback-surface iid-quotient posthoc-quotient feedback-quotient"}
for source_backend in $BACKENDS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      if [[ "$method" == standard ]]; then
        training_experiment=EXP-024
      else
        training_experiment=EXP-031
      fi
      variant="${source_backend}-${method}"
      EXPERIMENT_ID="$training_experiment" SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
        bash experiments/merge_numbered_checkpoint.sh
      run_id=$("$PYTHON" -m stackpilot.experiment_registry run-id "$training_experiment" \
        --seed "$seed" --profile "$PROFILE" --variant "$variant")
      model_ref=$ROOT/work/experiments/$training_experiment/merged/$run_id
      EXPERIMENT_ID="$training_experiment" TAG="rf-${source_backend}-${method}" \
        SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" MODEL_REF="$model_ref" \
        LIMIT="$LIMIT" BACKENDS="bm25 e5 hybrid" TOPKS="3" \
        bash experiments/eval_numbered_policy.sh
    done
  done
done
"$PYTHON" -m stackpilot.response_feedback_training_report \
  --config configs/behavior_feedback.yaml --profile "$PROFILE"

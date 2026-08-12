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
LIMIT=${LIMIT:-$("$PYTHON" - configs/query_credit.yaml "$PROFILE" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(cfg['profiles'][sys.argv[2]]['eval_limit'])
PY
)}
METHODS=${METHODS:-"outcome doc-to-action alias-normalized shuffled-doc"}
BACKENDS=${SOURCE_BACKENDS:-"bm25 e5"}
for source_backend in $BACKENDS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      variant="${source_backend}-${method}"
      EXPERIMENT_ID=EXP-055 SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
        bash experiments/merge_numbered_checkpoint.sh
      run_id=$("$PYTHON" -m stackpilot.experiment_registry run-id EXP-055 \
        --seed "$seed" --profile "$PROFILE" --variant "$variant")
      model_ref=$ROOT/work/experiments/EXP-055/merged/$run_id
      EXPERIMENT_ID=EXP-055 TAG="qc-${source_backend}-${method}" \
        SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" MODEL_REF="$model_ref" \
        LIMIT="$LIMIT" BACKENDS="bm25 e5 hybrid" TOPKS="3" \
        bash experiments/eval_numbered_policy.sh
    done
  done
done
"$PYTHON" -m stackpilot.query_credit_endpoint_report \
  --config configs/query_credit.yaml --profile "$PROFILE"

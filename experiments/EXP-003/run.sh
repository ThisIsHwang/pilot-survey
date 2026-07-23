#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
LIMIT=${LIMIT:-500}
TOPKS=${TOPKS:-"3 5 10"}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}

bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash scripts/bootstrap_searchr1.sh
bash hard_rq0/download_assets.sh
bash hard_rq0/prepare_data.sh

for seed in $SEEDS; do
  EXPERIMENT_ID=EXP-003 SEED="$seed" PROFILE="$PROFILE" \
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="$BASE_MODEL_REVISION" \
    bash experiments/train_mixed_policy.sh
  EXPERIMENT_ID=EXP-003 SEED="$seed" PROFILE="$PROFILE" VARIANT=blind \
    bash experiments/merge_numbered_checkpoint.sh
  run_id=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-003 \
      --seed "$seed" --profile "$PROFILE" --variant blind
  )
  EXPERIMENT_ID=EXP-003 TAG=mixed-blind SEED="$seed" PROFILE="$PROFILE" VARIANT=blind \
    MODEL_REF="$ROOT/work/experiments/EXP-003/merged/$run_id" \
    LIMIT="$LIMIT" TOPKS="$TOPKS" BACKENDS="bm25 e5" \
    bash experiments/eval_numbered_policy.sh
done

echo "EXP-003 complete: work/experiments/EXP-003"

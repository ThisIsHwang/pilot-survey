#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
if [[ -z ${LIMIT:-} ]]; then
  if [[ "$PROFILE" == smoke ]]; then LIMIT=20; else LIMIT=500; fi
fi
TOPKS=${TOPKS:-"3 5 10"}
DEFAULT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DEFAULT_BASE_MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
BASE_MODEL=${BASE_MODEL:-$DEFAULT_BASE_MODEL}
if [[ -z ${BASE_MODEL_REVISION:-} ]]; then
  if [[ "$BASE_MODEL" == "$DEFAULT_BASE_MODEL" ]]; then
    BASE_MODEL_REVISION=$DEFAULT_BASE_MODEL_REVISION
  else
    BASE_MODEL_REVISION=main
  fi
fi

if [[ ${NUMBERED_SETUP_READY:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_vllm.sh
  bash scripts/bootstrap_searchr1.sh
  bash hard_rq0/download_assets.sh
  bash hard_rq0/prepare_data.sh
fi

for seed in $SEEDS; do
  bash experiments/reset_searchr1_experiment_files.sh
  EXPERIMENT_ID=EXP-004 SEED="$seed" PROFILE="$PROFILE" \
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="$BASE_MODEL_REVISION" \
    bash experiments/train_mixed_policy.sh
  EXPERIMENT_ID=EXP-004 SEED="$seed" PROFILE="$PROFILE" VARIANT=backend-id \
    bash experiments/merge_numbered_checkpoint.sh
  run_id=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-004 \
      --seed "$seed" --profile "$PROFILE" --variant backend-id
  )
  EXPERIMENT_ID=EXP-004 TAG=mixed-backend-id SEED="$seed" PROFILE="$PROFILE" VARIANT=backend-id \
    MODEL_REF="$ROOT/work/experiments/EXP-004/merged/$run_id" \
    LIMIT="$LIMIT" TOPKS="$TOPKS" BACKENDS="bm25 e5" INJECT_BACKEND_ID=1 \
    bash experiments/eval_numbered_policy.sh
done

echo "EXP-004 complete: work/experiments/EXP-004"

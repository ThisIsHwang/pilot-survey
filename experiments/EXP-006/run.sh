#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"13 42 87"}
ORACLE_SEEDS=${ORACLE_SEEDS:-"42"}
LIMIT=${LIMIT:-500}
TOPKS=${TOPKS:-"3 5 10"}
BASE_MODEL_REF=${BASE_MODEL_REF:-Qwen/Qwen2.5-3B-Instruct}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}
REQUIRE_ALL=${REQUIRE_ALL:-1}

bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash hard_rq0/download_assets.sh
bash hard_rq0/prepare_data.sh
bash experiments/launch_hybrid_rrf.sh

run_eval() {
  local tag=$1
  local seed=$2
  local variant=$3
  local model_ref=$4
  local model_revision=${5:-main}
  local inject_backend_id=${6:-0}
  if [[ ! -e "$model_ref" && "$model_ref" == /* ]]; then
    if [[ "$REQUIRE_ALL" == 1 ]]; then
      echo "Missing source policy for EXP-006: $model_ref" >&2
      exit 1
    fi
    echo "Skipping missing source policy: $model_ref" >&2
    return
  fi
  local run_id
  run_id=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-006 \
      --seed "$seed" --profile "$PROFILE" --variant "$variant"
  )
  EXPERIMENT_ID=EXP-006 TAG="$tag" SEED="$seed" PROFILE="$PROFILE" \
    VARIANT="$variant" RUN_ID="$run_id" MODEL_REF="$model_ref" \
    MODEL_REVISION="$model_revision" LIMIT="$LIMIT" TOPKS="$TOPKS" \
    BACKENDS="hybrid" INJECT_BACKEND_ID="$inject_backend_id" \
    bash experiments/eval_numbered_policy.sh
}

run_eval base-qwen 0 base-qwen "$BASE_MODEL_REF" "$BASE_MODEL_REVISION"

for seed in $SEEDS; do
  for backend in bm25 e5; do
    specialist="$ROOT/work/hard_rq0/merged/hard-rq0-${backend}-seed${seed}-${PROFILE}"
    run_eval "${backend}-specialist" "$seed" "exp002-${backend}-specialist" "$specialist"
  done
  mixed_run=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-003 \
      --seed "$seed" --profile "$PROFILE" --variant blind
  )
  run_eval mixed-blind "$seed" exp003-mixed-blind \
    "$ROOT/work/experiments/EXP-003/merged/$mixed_run"
done

for seed in $ORACLE_SEEDS; do
  oracle_run=$(
    .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-004 \
      --seed "$seed" --profile "$PROFILE" --variant backend-id
  )
  # "hybrid" is intentionally out of the oracle policy's BM25/E5 training labels.
  run_eval mixed-backend-id "$seed" exp004-backend-id \
    "$ROOT/work/experiments/EXP-004/merged/$oracle_run" main 1
done

echo "EXP-006 complete: work/experiments/EXP-006"

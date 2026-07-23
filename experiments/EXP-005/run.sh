#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
SEEDS=${SEEDS:-"42"}
BACKEND_LIST=${BACKEND_LIST:-"bm25 e5"}
LIMIT=${LIMIT:-500}
TOPKS=${TOPKS:-"3 5 10"}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}
export ANSWER_REWARD_WEIGHT=${ANSWER_REWARD_WEIGHT:-1.0}
export EVIDENCE_REWARD_WEIGHT=${EVIDENCE_REWARD_WEIGHT:-0.5}
export SEARCH_COST_WEIGHT=${SEARCH_COST_WEIGHT:-0.02}

bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
bash scripts/bootstrap_searchr1.sh
bash hard_rq0/download_assets.sh
bash hard_rq0/prepare_data.sh
bash hard_rq0/launch_retrievers.sh

.venv-pilot/bin/python - \
  "$ANSWER_REWARD_WEIGHT" "$EVIDENCE_REWARD_WEIGHT" "$SEARCH_COST_WEIGHT" <<'PY'
import math, sys
names = ("ANSWER_REWARD_WEIGHT", "EVIDENCE_REWARD_WEIGHT", "SEARCH_COST_WEIGHT")
for name, value in zip(names, sys.argv[1:]):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SystemExit(f"{name} must be finite and non-negative; got {value!r}")
PY
reward_suffix="a${ANSWER_REWARD_WEIGHT}-e${EVIDENCE_REWARD_WEIGHT}-c${SEARCH_COST_WEIGHT}"

for backend in $BACKEND_LIST; do
  for seed in $SEEDS; do
    # Bootstrap preserves verified local patches; the evidence patch is idempotent.
    bash scripts/bootstrap_searchr1.sh
    .venv-searchr1/bin/python hard_rq0/patch_searchr1_evidence_reward.py \
      --search-r1-root upstream/Search-R1
    variant="${backend}-evidence-${reward_suffix}"
    run_id=$(
      .venv-pilot/bin/python -m stackpilot.experiment_registry run-id EXP-005 \
        --seed "$seed" --profile "$PROFILE" --variant "$variant"
    )
    EXP="$run_id" \
    CHECKPOINT_DIR="$ROOT/work/experiments/EXP-005/checkpoints/$run_id" \
    LOG_FILE="$ROOT/logs/experiments/EXP-005/${run_id}.log" \
    BACKEND="$backend" SEED="$seed" PROFILE="$PROFILE" \
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="$BASE_MODEL_REVISION" \
      bash hard_rq0/train_specialist.sh

    EXPERIMENT_ID=EXP-005 SEED="$seed" PROFILE="$PROFILE" VARIANT="$variant" \
      RUN_ID="$run_id" bash experiments/merge_numbered_checkpoint.sh
    EXPERIMENT_ID=EXP-005 TAG="evidence-${backend}" SEED="$seed" \
      PROFILE="$PROFILE" VARIANT="$variant" RUN_ID="$run_id" \
      MODEL_REF="$ROOT/work/experiments/EXP-005/merged/$run_id" \
      LIMIT="$LIMIT" TOPKS="$TOPKS" BACKENDS="bm25 e5" \
      bash experiments/eval_numbered_policy.sh
  done
done

echo "EXP-005 complete: work/experiments/EXP-005"

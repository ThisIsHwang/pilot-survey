#!/usr/bin/env bash
# Stage-2 specialist baseline. Run inside the official Search-R1 environment.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEARCH_R1=$ROOT/upstream/Search-R1
BACKEND=${BACKEND:-bm25}
PROFILE=${PROFILE:-smoke}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}

case "$BACKEND" in
  bm25) PORT=${PORT:-8001} ;;
  e5) PORT=${PORT:-8002} ;;
  *) echo "BACKEND must be bm25 or e5" >&2; exit 2 ;;
esac

case "$PROFILE" in
  smoke)
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    TRAIN_BATCH=${TRAIN_BATCH:-32}
    VAL_BATCH=${VAL_BATCH:-32}
    MINI_BATCH=${MINI_BATCH:-16}
    MICRO_BATCH=${MICRO_BATCH:-2}
    SAVE_FREQ=${SAVE_FREQ:-20}
    TEST_FREQ=${TEST_FREQ:-10}
    ;;
  pilot)
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
    TRAIN_BATCH=${TRAIN_BATCH:-128}
    VAL_BATCH=${VAL_BATCH:-128}
    MINI_BATCH=${MINI_BATCH:-64}
    MICRO_BATCH=${MICRO_BATCH:-8}
    SAVE_FREQ=${SAVE_FREQ:-100}
    TEST_FREQ=${TEST_FREQ:-50}
    ;;
  *) echo "PROFILE must be smoke or pilot" >&2; exit 2 ;;
esac

EXP=${EXP:-hotpot-${BACKEND}-${PROFILE}-grpo}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/work/checkpoints/$EXP}
LOG_FILE=$ROOT/logs/${EXP}.log
mkdir -p "$ROOT/logs" "$CHECKPOINT_DIR"

[[ -d "$SEARCH_R1" ]] || { echo "Missing $SEARCH_R1; run scripts/bootstrap.sh first" >&2; exit 1; }
[[ -f "$ROOT/work/searchr1_hotpot/train.parquet" ]] || {
  echo "Missing Search-R1 train data; run searchr1_stage2/prepare_searchr1_data.sh" >&2
  exit 1
}

cd "$SEARCH_R1"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$ROOT/work/searchr1_hotpot/train.parquet" \
  data.val_files="$ROOT/work/searchr1_hotpot/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH" data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=700 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="['console','wandb']" \
  +trainer.val_only=false +trainer.val_before_train=true \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptPilot trainer.experiment_name="$EXP" \
  trainer.total_epochs="$TOTAL_EPOCHS" trainer.default_local_dir="$CHECKPOINT_DIR" \
  max_turns=3 retriever.url="http://127.0.0.1:${PORT}/retrieve" retriever.topk=10 \
  2>&1 | tee "$LOG_FILE"

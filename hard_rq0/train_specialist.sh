#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEARCH_R1=$ROOT/upstream/Search-R1
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
SEED=${SEED:?Set SEED, e.g. 13}
PROFILE=${PROFILE:-pilot}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
TOPK=${TOPK:-3}
LOGGER=${LOGGER:-"['console']"}

case "$BACKEND" in
  bm25) PORT=${PORT:-8101} ;;
  e5) PORT=${PORT:-8102} ;;
  *) echo "BACKEND must be bm25 or e5" >&2; exit 2 ;;
esac

case "$PROFILE" in
  smoke)
    TOTAL_STEPS=${TOTAL_STEPS:-20}
    TRAIN_BATCH=${TRAIN_BATCH:-28}
    VAL_BATCH=${VAL_BATCH:-28}
    MINI_BATCH=${MINI_BATCH:-14}
    MICRO_BATCH=${MICRO_BATCH:-2}
    SAVE_FREQ=${SAVE_FREQ:-20}
    TEST_FREQ=${TEST_FREQ:-20}
    ;;
  pilot)
    TOTAL_STEPS=${TOTAL_STEPS:-200}
    TRAIN_BATCH=${TRAIN_BATCH:-112}
    VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}
    MICRO_BATCH=${MICRO_BATCH:-4}
    SAVE_FREQ=${SAVE_FREQ:-100}
    TEST_FREQ=${TEST_FREQ:-100}
    ;;
  full)
    TOTAL_STEPS=${TOTAL_STEPS:-500}
    TRAIN_BATCH=${TRAIN_BATCH:-112}
    VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}
    MICRO_BATCH=${MICRO_BATCH:-4}
    SAVE_FREQ=${SAVE_FREQ:-100}
    TEST_FREQ=${TEST_FREQ:-100}
    ;;
  *) echo "PROFILE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

EXP=${EXP:-hard-rq0-${BACKEND}-seed${SEED}-${PROFILE}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/work/hard_rq0/checkpoints/$EXP}
LOG_FILE=$ROOT/logs/hard_rq0/${EXP}.log
TRAIN_DATA=$ROOT/work/hard_rq0/searchr1/train.parquet
VAL_DATA=$ROOT/work/hard_rq0/searchr1/test.parquet
mkdir -p "$CHECKPOINT_DIR" "$ROOT/logs/hard_rq0"

[[ -d "$SEARCH_R1" ]] || { echo "Missing Search-R1 checkout; run scripts/bootstrap.sh" >&2; exit 1; }
[[ -f "$TRAIN_DATA" && -f "$VAL_DATA" ]] || { echo "Run hard_rq0/prepare_data.sh first" >&2; exit 1; }
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "Retriever $BACKEND is not ready on port $PORT; run hard_rq0/launch_retrievers.sh" >&2
  exit 1
}

"$ROOT/.venv-pilot/bin/python" "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  --search-r1-root "$SEARCH_R1"

cd "$SEARCH_R1"
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export TOKENIZERS_PARALLELISM=true
export RQ0_SEED=$SEED
export PYTHONHASHSEED=$SEED
export PYTHONPATH="$ROOT/hard_rq0:$SEARCH_R1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" \
  data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=700 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=14 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=14 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.state_masking=true \
  algorithm.no_think_rl=false \
  trainer.logger="$LOGGER" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptHardRQ0 \
  trainer.experiment_name="$EXP" \
  trainer.total_epochs=100 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  max_turns=4 \
  retriever.url="http://127.0.0.1:${PORT}/retrieve" \
  retriever.topk="$TOPK" \
  2>&1 | tee "$LOG_FILE"

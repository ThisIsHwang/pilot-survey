#!/usr/bin/env bash
# Stage-2 baseline after the pilot passes. Run inside the official Search-R1 environment.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEARCH_R1=$ROOT/upstream/Search-R1
BACKEND=${BACKEND:-bm25}
PORT=${PORT:-8001}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B}
EXP=${EXP:-hotpot-${BACKEND}-grpo}

cd "$SEARCH_R1"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=XFORMERS
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$ROOT/work/searchr1_hotpot/train.parquet \
  data.val_files=$ROOT/work/searchr1_hotpot/test.parquet \
  data.train_batch_size=256 data.val_batch_size=128 \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=700 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger=['console','wandb'] \
  +trainer.val_only=false +trainer.val_before_train=true \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq=100 trainer.test_freq=50 \
  trainer.project_name=StackAdaptPilot trainer.experiment_name=$EXP \
  trainer.total_epochs=3 trainer.default_local_dir=$ROOT/work/checkpoints/$EXP \
  max_turns=3 retriever.url=http://127.0.0.1:${PORT}/retrieve retriever.topk=10 \
  2>&1 | tee $ROOT/logs/${EXP}.log

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EXPERIMENT_ID=${EXPERIMENT_ID:?Set EXPERIMENT_ID=EXP-003 or EXP-004}
SEED=${SEED:?Set SEED, e.g. 13}
PROFILE=${PROFILE:-pilot}
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
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
E5_GPU=${E5_GPU:-7}
MIXED_PORT=${MIXED_PORT:-8200}
TOPK=${TOPK:-3}
N_AGENT=${N_AGENT:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=500
MAX_START_LENGTH=2048
MAX_OBS_LENGTH=500
MAX_TURNS=4
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}
# EXP-003/004 isolate retriever routing while retaining the answer-only reward.
unset ANSWER_REWARD_WEIGHT EVIDENCE_REWARD_WEIGHT SEARCH_COST_WEIGHT
export SEARCH_R1_REWARD_MODE=answer

case "$EXPERIMENT_ID" in
  EXP-003) MIXED_MODE=blind; VARIANT=blind ;;
  EXP-004) MIXED_MODE=oracle; VARIANT=backend-id ;;
  *) echo "train_mixed_policy.sh supports EXP-003 and EXP-004; got $EXPERIMENT_ID" >&2; exit 2 ;;
esac
case "$PROFILE" in
  smoke)
    TOTAL_UPDATES=${TOTAL_UPDATES:-20}; TRAIN_BATCH=${TRAIN_BATCH:-28}; VAL_BATCH=${VAL_BATCH:-28}
    MINI_BATCH=${MINI_BATCH:-14}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-20}; TEST_FREQ=${TEST_FREQ:-20}
    ;;
  pilot)
    TOTAL_UPDATES=${TOTAL_UPDATES:-200}; TRAIN_BATCH=${TRAIN_BATCH:-112}; VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-100}; TEST_FREQ=${TEST_FREQ:-100}
    ;;
  full)
    TOTAL_UPDATES=${TOTAL_UPDATES:-500}; TRAIN_BATCH=${TRAIN_BATCH:-112}; VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}; MICRO_BATCH=${MICRO_BATCH:-7}; SAVE_FREQ=${SAVE_FREQ:-100}; TEST_FREQ=${TEST_FREQ:-100}
    ;;
  *) echo "PROFILE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
for executable in "$PILOT_PYTHON" "$SEARCH_R1_PYTHON"; do
  [[ -x "$executable" ]] || { echo "Missing environment; run bootstrap scripts first" >&2; exit 1; }
done
[[ -e "$SEARCH_R1/.git" ]] || {
  echo "Missing isolated Search-R1 checkout: $SEARCH_R1" >&2
  exit 1
}
if [[ "$N_GPUS" != 7 || "$TRAIN_GPUS" != "0,1,2,3,4,5,6" || "$E5_GPU" != 7 ]]; then
  echo "Numbered mixed experiments reserve GPUs 0-6 for GRPO and GPU 7 for E5" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer" >&2
  exit 2
fi
if [[ ! "$N_AGENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_AGENT must be a positive integer" >&2
  exit 2
fi
for value in TRAIN_BATCH VAL_BATCH MINI_BATCH MICRO_BATCH TOTAL_UPDATES TOPK; do
  number=${!value}
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "$value must be a positive integer" >&2; exit 2; }
done
for boolean_name in ACTOR_PARAM_OFFLOAD ACTOR_GRAD_OFFLOAD \
  ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD; do
  boolean_value=${!boolean_name}
  if [[ "$boolean_value" != true && "$boolean_value" != false ]]; then
    echo "$boolean_name must be true or false; got '$boolean_value'." >&2
    exit 2
  fi
done
if (( TRAIN_BATCH % N_GPUS != 0 || VAL_BATCH % N_GPUS != 0 || \
      MINI_BATCH % N_GPUS != 0 || MICRO_BATCH % N_GPUS != 0 )); then
  echo "TRAIN_BATCH, VAL_BATCH, MINI_BATCH, and MICRO_BATCH must be divisible by N_GPUS=$N_GPUS" >&2
  exit 2
fi
if (( MINI_BATCH / N_GPUS < MICRO_BATCH / N_GPUS || \
      (MINI_BATCH / N_GPUS) % (MICRO_BATCH / N_GPUS) != 0 )); then
  echo "Per-GPU MINI_BATCH must be a positive multiple of per-GPU MICRO_BATCH" >&2
  exit 2
fi
if (( (TRAIN_BATCH * N_AGENT) % MINI_BATCH != 0 )); then
  echo "TRAIN_BATCH*N_AGENT must be divisible by MINI_BATCH" >&2
  exit 2
fi
if [[ "$MIXED_MODE" == blind ]] && (( TRAIN_BATCH % 2 != 0 )); then
  echo "Blind mixed training requires an even TRAIN_BATCH for group-level 50:50 routing" >&2
  exit 2
fi

RUN_ID=$(
  "$PILOT_PYTHON" -m stackpilot.experiment_registry run-id "$EXPERIMENT_ID" \
    --seed "$SEED" --profile "$PROFILE" --variant "$VARIANT"
)
EXPERIMENT_ROOT=$ROOT/work/experiments/$EXPERIMENT_ID
CHECKPOINT_DIR=$EXPERIMENT_ROOT/checkpoints/$RUN_ID
LOG_DIR=$ROOT/logs/experiments/$EXPERIMENT_ID
LOG_FILE=$LOG_DIR/${RUN_ID}.log
COMPLETE_MARKER=$CHECKPOINT_DIR/.complete.json
mkdir -p "$LOG_DIR"

SOURCE_TRAIN=$ROOT/work/hard_rq0/searchr1/train.parquet
SOURCE_DEV=$ROOT/work/hard_rq0/searchr1/dev.parquet
DATA_MANIFEST=$ROOT/work/hard_rq0/data/.hard-rq0-data-manifest.json
for path in "$SOURCE_TRAIN" "$SOURCE_DEV"; do
  [[ -s "$path" ]] || { echo "Missing hard-RQ0 data: $path" >&2; exit 1; }
done
"$PILOT_PYTHON" -m stackpilot.prepare_hard_rq0 \
  --config "$ROOT/configs/hard_rq0.yaml" --check
if [[ "$MIXED_MODE" == oracle ]]; then
  DATA_DIR=$EXPERIMENT_ROOT/data/seed-${SEED}
  TRAIN_DATA=$DATA_DIR/train.parquet
  VAL_DATA=$DATA_DIR/dev.parquet
  "$PILOT_PYTHON" -m stackpilot.prepare_mixed_data \
    --input "$SOURCE_TRAIN" --output "$TRAIN_DATA" --seed "$SEED" \
    --mode backend-id
  "$PILOT_PYTHON" -m stackpilot.prepare_mixed_data \
    --input "$SOURCE_DEV" --output "$VAL_DATA" --seed "$SEED" \
    --mode backend-id
else
  TRAIN_DATA=$SOURCE_TRAIN
  VAL_DATA=$SOURCE_DEV
fi
"$PILOT_PYTHON" -m stackpilot.prepare_hard_rq0 \
  --config "$ROOT/configs/hard_rq0.yaml" \
  --validate-training-inputs \
  --train-file "$TRAIN_DATA" \
  --val-file "$VAL_DATA"

if [[ ${NUMBERED_SETUP_READY:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap_searchr1.sh"
fi
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_seed.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_validation.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_action_protocol.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_reward_protocol.py" --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_mixed.py" --search-r1-root "$SEARCH_R1"

BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" "$BASE_MODEL" "$BASE_MODEL_REVISION" "$SEARCH_R1_PYTHON")
TRAINER_STOP_STEP=$((TOTAL_UPDATES + 1))
LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-14}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
LOGGER=${LOGGER:-"['console']"}

TRAIN_SIGNATURE=$(
  "$SEARCH_R1_PYTHON" - \
    "$TRAIN_DATA" "$VAL_DATA" "$BASE_MODEL" "$SEARCH_R1" \
    "$EXPERIMENT_ID" "$RUN_ID" "$MIXED_MODE" "$SEED" "$PROFILE" \
    "$TOTAL_UPDATES" "$TRAIN_BATCH" "$VAL_BATCH" "$MINI_BATCH" "$MICRO_BATCH" \
    "$N_AGENT" "$TOPK" "$N_GPUS" "$TRAIN_GPUS" "$MAX_PROMPT_LENGTH" \
    "$MAX_RESPONSE_LENGTH" "$MAX_START_LENGTH" "$MAX_OBS_LENGTH" "$MAX_TURNS" \
    "$LEARNING_RATE" "$ROLLOUT_GPU_MEMORY" "$LOG_PROB_MICRO_BATCH" \
    "$ACTOR_PARAM_OFFLOAD" "$ACTOR_GRAD_OFFLOAD" "$ACTOR_OPTIMIZER_OFFLOAD" "$REF_PARAM_OFFLOAD" \
    "$ROOT/hard_rq0/patch_searchr1_mixed.py" \
    "$ROOT/hard_rq0/patch_searchr1_experiment_env.py" \
    "$ROOT/hard_rq0/patch_searchr1_seed.py" \
    "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" \
    "$ROOT/hard_rq0/patch_searchr1_validation.py" \
    "$ROOT/hard_rq0/patch_searchr1_action_protocol.py" \
    "$ROOT/hard_rq0/patch_searchr1_reward_protocol.py" \
    "$ROOT/hard_rq0/sitecustomize.py" \
    "$ROOT/stackpilot/action_protocol.py" \
    "$ROOT/stackpilot/prepare_mixed_data.py" \
    "$ROOT/stackpilot/mixed_retriever_server.py" "$DATA_MANIFEST" \
    "$ROOT/experiments/train_mixed_policy.sh" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
(
    train, val, model, search_r1, experiment_id, run_id, mode, seed, profile,
    updates, train_batch, val_batch, mini_batch, micro_batch, n_agent, topk,
    n_gpus, train_gpus, max_prompt, max_response, max_start, max_obs, max_turns,
    learning_rate, rollout_memory, log_batch, actor_param_offload,
    actor_grad_offload, actor_optimizer_offload, ref_param_offload,
    mixed_patch, env_patch, seed_patch, worker_cuda_patch, validation_patch,
    action_protocol_patch, reward_protocol_patch, sitecustomize,
    action_protocol, mixed_preparer,
    mixed_router, data_manifest, wrapper,
) = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def model_identity(path):
    root = Path(path).resolve()
    files = {}
    for pattern in (
        "config.json",
        "tokenizer*",
        "*.index.json",
        "*.safetensors",
        "*.bin",
    ):
        for item in sorted(root.glob(pattern)):
            if item.is_file():
                files[item.name] = {
                    "size": item.stat().st_size,
                    "mtime_ns": item.stat().st_mtime_ns,
                }
    if not files:
        raise SystemExit(f"No model artifacts found under {root}")
    return {"resolved_path": str(root), "files": files}
diff = subprocess.run(
    ['git', '-C', search_r1, 'diff', '--binary', 'HEAD'],
    check=True,
    stdout=subprocess.PIPE,
).stdout
payload = {
    'schema': 3,
    'experiment_id': experiment_id,
    'run_id': run_id,
    'mode': mode,
    'seed': int(seed),
    'profile': profile,
    'total_updates': int(updates),
    'train_batch': int(train_batch),
    'val_batch': int(val_batch),
    'mini_batch': int(mini_batch),
    'micro_batch': int(micro_batch),
    'n_agent': int(n_agent),
    'topk': int(topk),
    'n_gpus': int(n_gpus),
    'train_gpus': train_gpus,
    'max_prompt': int(max_prompt),
    'max_response': int(max_response),
    'max_start': int(max_start),
    'max_obs': int(max_obs),
    'max_turns': int(max_turns),
    'learning_rate': learning_rate,
    'rollout_memory': float(rollout_memory),
    'log_batch': int(log_batch),
    'actor_param_offload': actor_param_offload == 'true',
    'actor_grad_offload': actor_grad_offload == 'true',
    'actor_optimizer_offload': actor_optimizer_offload == 'true',
    'ref_param_offload': ref_param_offload == 'true',
    'train_sha256': digest(train),
    'val_sha256': digest(val),
    'data_manifest_sha256': digest(data_manifest),
    'model': model_identity(model),
    'search_r1_diff_sha256': hashlib.sha256(diff).hexdigest(),
    'patches': {
        'mixed': digest(mixed_patch),
        'environment': digest(env_patch),
        'seed': digest(seed_patch),
        'worker_cuda': digest(worker_cuda_patch),
        'validation': digest(validation_patch),
        'action_protocol_patch': digest(action_protocol_patch),
        'reward_protocol_patch': digest(reward_protocol_patch),
        'sitecustomize': digest(sitecustomize),
        'action_protocol': digest(action_protocol),
        'mixed_preparer': digest(mixed_preparer),
        'mixed_router': digest(mixed_router),
        'wrapper': digest(wrapper),
    },
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
)

FINAL_CHECKPOINT=$CHECKPOINT_DIR/actor/global_step_$TOTAL_UPDATES

validate_checkpoint_artifact() {
  "$SEARCH_R1_PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
config = checkpoint / "config.json"
try:
    json.loads(config.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid checkpoint config {config}: {exc}") from exc
index_files = sorted(checkpoint.glob("*.safetensors.index.json")) or sorted(
    checkpoint.glob("*.bin.index.json")
)
if len(index_files) > 1:
    raise SystemExit(f"Checkpoint has multiple weight indexes: {checkpoint}")
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [
            checkpoint / name
            for name in sorted(set(index["weight_map"].values()))
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid checkpoint weight index: {index_files[0]}") from exc
else:
    weights = sorted(checkpoint.glob("*.safetensors")) or sorted(
        checkpoint.glob("*.bin")
    )
if not weights or any(
    not path.is_file() or path.stat().st_size < 1024 * 1024 for path in weights
):
    raise SystemExit(f"Checkpoint weights are missing or incomplete: {checkpoint}")
if not (checkpoint / "tokenizer.json").is_file() and not (
    checkpoint / "tokenizer.model"
).is_file():
    raise SystemExit(f"Checkpoint tokenizer is missing: {checkpoint}")
PY
}

validate_completed_run() {
  "$SEARCH_R1_PYTHON" - \
    "$COMPLETE_MARKER" "$FINAL_CHECKPOINT" "$RUN_ID" "$EXPERIMENT_ID" \
    "$MIXED_MODE" "$SEED" "$PROFILE" "$TOTAL_UPDATES" "$TRAIN_SIGNATURE" \
    <<'PY' || return 1
import json
import sys
from pathlib import Path

(
    marker_path,
    final_checkpoint,
    run_id,
    experiment_id,
    routing_mode,
    seed,
    profile,
    total_updates,
    training_signature,
) = sys.argv[1:]
marker = Path(marker_path)
checkpoint = Path(final_checkpoint).resolve()
try:
    payload = json.loads(marker.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid completion marker {marker}: {exc}") from exc
expected = {
    "schema": 2,
    "experiment": run_id,
    "experiment_id": experiment_id,
    "routing_mode": routing_mode,
    "seed": int(seed),
    "profile": profile,
    "total_updates": int(total_updates),
    "training_signature": training_signature,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"Completion marker {key}={payload.get(key)!r}, expected {value!r}"
        )
recorded = Path(payload.get("checkpoint", "")).resolve()
if recorded != checkpoint:
    raise SystemExit(
        f"Completion marker points to {recorded}, expected {checkpoint}"
    )
PY
  validate_checkpoint_artifact "$FINAL_CHECKPOINT" || return 1
}

if [[ -f "$COMPLETE_MARKER" && ${FORCE_TRAIN:-0} != 1 ]]; then
  marker_signature=$(
    "$SEARCH_R1_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["training_signature"])' \
      "$COMPLETE_MARKER" 2>/dev/null || true
  )
  if [[ "$marker_signature" == "$TRAIN_SIGNATURE" ]]; then
    if validate_completed_run; then
      "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
      echo "Reusing validated completed run: $RUN_ID"
      exit 0
    fi
    corrupt_backup="${CHECKPOINT_DIR}.corrupt.$(date +%Y%m%d-%H%M%S).$$"
    mv -- "$CHECKPOINT_DIR" "$corrupt_backup"
    echo "Archived corrupt completed run before retraining: $corrupt_backup" >&2
  else
    stale_backup="${CHECKPOINT_DIR}.stale.$(date +%Y%m%d-%H%M%S).$$"
    mv -- "$CHECKPOINT_DIR" "$stale_backup"
    echo "Archived protocol-incompatible completed run before retraining: $stale_backup" >&2
  fi
fi
if [[ -e "$CHECKPOINT_DIR" ]]; then
  if [[ ${FORCE_TRAIN:-0} != 1 ]]; then
    echo "Checkpoint directory already exists without a reusable marker: $CHECKPOINT_DIR" >&2
    exit 1
  fi
  mv "$CHECKPOINT_DIR" "${CHECKPOINT_DIR}.previous.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$CHECKPOINT_DIR"

bash "$ROOT/experiments/launch_mixed_router.sh"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy
health=$(curl --noproxy '*' -fsS "http://127.0.0.1:${MIXED_PORT}/health")
"$PILOT_PYTHON" - \
  "$health" "$ROOT/stackpilot/mixed_retriever_server.py" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
expected_digest = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if (
    payload.get("status") != "ok"
    or payload.get("backend") != "mixed"
    or payload.get("server_file_sha256") != expected_digest
):
    raise SystemExit(f"Unexpected or stale mixed-router health: {payload}")
PY

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export TOKENIZERS_PARALLELISM=false
export RQ0_SEED=$SEED
export PYTHONHASHSEED=$SEED
export SEARCH_R1_MIXED_MODE=$MIXED_MODE
export SEARCH_R1_N_AGENT=$N_AGENT
export PYTHONPATH="$ROOT:$ROOT/hard_rq0:$SEARCH_R1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export SEARCH_R1_RETRIEVER_TIMEOUT=${SEARCH_R1_RETRIEVER_TIMEOUT:-180}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  if [[ $status -ne 0 ]]; then
    echo "Mixed GRPO failed; log: $LOG_FILE" >&2
    tail -n 100 "$LOG_FILE" >&2 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$SEARCH_R1"
echo "Training $RUN_ID mode=$MIXED_MODE seed=$SEED updates=$TOTAL_UPDATES"
"$SEARCH_R1_PYTHON" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" data.max_obs_length="$MAX_OBS_LENGTH" \
  data.shuffle_train_dataloader=true algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="$ACTOR_GRAD_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" actor_rollout_ref.actor.state_masking=true \
  algorithm.no_think_rl=false trainer.logger="$LOGGER" \
  +trainer.val_only=false +trainer.val_before_train=true \
  trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptNumberedExperiments trainer.experiment_name="$RUN_ID" \
  trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TRAINER_STOP_STEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" trainer.default_hdfs_dir=null \
  max_turns="$MAX_TURNS" retriever.url="http://127.0.0.1:${MIXED_PORT}/retrieve" \
  retriever.topk="$TOPK" 2>&1 | tee "$LOG_FILE"

validate_checkpoint_artifact "$FINAL_CHECKPOINT"
"$SEARCH_R1_PYTHON" - "$COMPLETE_MARKER" "$FINAL_CHECKPOINT" "$RUN_ID" \
  "$EXPERIMENT_ID" "$MIXED_MODE" "$SEED" "$PROFILE" "$TOTAL_UPDATES" "$TRAIN_SIGNATURE" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
marker, checkpoint, run_id, experiment_id, mode, seed, profile, updates, signature = sys.argv[1:]
payload = {
    'schema': 2,
    'experiment': run_id,
    'experiment_id': experiment_id,
    'routing_mode': mode,
    'seed': int(seed),
    'profile': profile,
    'total_updates': int(updates),
    'training_signature': signature,
    'checkpoint': str(Path(checkpoint).resolve()),
    'completed_at': datetime.now(timezone.utc).isoformat(),
}
path = Path(marker); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
os.replace(tmp, path)
PY
echo "Completed $RUN_ID: $FINAL_CHECKPOINT"

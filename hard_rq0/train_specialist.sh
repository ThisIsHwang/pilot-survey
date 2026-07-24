#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
BACKEND=${BACKEND:?Set BACKEND=bm25 or e5}
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
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
N_GPUS=${N_GPUS:-7}
E5_GPU=${E5_GPU:-7}
DEFAULT_E5_MODEL=intfloat/e5-base-v2
DEFAULT_E5_MODEL_REVISION=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
E5_MODEL_SOURCE=${E5_MODEL:-$DEFAULT_E5_MODEL}
if [[ -z ${E5_MODEL_REVISION:-} ]]; then
  if [[ "$E5_MODEL_SOURCE" == "$DEFAULT_E5_MODEL" ]]; then
    E5_MODEL_REVISION=$DEFAULT_E5_MODEL_REVISION
  else
    E5_MODEL_REVISION=main
  fi
fi
N_AGENT=${N_AGENT:-4}
TOPK=${TOPK:-3}
# Search-R1 Appendix B.2 and the pinned v0.2 GRPO recipe use this rollout
# geometry. Keep these values explicit so the training signature records the
# scientific protocol instead of relying only on the wrapper's file hash.
readonly MAX_PROMPT_LENGTH=4096
readonly MAX_RESPONSE_LENGTH=500
readonly MAX_START_LENGTH=2048
readonly MAX_OBS_LENGTH=500
readonly MAX_TURNS=4
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
LOGGER=${LOGGER:-"['console']"}
SEARCH_R1_REWARD_MODE=${SEARCH_R1_REWARD_MODE:-answer}
ANSWER_REWARD_WEIGHT=${ANSWER_REWARD_WEIGHT:-1.0}
EVIDENCE_REWARD_WEIGHT=${EVIDENCE_REWARD_WEIGHT:-0.5}
SEARCH_COST_WEIGHT=${SEARCH_COST_WEIGHT:-0.02}
case "$SEARCH_R1_REWARD_MODE" in
  answer|evidence) ;;
  *)
    echo "SEARCH_R1_REWARD_MODE must be answer or evidence; got '$SEARCH_R1_REWARD_MODE'." >&2
    exit 2
    ;;
esac
export SEARCH_R1_REWARD_MODE ANSWER_REWARD_WEIGHT EVIDENCE_REWARD_WEIGHT SEARCH_COST_WEIGHT
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
ASSET_MANIFEST=$ASSET_ROOT/.hard-rq0-assets-manifest.json
DATA_MANIFEST=$ROOT/work/hard_rq0/data/.hard-rq0-data-manifest.json
CORPUS_PATH=$ASSET_ROOT/wiki-18.jsonl
TRAIN_DATA=$ROOT/work/hard_rq0/searchr1/train.parquet
VAL_DATA=$ROOT/work/hard_rq0/searchr1/dev.parquet

case "$BACKEND" in
  bm25)
    PORT=${PORT:-${BM25_PORT:-8101}}
    INDEX_PATH=$ASSET_ROOT/bm25
    ;;
  e5)
    PORT=${PORT:-${E5_PORT:-8102}}
    INDEX_PATH=$ASSET_ROOT/e5_Flat.index
    ;;
  *) echo "BACKEND must be bm25 or e5" >&2; exit 2 ;;
esac

case "$PROFILE" in
  smoke)
    DEFAULT_TOTAL_UPDATES=20
    TRAIN_BATCH=${TRAIN_BATCH:-28}
    VAL_BATCH=${VAL_BATCH:-28}
    MINI_BATCH=${MINI_BATCH:-14}
    MICRO_BATCH=${MICRO_BATCH:-7}
    DEFAULT_SAVE_FREQ=20
    DEFAULT_TEST_FREQ=20
    ;;
  pilot)
    DEFAULT_TOTAL_UPDATES=200
    TRAIN_BATCH=${TRAIN_BATCH:-112}
    VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}
    MICRO_BATCH=${MICRO_BATCH:-7}
    DEFAULT_SAVE_FREQ=100
    DEFAULT_TEST_FREQ=100
    ;;
  full)
    DEFAULT_TOTAL_UPDATES=500
    TRAIN_BATCH=${TRAIN_BATCH:-112}
    VAL_BATCH=${VAL_BATCH:-112}
    MINI_BATCH=${MINI_BATCH:-56}
    MICRO_BATCH=${MICRO_BATCH:-7}
    DEFAULT_SAVE_FREQ=100
    DEFAULT_TEST_FREQ=100
    ;;
  *) echo "PROFILE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

if [[ -n ${TOTAL_UPDATES+x} && -n ${TOTAL_STEPS+x} && "$TOTAL_UPDATES" != "$TOTAL_STEPS" ]]; then
  echo "TOTAL_UPDATES and legacy TOTAL_STEPS disagree: $TOTAL_UPDATES != $TOTAL_STEPS" >&2
  exit 2
fi
if [[ -n ${TOTAL_UPDATES+x} ]]; then
  TOTAL_UPDATES_VALUE=$TOTAL_UPDATES
elif [[ -n ${TOTAL_STEPS+x} ]]; then
  TOTAL_UPDATES_VALUE=$TOTAL_STEPS
else
  TOTAL_UPDATES_VALUE=$DEFAULT_TOTAL_UPDATES
fi
TOTAL_UPDATES=$TOTAL_UPDATES_VALUE
SAVE_FREQ=${SAVE_FREQ:-$DEFAULT_SAVE_FREQ}
TEST_FREQ=${TEST_FREQ:-$DEFAULT_TEST_FREQ}
LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-14}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}

EXP=${EXP:-hard-rq0-${BACKEND}-seed${SEED}-${PROFILE}}
if [[ ! "$EXP" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXP may contain only letters, digits, dot, underscore, and dash: $EXP" >&2
  exit 2
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/work/hard_rq0/checkpoints/$EXP}
LOG_FILE=${LOG_FILE:-$ROOT/logs/hard_rq0/${EXP}.log}

[[ -x "$SEARCH_R1_PYTHON" ]] || {
  echo "Missing .venv-searchr1; run bash scripts/bootstrap_searchr1.sh" >&2
  exit 1
}
[[ -x "$PILOT_PYTHON" ]] || {
  echo "Missing .venv-pilot; run bash scripts/bootstrap.sh" >&2
  exit 1
}
"$SEARCH_R1_PYTHON" - \
  "$ANSWER_REWARD_WEIGHT" "$EVIDENCE_REWARD_WEIGHT" "$SEARCH_COST_WEIGHT" <<'PY'
import math
import sys

names = ("ANSWER_REWARD_WEIGHT", "EVIDENCE_REWARD_WEIGHT", "SEARCH_COST_WEIGHT")
for name, raw_value in zip(names, sys.argv[1:]):
    value = float(raw_value)
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"{name} must be finite and non-negative; got {raw_value!r}")
PY
[[ -e "$SEARCH_R1/.git" ]] || {
  echo "Missing pinned Search-R1 checkout; run bash scripts/bootstrap_searchr1.sh" >&2
  exit 1
}
CHECKPOINT_DIR=$("$SEARCH_R1_PYTHON" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$CHECKPOINT_DIR")
COMPLETE_MARKER=$CHECKPOINT_DIR/.complete.json
for data_file in "$TRAIN_DATA" "$VAL_DATA"; do
  [[ -s "$data_file" ]] || {
    echo "Missing Search-R1 data: $data_file" >&2
    echo "Run: bash hard_rq0/prepare_data.sh" >&2
    exit 1
  }
done
[[ -s "$CORPUS_PATH" ]] || {
  echo "Missing hard-RQ0 corpus: $CORPUS_PATH" >&2
  echo "Run: bash hard_rq0/download_assets.sh" >&2
  exit 1
}
[[ -s "$ASSET_MANIFEST" ]] || {
  echo "Missing hard-RQ0 asset manifest: $ASSET_MANIFEST" >&2
  echo "Run: bash hard_rq0/download_assets.sh" >&2
  exit 1
}
"$PILOT_PYTHON" -m stackpilot.prepare_hard_rq0 \
  --config "$ROOT/configs/hard_rq0.yaml" --check
"$PILOT_PYTHON" -m stackpilot.prepare_hard_rq0 \
  --config "$ROOT/configs/hard_rq0.yaml" \
  --validate-training-inputs \
  --train-file "$TRAIN_DATA" \
  --val-file "$VAL_DATA"
if [[ "$BACKEND" == bm25 ]]; then
  [[ -d "$INDEX_PATH" ]] || {
    echo "Missing hard-RQ0 BM25 index: $INDEX_PATH" >&2
    exit 1
  }
else
  [[ -s "$INDEX_PATH" ]] || {
    echo "Missing hard-RQ0 E5 index: $INDEX_PATH" >&2
    exit 1
  }
fi

source "$ROOT/scripts/lib/runtime.sh"
if [[ "$N_GPUS" != 7 ]]; then
  echo "Hard-RQ0 training reserves GPU 7 for E5 and requires N_GPUS=7; got '$N_GPUS'." >&2
  exit 2
fi
validate_gpu_list "$TRAIN_GPUS" 7 "Hard-RQ0 Search-R1 training" || exit 1
validate_gpu_list "$E5_GPU" 1 "Hard-RQ0 E5 retrieval" || exit 1
IFS=',' read -r -a train_gpu_ids <<< "$TRAIN_GPUS"
IFS=',' read -r -a e5_gpu_ids <<< "$E5_GPU"
declare -A assigned_gpu_ids=()
for gpu_id in "${train_gpu_ids[@]}" "${e5_gpu_ids[@]}"; do
  gpu_index=$((10#$gpu_id))
  if (( gpu_index > 7 )); then
    echo "Hard-RQ0 GPU IDs must be logical device indices 0-7; got '$gpu_id'." >&2
    exit 2
  fi
  if [[ -n ${assigned_gpu_ids[$gpu_index]+x} ]]; then
    echo "Hard-RQ0 GPU assignments overlap at logical device $gpu_index: TRAIN_GPUS=$TRAIN_GPUS E5_GPU=$E5_GPU." >&2
    exit 2
  fi
  assigned_gpu_ids[$gpu_index]=1
done
if [[ ! "$SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer; got '$SEED'." >&2
  exit 2
fi
for numeric_name in TOTAL_UPDATES TOTAL_EPOCHS TRAIN_BATCH VAL_BATCH MINI_BATCH \
  MICRO_BATCH LOG_PROB_MICRO_BATCH N_AGENT TOPK PORT SAVE_FREQ; do
  numeric_value=${!numeric_name}
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric_name must be a positive integer; got '$numeric_value'." >&2
    exit 2
  fi
done
if (( PORT > 65535 )); then
  echo "PORT must be at most 65535; got '$PORT'." >&2
  exit 2
fi
if [[ ! "$TEST_FREQ" =~ ^-1$|^[1-9][0-9]*$ ]]; then
  echo "TEST_FREQ must be -1 or a positive integer; got '$TEST_FREQ'." >&2
  exit 2
fi
if [[ "$VAL_BEFORE_TRAIN" != true && "$VAL_BEFORE_TRAIN" != false ]]; then
  echo "VAL_BEFORE_TRAIN must be true or false; got '$VAL_BEFORE_TRAIN'." >&2
  exit 2
fi
for boolean_name in ACTOR_PARAM_OFFLOAD ACTOR_GRAD_OFFLOAD \
  ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD; do
  boolean_value=${!boolean_name}
  if [[ "$boolean_value" != true && "$boolean_value" != false ]]; then
    echo "$boolean_name must be true or false; got '$boolean_value'." >&2
    exit 2
  fi
done
if (( TRAIN_BATCH % 7 != 0 || VAL_BATCH % 7 != 0 || MINI_BATCH % 7 != 0 || \
      MICRO_BATCH % 7 != 0 || LOG_PROB_MICRO_BATCH % 7 != 0 )); then
  echo "TRAIN_BATCH, VAL_BATCH, MINI_BATCH, MICRO_BATCH, and LOG_PROB_MICRO_BATCH must be divisible by 7." >&2
  exit 2
fi
if (( MINI_BATCH / 7 < MICRO_BATCH / 7 || (MINI_BATCH / 7) % (MICRO_BATCH / 7) != 0 )); then
  echo "Per-GPU MINI_BATCH must be a positive multiple of per-GPU MICRO_BATCH." >&2
  exit 2
fi
if (( (TRAIN_BATCH * N_AGENT) % MINI_BATCH != 0 )); then
  echo "TRAIN_BATCH*N_AGENT must be divisible by MINI_BATCH; got $TRAIN_BATCH*$N_AGENT/$MINI_BATCH." >&2
  exit 2
fi
if (( SAVE_FREQ > TOTAL_UPDATES || TOTAL_UPDATES % SAVE_FREQ != 0 )); then
  echo "SAVE_FREQ must divide TOTAL_UPDATES and include the final update; got $SAVE_FREQ/$TOTAL_UPDATES." >&2
  exit 2
fi
if ! awk -v value="$ROLLOUT_GPU_MEMORY" 'BEGIN { exit !(value > 0 && value < 1) }'; then
  echo "ROLLOUT_GPU_MEMORY must be between 0 and 1; got '$ROLLOUT_GPU_MEMORY'." >&2
  exit 2
fi

TRAIN_MIN_DISK_GIB=${TRAIN_MIN_DISK_GIB:-15}
if [[ ! "$TRAIN_MIN_DISK_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_MIN_DISK_GIB must be a positive integer; got '$TRAIN_MIN_DISK_GIB'." >&2
  exit 2
fi
bash "$ROOT/scripts/apply_searchr1_runtime_patch.sh"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" \
  --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_validation.py" \
  --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_action_protocol.py" \
  --search-r1-root "$SEARCH_R1"
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_reward_protocol.py" \
  --search-r1-root "$SEARCH_R1"
if [[ "$SEARCH_R1_REWARD_MODE" == evidence ]]; then
  "$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_evidence_reward.py" \
    --search-r1-root "$SEARCH_R1"
fi
"$SEARCH_R1_PYTHON" "$ROOT/hard_rq0/patch_searchr1_experiment_env.py" \
  --search-r1-root "$SEARCH_R1"
"$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
bash "$ROOT/scripts/stop_servers.sh" >/dev/null 2>&1 || true
STAGE2_MIN_DISK_GIB=$TRAIN_MIN_DISK_GIB bash "$ROOT/scripts/preflight_searchr1.sh"

AVAILABLE_UPDATES=$("$SEARCH_R1_PYTHON" - \
  "$TRAIN_DATA" "$VAL_DATA" "$TRAIN_BATCH" "$VAL_BATCH" "$TOTAL_EPOCHS" <<'PY'
import sys

import pyarrow.parquet as pq

train_path, val_path, train_batch, val_batch, epochs = sys.argv[1:]
train_batch, val_batch, epochs = int(train_batch), int(val_batch), int(epochs)
train_rows = pq.ParquetFile(train_path).metadata.num_rows
val_rows = pq.ParquetFile(val_path).metadata.num_rows
steps_per_epoch = train_rows // train_batch
if steps_per_epoch < 1:
    raise SystemExit(f"Training data has {train_rows} rows, fewer than TRAIN_BATCH={train_batch}")
if val_rows < val_batch:
    raise SystemExit(f"Validation data has {val_rows} rows, fewer than VAL_BATCH={val_batch}")
print(steps_per_epoch * epochs)
PY
)
if (( TOTAL_UPDATES > AVAILABLE_UPDATES )); then
  echo "Requested $TOTAL_UPDATES updates, but the data and TOTAL_EPOCHS provide only $AVAILABLE_UPDATES." >&2
  exit 2
fi

# The pinned trainer starts at global_steps=1, increments after each update,
# and returns on >= total_training_steps. N actual updates therefore require N+1.
TRAINER_STOP_STEP=$((TOTAL_UPDATES + 1))

BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
  bash "$ROOT/scripts/resolve_hf_model.sh" \
    "$BASE_MODEL" "$BASE_MODEL_REVISION" "$SEARCH_R1_PYTHON")
RETRIEVER_MODEL=none
RETRIEVER_MODEL_REVISION=
if [[ "$BACKEND" == e5 ]]; then
  RETRIEVER_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
    bash "$ROOT/scripts/resolve_hf_model.sh" \
      "$E5_MODEL_SOURCE" "$E5_MODEL_REVISION" "$SEARCH_R1_PYTHON")
  RETRIEVER_MODEL_REVISION=$(
    "$SEARCH_R1_PYTHON" -c \
      'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); print(p.name if p.parent.name == "snapshots" else "local")' \
      "$RETRIEVER_MODEL"
  )
fi
SEARCH_R1_DIRTY_SHA=$(git -C "$SEARCH_R1" diff --no-ext-diff --binary HEAD | sha256sum | awk '{print $1}')

TRAIN_SIGNATURE=$("$SEARCH_R1_PYTHON" - \
  "$TRAIN_DATA" "$VAL_DATA" "$BASE_MODEL" "$BASE_MODEL_REVISION" \
  "$BACKEND" "$SEED" "$PROFILE" "$TRAIN_BATCH" "$VAL_BATCH" \
  "$MINI_BATCH" "$MICRO_BATCH" "$LOG_PROB_MICRO_BATCH" "$N_AGENT" \
  "$TOPK" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_START_LENGTH" \
  "$MAX_OBS_LENGTH" "$MAX_TURNS" \
  "$TOTAL_UPDATES" "$TRAINER_STOP_STEP" "$TOTAL_EPOCHS" \
  "$SAVE_FREQ" "$TEST_FREQ" "$VAL_BEFORE_TRAIN" "$ROLLOUT_GPU_MEMORY" "$ATTENTION_BACKEND" \
  "$ACTOR_PARAM_OFFLOAD" "$ACTOR_GRAD_OFFLOAD" "$ACTOR_OPTIMIZER_OFFLOAD" "$REF_PARAM_OFFLOAD" \
  "$TRAIN_GPUS" "$N_GPUS" "$E5_GPU" "$PORT" "$SEARCH_R1_COMMIT" "$SEARCH_R1_DIRTY_SHA" \
  "$SEARCH_R1_REWARD_MODE" "$ANSWER_REWARD_WEIGHT" "$EVIDENCE_REWARD_WEIGHT" "$SEARCH_COST_WEIGHT" \
  "$RETRIEVER_MODEL" "$E5_MODEL_REVISION" "$RETRIEVER_MODEL_REVISION" \
  "$ROOT/searchr1_stage2/searchr1-runtime.patch" \
  "$ROOT/hard_rq0/patch_searchr1_seed.py" \
  "$ROOT/hard_rq0/patch_searchr1_worker_cuda.py" \
  "$ROOT/hard_rq0/patch_searchr1_validation.py" \
  "$ROOT/hard_rq0/patch_searchr1_action_protocol.py" \
  "$ROOT/hard_rq0/patch_searchr1_reward_protocol.py" \
  "$ROOT/hard_rq0/patch_searchr1_evidence_reward.py" \
  "$ROOT/hard_rq0/patch_searchr1_experiment_env.py" \
  "$ROOT/stackpilot/action_protocol.py" \
  "$ROOT/hard_rq0/sitecustomize.py" \
  "$ROOT/hard_rq0/train_specialist.sh" "$ASSET_MANIFEST" "$DATA_MANIFEST" \
  "$CORPUS_PATH" "$INDEX_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    train_file,
    val_file,
    base_model,
    base_model_revision,
    backend,
    seed,
    profile,
    train_batch,
    val_batch,
    mini_batch,
    micro_batch,
    log_prob_micro_batch,
    n_agent,
    topk,
    max_prompt_length,
    max_response_length,
    max_start_length,
    max_obs_length,
    max_turns,
    total_updates,
    trainer_stop_step,
    total_epochs,
    save_freq,
    test_freq,
    val_before_train,
    rollout_gpu_memory,
    attention_backend,
    actor_param_offload,
    actor_grad_offload,
    actor_optimizer_offload,
    ref_param_offload,
    train_gpus,
    n_gpus,
    e5_gpu,
    port,
    search_r1_commit,
    search_r1_dirty_sha,
    reward_mode,
    answer_reward_weight,
    evidence_reward_weight,
    search_cost_weight,
    retriever_model,
    e5_model_revision,
    retriever_model_revision,
    runtime_patch,
    seed_patch,
    worker_cuda_patch,
    validation_patch,
    action_protocol_patch,
    reward_protocol_patch,
    evidence_reward_patch,
    experiment_env_patch,
    action_protocol,
    sitecustomize,
    training_wrapper,
    asset_manifest,
    data_manifest,
    corpus_path,
    index_path,
) = sys.argv[1:]


def digest(path: str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_identity(path_text: str) -> dict:
    root = Path(path_text).resolve()
    files = {}
    for pattern in ("config.json", "tokenizer*", "*.index.json", "*.safetensors", "*.bin"):
        for path in root.glob(pattern):
            if path.is_file():
                files[path.name] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    return {"resolved_path": str(root), "files": files}


def asset_identity(path_text: str) -> dict:
    path = Path(path_text)
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {"kind": "file", "size": stat.st_size}
    files = []
    for child in sorted(resolved.rglob("*")):
        if child.is_file():
            files.append(
                {
                    "name": str(child.relative_to(resolved)),
                    "size": child.stat().st_size,
                }
            )
    return {"kind": "directory", "files": files}


payload = {
    "schema": 2,
    "base_model": model_identity(base_model),
    "base_model_revision": base_model_revision,
    "backend": backend,
    "retriever_model": (
        model_identity(retriever_model) if retriever_model != "none" else None
    ),
    "e5_model_revision": e5_model_revision if retriever_model != "none" else None,
    "retriever_model_revision": (
        retriever_model_revision if retriever_model != "none" else None
    ),
    "seed": int(seed),
    "profile": profile,
    "train_sha256": digest(train_file),
    "val_sha256": digest(val_file),
    "data_manifest_sha256": digest(data_manifest),
    "assets": {
        "manifest_sha256": digest(asset_manifest),
        "corpus": asset_identity(corpus_path),
        "index": asset_identity(index_path),
    },
    "protocol": {
        "train_batch": int(train_batch),
        "val_batch": int(val_batch),
        "mini_batch": int(mini_batch),
        "micro_batch": int(micro_batch),
        "log_prob_micro_batch": int(log_prob_micro_batch),
        "n_agent": int(n_agent),
        "topk": int(topk),
        "max_prompt_length": int(max_prompt_length),
        "max_response_length": int(max_response_length),
        "max_start_length": int(max_start_length),
        "max_obs_length": int(max_obs_length),
        "total_updates": int(total_updates),
        "trainer_stop_step": int(trainer_stop_step),
        "total_epochs": int(total_epochs),
        "save_freq": int(save_freq),
        "test_freq": int(test_freq),
        "val_before_train": val_before_train == "true",
        "rollout_gpu_memory": float(rollout_gpu_memory),
        "attention_backend": attention_backend,
        "actor_param_offload": actor_param_offload == "true",
        "actor_grad_offload": actor_grad_offload == "true",
        "actor_optimizer_offload": actor_optimizer_offload == "true",
        "ref_param_offload": ref_param_offload == "true",
        "train_gpus": train_gpus,
        "n_gpus": int(n_gpus),
        "e5_gpu": int(e5_gpu),
        "retriever_port": int(port),
        "max_turns": int(max_turns),
        "reward_mode": reward_mode,
        "answer_reward_weight": float(answer_reward_weight),
        "evidence_reward_weight": float(evidence_reward_weight),
        "search_cost_weight": float(search_cost_weight),
    },
    "search_r1_commit": search_r1_commit,
    "search_r1_dirty_sha256": search_r1_dirty_sha,
    "runtime_patch_sha256": digest(runtime_patch),
    "seed_patch_sha256": digest(seed_patch),
    "worker_cuda_patch_sha256": digest(worker_cuda_patch),
    "validation_patch_sha256": digest(validation_patch),
    "action_protocol_patch_sha256": digest(action_protocol_patch),
    "reward_protocol_patch_sha256": digest(reward_protocol_patch),
    "evidence_reward_patch_sha256": digest(evidence_reward_patch),
    "experiment_env_patch_sha256": digest(experiment_env_patch),
    "action_protocol_sha256": digest(action_protocol),
    "sitecustomize_sha256": digest(sitecustomize),
    "training_wrapper_sha256": digest(training_wrapper),
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(canonical.encode()).hexdigest())
PY
)

validate_checkpoint() {
  local checkpoint=$1
  "$SEARCH_R1_PYTHON" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = root / "config.json"
if not config.is_file():
    raise SystemExit(1)
try:
    json.loads(config.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
index_files = sorted(root.glob("*.safetensors.index.json")) or sorted(root.glob("*.bin.index.json"))
if len(index_files) > 1:
    raise SystemExit(1)
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weights = [root / name for name in sorted(set(index["weight_map"].values()))]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise SystemExit(1)
else:
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
if not weights or any(not path.is_file() or path.stat().st_size < 1024 * 1024 for path in weights):
    raise SystemExit(1)
if not (root / "tokenizer.json").is_file() and not (root / "tokenizer.model").is_file():
    raise SystemExit(1)
PY
}

latest_checkpoint() {
  local candidate=$CHECKPOINT_DIR/actor/global_step_$TOTAL_UPDATES
  [[ -d "$candidate" ]] || return 1
  validate_checkpoint "$candidate" || return 1
  printf '%s\n' "$candidate"
}

if [[ ${FORCE_TRAIN:-0} != 1 && -f "$COMPLETE_MARKER" ]]; then
  marker_signature=$("$SEARCH_R1_PYTHON" - "$COMPLETE_MARKER" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["training_signature"])
except (OSError, KeyError, TypeError, json.JSONDecodeError):
    print("")
PY
  )
  if [[ "$marker_signature" == "$TRAIN_SIGNATURE" ]] && completed_checkpoint=$(latest_checkpoint); then
    echo "Reusing completed $EXP checkpoint: $completed_checkpoint"
    exit 0
  fi
  echo "Completion marker does not match this hard-RQ0 training configuration; training will restart." >&2
fi

if [[ -e "$CHECKPOINT_DIR" ]]; then
  case "$CHECKPOINT_DIR/" in
    "$ROOT/work/hard_rq0/checkpoints/"*) ;;
    *)
      echo "Refusing to rotate nonstandard existing CHECKPOINT_DIR: $CHECKPOINT_DIR" >&2
      echo "Choose a fresh path or move it manually." >&2
      exit 1
      ;;
  esac
  if [[ ${FORCE_TRAIN:-0} == 1 ]]; then
    backup_kind=previous
  else
    backup_kind=incomplete
  fi
  checkpoint_backup="${CHECKPOINT_DIR}.${backup_kind}.$(date +%Y%m%d-%H%M%S).$$"
  mv -- "$CHECKPOINT_DIR" "$checkpoint_backup"
  echo "Moved prior training output to: $checkpoint_backup"
fi
mkdir -p "$CHECKPOINT_DIR" "$(dirname "$LOG_FILE")"

if [[ ${ENABLE_WANDB:-0} == 1 ]]; then
  LOGGER="['console','wandb']"
else
  export WANDB_MODE=disabled
fi

ensure_local_no_proxy
health_response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 30 \
  "http://127.0.0.1:${PORT}/health") || {
  echo "Retriever $BACKEND is not ready on port $PORT; run hard_rq0/launch_retrievers.sh" >&2
  exit 1
}
"$SEARCH_R1_PYTHON" -c \
  'import json,sys; from pathlib import Path; p=json.loads(sys.argv[1]); expected_backend, expected_index, expected_corpus, expected_e5_gpu, expected_model, expected_revision=sys.argv[2:]; ok=p.get("status")=="ok" and p.get("backend")==expected_backend and Path(p.get("index_path", "")).resolve()==Path(expected_index).resolve() and Path(p.get("corpus_path", "")).resolve()==Path(expected_corpus).resolve() and (expected_backend!="e5" or (p.get("faiss_gpu") is True and int(p.get("faiss_gpu_count", 0))==1 and str(p.get("cuda_visible_devices"))==expected_e5_gpu and Path(p.get("retriever_model", "")).resolve()==Path(expected_model).resolve() and str(p.get("retriever_model_revision") or "")==expected_revision)); sys.exit(None if ok else f"Unexpected retriever health: {p}")' \
  "$health_response" "$BACKEND" "$INDEX_PATH" "$CORPUS_PATH" "$E5_GPU" \
  "$RETRIEVER_MODEL" "$RETRIEVER_MODEL_REVISION"
probe_response=$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 120 \
  -X POST "http://127.0.0.1:${PORT}/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who wrote Hamlet?"],"topk":1,"return_scores":true}')
"$SEARCH_R1_PYTHON" -c \
  'import json,sys; p=json.loads(sys.argv[1]); ok=bool(p.get("result") and p["result"][0]); sys.exit(None if ok else f"Retriever probe returned no results: {p}")' \
  "$probe_response"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  "$ROOT/.venv-searchr1/bin/ray" stop --force >/dev/null 2>&1 || true
  if [[ $status -ne 0 ]]; then
    echo "Hard-RQ0 training failed; log: $LOG_FILE" >&2
    tail -n 100 "$LOG_FILE" >&2 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export VLLM_ATTENTION_BACKEND=$ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false
export RQ0_SEED=$SEED
export PYTHONHASHSEED=$SEED
export PYTHONPATH="$ROOT:$ROOT/hard_rq0:$SEARCH_R1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export SEARCH_R1_RETRIEVER_TIMEOUT=${SEARCH_R1_RETRIEVER_TIMEOUT:-120}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Training $EXP: backend=$BACKEND seed=$SEED profile=$PROFILE updates=$TOTAL_UPDATES GPUs=$TRAIN_GPUS"
echo "Base model: $BASE_MODEL; checkpoints: $CHECKPOINT_DIR"

cd "$SEARCH_R1"
"$SEARCH_R1_PYTHON" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" \
  data.val_batch_size="$VAL_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" \
  data.max_obs_length="$MAX_OBS_LENGTH" \
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
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="$ACTOR_GRAD_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.state_masking=true \
  algorithm.no_think_rl=false \
  trainer.logger="$LOGGER" \
  +trainer.val_only=false \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=StackAdaptHardRQ0 \
  trainer.experiment_name="$EXP" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TRAINER_STOP_STEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.default_hdfs_dir=null \
  max_turns="$MAX_TURNS" \
  retriever.url="http://127.0.0.1:${PORT}/retrieve" \
  retriever.topk="$TOPK" \
  2>&1 | tee "$LOG_FILE"

completed_checkpoint=$(latest_checkpoint) || {
  echo "Training exited successfully but exact final checkpoint global_step_$TOTAL_UPDATES is missing or invalid." >&2
  exit 1
}
"$SEARCH_R1_PYTHON" - \
  "$COMPLETE_MARKER" "$completed_checkpoint" "$EXP" "$BACKEND" "$SEED" \
  "$PROFILE" "$TOTAL_UPDATES" "$TRAINER_STOP_STEP" "$TRAIN_SIGNATURE" \
  "$BASE_MODEL" "$TOPK" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" \
  "$MAX_START_LENGTH" "$MAX_OBS_LENGTH" "$MAX_TURNS" "$SEARCH_R1_COMMIT" \
  "$SEARCH_R1_REWARD_MODE" "$ANSWER_REWARD_WEIGHT" \
  "$EVIDENCE_REWARD_WEIGHT" "$SEARCH_COST_WEIGHT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    marker,
    checkpoint,
    experiment,
    backend,
    seed,
    profile,
    updates,
    stop_step,
    signature,
    base_model,
    topk,
    max_prompt_length,
    max_response_length,
    max_start_length,
    max_obs_length,
    max_turns,
    search_r1_commit,
    reward_mode,
    answer_reward_weight,
    evidence_reward_weight,
    search_cost_weight,
) = sys.argv[1:]
payload = {
    "schema": 2,
    "experiment": experiment,
    "backend": backend,
    "seed": int(seed),
    "profile": profile,
    "total_updates": int(updates),
    "trainer_stop_step": int(stop_step),
    "training_signature": signature,
    "checkpoint": str(Path(checkpoint).resolve()),
    "base_model": str(Path(base_model).resolve()),
    "retrieval_topk": int(topk),
    "rollout_protocol": {
        "max_prompt_length": int(max_prompt_length),
        "max_response_length": int(max_response_length),
        "max_start_length": int(max_start_length),
        "max_obs_length": int(max_obs_length),
        "max_turns": int(max_turns),
    },
    "reward_protocol": {
        "mode": reward_mode,
        "answer_weight": float(answer_reward_weight),
        "evidence_weight": float(evidence_reward_weight),
        "search_cost_weight": float(search_cost_weight),
    },
    "search_r1_commit": search_r1_commit,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
marker_path = Path(marker)
temporary = marker_path.with_name(f"{marker_path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
PY
echo "Completed $EXP: $completed_checkpoint"

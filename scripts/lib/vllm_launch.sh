#!/usr/bin/env bash

configure_vllm_launch() {
  local root=$1
  local physical_gpus
  local id
  local require_local_model=0

  VLLM_PYTHON=$root/.venv-vllm/bin/python
  VLLM_BIN=$root/.venv-vllm/bin/vllm
  if [[ ! -x "$VLLM_PYTHON" || ! -x "$VLLM_BIN" ]]; then
    echo "Missing .venv-vllm. Run: bash scripts/bootstrap_vllm.sh" >&2
    return 1
  fi

  if [[ -n "${MODEL_PATH+x}" ]]; then
    require_local_model=1
  fi
  MODEL_PATH=${MODEL_PATH:-${MODEL:-Qwen/Qwen2.5-7B-Instruct}}
  SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
  LLM_GPUS=${LLM_GPUS:-0,1,2,3}
  TP=${TP:-4}
  LLM_PORT=${LLM_PORT:-9000}
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

  validate_gpu_list "$LLM_GPUS" "$TP" "vLLM tensor parallelism" || return 1
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to launch vLLM." >&2
    return 1
  fi
  physical_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  IFS=',' read -r -a selected_gpu_ids <<< "$LLM_GPUS"
  for id in "${selected_gpu_ids[@]}"; do
    if (( id >= physical_gpus )); then
      echo "GPU $id does not exist; this node exposes $physical_gpus GPUs." >&2
      return 1
    fi
  done

  if [[ $require_local_model -eq 1 || "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || -e "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
      echo "Local MODEL_PATH must be a Hugging Face model directory with config.json: $MODEL_PATH" >&2
      return 1
    fi
    "$VLLM_PYTHON" - "$MODEL_PATH" "$TP" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1]).expanduser().resolve()
tp = int(sys.argv[2])
try:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid model config in {model_path}: {exc}") from exc

heads = int(config.get("num_attention_heads", 0))
if heads <= 0 or heads % tp:
    raise SystemExit(f"num_attention_heads={heads} is incompatible with TP={tp}")

index_files = sorted(model_path.glob("*.safetensors.index.json"))
weights: list[Path]
if index_files:
    try:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid safetensors index {index_files[0]}: {exc}") from exc
    weights = [model_path / name for name in shard_names]
else:
    weights = sorted(model_path.glob("*.safetensors")) or sorted(model_path.glob("*.bin"))
if not weights:
    raise SystemExit(f"No model weights found in {model_path}")
bad = [str(path) for path in weights if not path.is_file() or path.stat().st_size < 1024 * 1024]
if bad:
    raise SystemExit(f"Missing or incomplete model weight shards: {bad}")
if not (model_path / "tokenizer.json").is_file() and not (model_path / "tokenizer.model").is_file():
    raise SystemExit(f"Tokenizer files are missing from {model_path}")
print(f"Validated local model: {model_path} ({len(weights)} weight file(s), TP={tp})")
PY
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  fi

  VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
  export MODEL_PATH SERVED_MODEL_NAME LLM_GPUS TP LLM_PORT VLLM_NO_USAGE_STATS
  # Consumed by the launcher scripts after this helper is sourced.
  # shellcheck disable=SC2034
  VLLM_ARGS=(
    serve "$MODEL_PATH"
    --host 127.0.0.1
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$TP"
    --distributed-executor-backend mp
    --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --port "$LLM_PORT"
  )
}

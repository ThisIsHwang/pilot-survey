#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-vllm/bin/activate

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
LLM_GPUS=${LLM_GPUS:-0,1,2,3}
TP=${TP:-4}
CUDA_VISIBLE_DEVICES=$LLM_GPUS python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.88 \
  --max-model-len 16384 \
  --port 9000

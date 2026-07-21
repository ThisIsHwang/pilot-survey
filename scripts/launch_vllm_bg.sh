#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv-vllm/bin/activate
mkdir -p logs work/pids
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
LLM_GPUS=${LLM_GPUS:-0,1,2,3}
TP=${TP:-4}
CUDA_VISIBLE_DEVICES=$LLM_GPUS nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.88 \
  --max-model-len 16384 --port 9000 > logs/vllm.log 2>&1 &
echo $! > work/pids/vllm.pid
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:9000/v1/models >/dev/null 2>&1; then
    echo "vLLM ready"
    exit 0
  fi
  sleep 5
done
echo "vLLM did not become ready. Check logs/vllm.log" >&2
exit 1

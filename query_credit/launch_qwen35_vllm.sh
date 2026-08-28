#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

VLLM_PYTHON=$ROOT/.venv-vllm/bin/python
VLLM_BIN=$ROOT/.venv-vllm/bin/vllm
QWEN_PYTHON=$ROOT/.venv-qwen35/bin/python
[[ -x "$VLLM_BIN" && -x "$VLLM_PYTHON" ]] || {
  echo "Run scripts/bootstrap_vllm.sh first." >&2
  exit 1
}
[[ -x "$QWEN_PYTHON" ]] || {
  echo "Run scripts/bootstrap_qwen35.sh first." >&2
  exit 1
}
: "${MODEL_PATH:?MODEL_PATH must point to the resolved Qwen3.5 snapshot}"

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3.5-9B}
LLM_GPUS=${LLM_GPUS:-0}
TP=${TP:-1}
DP=${DP:-1}
VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-$DP}
LLM_PORT=${LLM_PORT:-9000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_BATCH_INVARIANT=0
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$VLLM_PYTHON" - <<'PY'
from packaging.version import Version
import vllm
assert Version(vllm.__version__) >= Version("0.19.0"), vllm.__version__
print(f"Validated vLLM {vllm.__version__} for Qwen3.5 serving")
PY

validate_gpu_list "$LLM_GPUS" "$((TP * DP))" "Qwen3.5 vLLM TP=$TP x DP=$DP"
"$QWEN_PYTHON" - "$MODEL_PATH" "$TP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
tp = int(sys.argv[2])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
text = config.get("text_config", config)
model_type = str(text.get("model_type", config.get("model_type", "")))
if model_type not in {"qwen3_5", "qwen3_5_text"}:
    raise SystemExit(f"Expected Qwen3.5 text config, found model_type={model_type!r}")
heads = int(text.get("num_attention_heads", 0))
if heads <= 0 or heads % tp:
    raise SystemExit(f"num_attention_heads={heads} is incompatible with TP={tp}")
if not list(root.glob("*.safetensors")) and not list(root.glob("*.safetensors.index.json")):
    raise SystemExit(f"No safetensors weights found in {root}")
print(f"Validated Qwen3.5 snapshot: {root}; attention heads={heads}; TP={tp}")
PY

RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/query_credit_weekend/runtime}
RUNTIME_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/query_credit_weekend}
mkdir -p "$RUNTIME_WORK_ROOT/pids" "$RUNTIME_LOG_ROOT"
PID_FILE=$RUNTIME_WORK_ROOT/pids/vllm.pid
LOG_FILE=$RUNTIME_LOG_ROOT/vllm_qwen35.log
stop_managed_pid "$PID_FILE" "$VLLM_BIN" "$ROOT" 1 || true
require_free_port "$VLLM_PYTHON" "$LLM_PORT"

ARGS=(
  serve "$MODEL_PATH"
  --host 127.0.0.1
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --data-parallel-size "$DP"
  --api-server-count "$VLLM_API_SERVER_COUNT"
  --distributed-executor-backend mp
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --language-model-only
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking": false}'
  --port "$LLM_PORT"
)
if [[ -n "$VLLM_ATTENTION_BACKEND" ]]; then
  ARGS+=(--attention-backend "$VLLM_ATTENTION_BACKEND")
fi

VLLM_PID=$(CUDA_VISIBLE_DEVICES="$LLM_GPUS" start_managed_process \
  "$VLLM_PYTHON" "$LOG_FILE" "$VLLM_BIN" "${ARGS[@]}")
echo "$VLLM_PID" > "$PID_FILE"

cleanup_started() {
  local status=$?
  trap - EXIT INT TERM
  stop_managed_pid "$PID_FILE" "$VLLM_BIN" "$ROOT" 1 || true
  exit "$status"
}
trap cleanup_started EXIT INT TERM
wait_for_http "$VLLM_PID" "http://127.0.0.1:${LLM_PORT}/v1/models" \
  "${VLLM_READY_TIMEOUT:-1800}" "$LOG_FILE"

# Qwen3.5 GDN currently cannot use vLLM's batch-invariant kernels. Instead,
# max-num-seqs=1 prevents within-engine batching. This probe sends identical
# seeded requests across all data-parallel engines and requires byte-identical,
# non-thinking responses before the scientific run starts.
"$QWEN_PYTHON" - "$LLM_PORT" "$SERVED_MODEL_NAME" "$DP" \
  "$RUNTIME_WORK_ROOT/qwen35_runtime_contract.json" <<'PY'
from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path

import requests

port, model, dp, output = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
url = f"http://127.0.0.1:{port}/v1/chat/completions"
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "Follow the requested output format exactly."},
        {"role": "user", "content": "Return exactly <answer>OK</answer>."},
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "seed": 20260828,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": False},
}

def call(_: int) -> str:
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    content = str(message.get("content") or "")
    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    ).strip()
    if reasoning or "<think" in content.lower() or "</think" in content.lower():
        raise RuntimeError(f"Thinking leakage detected: {message}")
    if not content.strip():
        raise RuntimeError(f"Empty Qwen3.5 probe response: {body}")
    return content

count = max(2, dp * 2)
with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, dp)) as pool:
    values = list(pool.map(call, range(count)))
unique = sorted(set(values))
if len(unique) != 1:
    raise SystemExit(
        "Identical seeded requests were not deterministic with max-num-seqs=1: "
        + repr(unique)
    )
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "schema": 1,
            "model": model,
            "enable_thinking": False,
            "language_model_only": True,
            "reasoning_parser": "qwen3",
            "batch_invariant": False,
            "max_num_seqs": 1,
            "data_parallel_size": dp,
            "probe_requests": count,
            "unique_probe_outputs": len(unique),
            "probe_output": unique[0],
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
print("Qwen3.5 non-thinking and deterministic single-sequence probe passed.")
PY

trap - EXIT INT TERM
echo "Qwen3.5-9B non-thinking vLLM ready on port $LLM_PORT (PID $VLLM_PID)."

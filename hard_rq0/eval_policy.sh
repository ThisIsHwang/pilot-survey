#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?Set TAG, e.g. base-qwen or bm25-specialist}
SEED=${SEED:-0}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a Hugging Face checkpoint directory or repo ID}
LIMIT=${LIMIT:-}
RESULT_SET=${RESULT_SET:-pilot}
DATA_FILE=${DATA_FILE:-$ROOT/work/hard_rq0/data/eval_all.jsonl}
BACKENDS=${BACKENDS:-"bm25 e5"}
TOPKS=${TOPKS:-"3 5 10"}

[[ "$RESULT_SET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RESULT_SET=$RESULT_SET" >&2; exit 2; }
EVAL_CONFIG=$ROOT/work/hard_rq0/configs/${RESULT_SET}.yaml
mkdir -p "$(dirname "$EVAL_CONFIG")"
"$ROOT/.venv-pilot/bin/python" - configs/hard_rq0.yaml "$EVAL_CONFIG" "$ROOT" "$RESULT_SET" <<'PY'
import sys
from pathlib import Path
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
root = Path(sys.argv[3])
result_set = sys.argv[4]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["work_dir"] = str(root / "work" / "hard_rq0" / "runs" / result_set)
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

export MODEL_PATH
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
export LLM_GPUS=${LLM_GPUS:-0,1}
export TP=${TP:-2}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

curl --noproxy '*' -fsS http://127.0.0.1:8101/health >/dev/null || {
  echo "Hard-RQ0 BM25 is not running; run hard_rq0/launch_retrievers.sh" >&2
  exit 1
}
curl --noproxy '*' -fsS http://127.0.0.1:8102/health >/dev/null || {
  echo "Hard-RQ0 E5 is not running; run hard_rq0/launch_retrievers.sh" >&2
  exit 1
}
[[ -f "$DATA_FILE" ]] || { echo "Missing $DATA_FILE; run hard_rq0/prepare_data.sh" >&2; exit 1; }

bash "$ROOT/scripts/stop_servers.sh" || true
bash "$ROOT/scripts/launch_vllm_bg.sh"

ARGS=(
  --config "$EVAL_CONFIG"
  --data-file "$DATA_FILE"
  --tag "$TAG"
  --seed "$SEED"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
# shellcheck disable=SC2206
BACKEND_ARGS=($BACKENDS)
# shellcheck disable=SC2206
TOPK_ARGS=($TOPKS)
ARGS+=(--backends "${BACKEND_ARGS[@]}" --topks "${TOPK_ARGS[@]}")

"$ROOT/.venv-pilot/bin/python" -m stackpilot.hard_policy_eval "${ARGS[@]}"

if [[ ${KEEP_VLLM:-0} != 1 ]]; then
  bash "$ROOT/scripts/stop_servers.sh"
fi

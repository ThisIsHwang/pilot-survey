#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=$ROOT/.venv-qwen35/bin/python
[[ -x "$PYTHON" ]] || {
  echo "Run scripts/bootstrap_qwen35.sh first." >&2
  exit 1
}
CONFIG=${WEEKEND_CONFIG:-configs/query_credit_weekend.yaml}
CAUSAL_CONFIG=${CAUSAL_QUERY_CONFIG:-configs/causal_query_weekend_qwen35.yaml}
RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/query_credit_weekend/runtime}
LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/query_credit_weekend}
mkdir -p "$RUNTIME_ROOT" "$LOG_ROOT"
export STACKPILOT_RUNTIME_ROOT=$RUNTIME_ROOT STACKPILOT_LOG_ROOT=$LOG_ROOT
export STACKPILOT_QWEN35_NO_THINK=1
export PYTHONPATH=$ROOT/query_credit/qwen35_site:$ROOT${PYTHONPATH:+:$PYTHONPATH}

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required." >&2; exit 1; }
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ ${PROFILE:-auto} == auto ]]; then
  if (( GPU_COUNT >= 8 )); then
    PROFILE=node8
  else
    PROFILE=single
  fi
fi
case "$PROFILE" in smoke|single|node8) ;; *) echo "PROFILE must be auto, smoke, single, or node8." >&2; exit 2;; esac
if [[ "$PROFILE" == node8 && $GPU_COUNT -lt 8 ]]; then
  echo "PROFILE=node8 requires at least 8 visible GPUs; found $GPU_COUNT." >&2
  exit 2
fi

"$PYTHON" - "$CONFIG" "$CAUSAL_CONFIG" <<'PY'
import sys
import yaml

weekend = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
causal = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
for name, cfg in (("weekend", weekend), ("causal", causal)):
    model = cfg["model"]
    if model.get("served_model_name", model.get("base_model")) != "Qwen/Qwen3.5-9B":
        raise SystemExit(f"{name} config does not target Qwen/Qwen3.5-9B: {model}")
    if model.get("enable_thinking") is not False:
        raise SystemExit(f"{name} config must set enable_thinking: false")
    if model.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise SystemExit(f"{name} config must disable thinking in the chat template")
budget = weekend["budget"]
if sum(int(value) for value in budget.values()) != 120:
    raise SystemExit(f"The declared five-day budget must sum to 120 hours: {budget}")
print("Qwen3.5-9B non-thinking configuration contract passed.")
PY

COLLECTION_HOURS=$("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['budget']['collection_hours'])
PY
)
IG_HOURS=$("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['budget']['ig_hours'])
PY
)
GRADIENT_HOURS=$("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['budget']['gradient_hours'])
PY
)
MICRO_HOURS=$("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['budget']['micro_hours'])
PY
)
WORK_DIR=$("$PYTHON" - "$CONFIG" <<'PY'
import os,sys,yaml
value=yaml.safe_load(open(sys.argv[1]))['work_dir']
print(os.path.abspath(value))
PY
)

write_manifest() {
  {
    echo "started_at_utc=$(date -u +%FT%TZ)"
    echo "profile=$PROFILE"
    echo "gpu_count=$GPU_COUNT"
    echo "model=Qwen/Qwen3.5-9B"
    echo "enable_thinking=false"
    echo "vllm_batch_invariant=false"
    echo "vllm_max_num_seqs=1"
    echo "declared_budget_hours=120"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
    echo "causal_config_sha256=$(sha256sum "$CAUSAL_CONFIG" | awk '{print $1}')"
    "$PYTHON" - <<'PY'
import accelerate, peft, torch, transformers
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"accelerate={accelerate.__version__}")
PY
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  } > "$RUNTIME_ROOT/run_manifest.txt"
}
write_manifest

LEAK_FINGERPRINT=$(printf '%s:%s:%s' \
  "$(sha256sum "$CONFIG" | awk '{print $1}')" \
  "$(sha256sum "$CAUSAL_CONFIG" | awk '{print $1}')" \
  "$PROFILE:$(git rev-parse HEAD 2>/dev/null || echo unknown)" | sha256sum | awk '{print $1}')
LEAK_MARKER=$RUNTIME_ROOT/thinking_leaks.fingerprint
export STACKPILOT_THINKING_LEAK_LOG=$RUNTIME_ROOT/thinking_leaks.jsonl
if [[ ! -f "$LEAK_MARKER" || "$(cat "$LEAK_MARKER")" != "$LEAK_FINGERPRINT" ]]; then
  : > "$STACKPILOT_THINKING_LEAK_LOG"
  printf '%s\n' "$LEAK_FINGERPRINT" > "$LEAK_MARKER"
fi

assert_no_thinking_leaks() {
  local leaks
  leaks=$("$PYTHON" - "$STACKPILOT_THINKING_LEAK_LOG" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.is_file() else 0)
PY
)
  if (( leaks > 0 )); then
    echo "Detected $leaks thinking-leak errors; refusing to analyze a mixed-mode run." >&2
    return 3
  fi
}

if [[ ${SKIP_COLLECTION:-0} != 1 ]]; then
  INPUT_COUNT=$("$PYTHON" - "$CONFIG" <<'PY'
import glob, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
raw = os.environ.get("QUERY_CREDIT_INPUTS", "").strip()
patterns = [value for value in raw.split(os.pathsep) if value] if raw else cfg["source"]["state_globs"]
paths = {path for pattern in patterns for path in glob.glob(os.path.expanduser(pattern), recursive=True) if os.path.isfile(path)}
print(len(paths))
PY
)
  if (( INPUT_COUNT == 0 )); then
    echo "No causal-query state files found. Set QUERY_CREDIT_INPUTS before reserving the H100 node." >&2
    exit 2
  fi
  echo "[qwen35] Found $INPUT_COUNT causal-query state files."
fi

services_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( services_started == 1 )); then
    bash "$ROOT/query_credit/stop_weekend_services.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ ${SKIP_COLLECTION:-0} != 1 ]]; then
  bash "$ROOT/query_credit/launch_weekend_services.sh"
  services_started=1
  if [[ -f "$RUNTIME_ROOT/model_path" ]]; then
    export BASE_MODEL=$(cat "$RUNTIME_ROOT/model_path")
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo "[qwen35] Collecting non-thinking fixed-cardinality counterfactuals (soft limit ${COLLECTION_HOURS}h)."
  set +e
  timeout --signal=TERM --kill-after=10m "${COLLECTION_HOURS}h" \
    "$PYTHON" -m stackpilot.query_credit_weekend_collect_qwen35 \
      --config "$CONFIG" --causal-config "$CAUSAL_CONFIG" --profile "$PROFILE"
  collection_status=$?
  set -e
  if [[ $collection_status -ne 0 && $collection_status -ne 124 && $collection_status -ne 143 ]]; then
    echo "Collection exited with status $collection_status; cached states will still be finalized." >&2
  fi
  "$PYTHON" -m stackpilot.query_credit_weekend_collect_qwen35 \
    --config "$CONFIG" --profile "$PROFILE" --finalize-only
  assert_no_thinking_leaks
  "$PYTHON" -m stackpilot.query_credit_weekend_report \
    --config "$CONFIG" --profile "$PROFILE"
  bash "$ROOT/query_credit/stop_weekend_services.sh" || true
  services_started=0
else
  [[ -f "$RUNTIME_ROOT/model_path" ]] && export BASE_MODEL=$(cat "$RUNTIME_ROOT/model_path")
fi

assert_no_thinking_leaks

AUDIT_DECISION=$WORK_DIR/$PROFILE/reports/audit/decision.json
AUDIT_GO=$("$PYTHON" - "$AUDIT_DECISION" <<'PY'
import json,sys
try:
    print(1 if json.load(open(sys.argv[1]))['go_to_optimization_audit'] else 0)
except Exception:
    print(0)
PY
)
if [[ "$AUDIT_GO" != 1 && ${FORCE_CONTINUE:-0} != 1 ]]; then
  echo "[qwen35] Audit gate did not pass. Expensive gradient/training stages are skipped."
  "$PYTHON" -m stackpilot.query_credit_weekend_summary --config "$CONFIG" --profile "$PROFILE"
  exit 0
fi

if [[ -z ${BASE_MODEL:-} ]]; then
  MODEL_SOURCE=${CAUSAL_QUERY_BASE_MODEL:-Qwen/Qwen3.5-9B}
  MODEL_REVISION=${MODEL_REVISION:-28a1d5547fecc4172665ca0ee26ea6c6dc8d3127}
  export BASE_MODEL=$(unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; \
    bash "$ROOT/scripts/resolve_hf_model.sh" "$MODEL_SOURCE" "$MODEL_REVISION" "$PYTHON")
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

WORKER_GPUS=$GPU_COUNT
(( WORKER_GPUS > 8 )) && WORKER_GPUS=8
(( WORKER_GPUS < 1 )) && WORKER_GPUS=1

run_partitioned_jobs() {
  local stage=$1
  local timeout_hours=$2
  local jobs_file=$3
  local queue_root=$RUNTIME_ROOT/${stage}_queues
  rm -rf "$queue_root"
  mkdir -p "$queue_root"
  local index=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local gpu=$((index % WORKER_GPUS))
    echo "$line" >> "$queue_root/gpu_${gpu}.txt"
    index=$((index + 1))
  done < "$jobs_file"
  local pids=()
  for ((gpu=0; gpu<WORKER_GPUS; gpu++)); do
    [[ -s "$queue_root/gpu_${gpu}.txt" ]] || continue
    timeout --signal=TERM --kill-after=5m "${timeout_hours}h" \
      bash "$ROOT/query_credit/run_weekend_worker_queue.sh" \
        "$stage" "$gpu" "$queue_root/gpu_${gpu}.txt" "$PYTHON" "$CONFIG" "$PROFILE" \
      > "$LOG_ROOT/${stage}_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

if [[ ${SKIP_IG:-0} != 1 ]]; then
  IG_JOBS=$RUNTIME_ROOT/ig_jobs.tsv
  : > "$IG_JOBS"
  for ((shard=0; shard<WORKER_GPUS; shard++)); do
    printf '%s\t%s\n' "$shard" "$WORKER_GPUS" >> "$IG_JOBS"
  done
  echo "[qwen35] Running the teacher-forced information-gain baseline."
  run_partitioned_jobs ig "$IG_HOURS" "$IG_JOBS" || true
  "$PYTHON" -m stackpilot.query_credit_weekend_ig \
    --config "$CONFIG" --profile "$PROFILE" --report || true
fi

if [[ ${SKIP_GRADIENT:-0} != 1 ]]; then
  GRADIENT_JOBS=$RUNTIME_ROOT/gradient_jobs.tsv
  "$PYTHON" - "$CONFIG" > "$GRADIENT_JOBS" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))
for seed in cfg['gradient']['init_seeds']:
    print(f"{seed}\t_")
PY
  echo "[qwen35] Running state-level gradient audit on up to $WORKER_GPUS GPUs."
  run_partitioned_jobs gradient "$GRADIENT_HOURS" "$GRADIENT_JOBS" || true
  "$PYTHON" -m stackpilot.query_credit_weekend_gradient \
    --config "$CONFIG" --profile "$PROFILE" --report || true
fi

if [[ ${SKIP_MICRO:-0} != 1 ]]; then
  MICRO_JOBS=$RUNTIME_ROOT/micro_jobs.tsv
  "$PYTHON" - "$CONFIG" > "$MICRO_JOBS" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))
for seed in cfg['micro_update']['seeds']:
    for method in cfg['micro_update']['methods']:
        print(f"{seed}\t{method}")
PY
  echo "[qwen35] Running dose-matched LoRA micro-updates on up to $WORKER_GPUS GPUs."
  run_partitioned_jobs micro "$MICRO_HOURS" "$MICRO_JOBS" || true
  "$PYTHON" -m stackpilot.query_credit_weekend_micro \
    --config "$CONFIG" --profile "$PROFILE" --report || true
fi

"$PYTHON" -m stackpilot.query_credit_weekend_summary --config "$CONFIG" --profile "$PROFILE"
echo "[qwen35] Finished. Read $WORK_DIR/$PROFILE/reports/WEEKEND_DECISION_KO.md"

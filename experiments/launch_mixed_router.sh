#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"
ensure_local_no_proxy

PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
MIXED_PORT=${MIXED_PORT:-8200}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
REQUEST_TIMEOUT=${MIXED_REQUEST_TIMEOUT:-180}
READY_TIMEOUT=${MIXED_READY_TIMEOUT:-600}
RUNTIME_WORK_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work}
RUNTIME_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs}
PID_ROOT=$RUNTIME_WORK_ROOT/experiments/services
LOG_ROOT=$RUNTIME_LOG_ROOT/experiments/services
mkdir -p "$PID_ROOT" "$LOG_ROOT"
ASSIGNMENT_LOG=${ASSIGNMENT_LOG:-$RUNTIME_WORK_ROOT/experiments/services/mixed-assignments.jsonl}

bash "$ROOT/experiments/ensure_retrievers.sh"

PID_FILE=$PID_ROOT/mixed-router.pid
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  health=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 20 \
    "http://127.0.0.1:${MIXED_PORT}/health" 2>/dev/null || true)
  if "$PILOT_PYTHON" - \
    "$health" "$BM25_PORT" "$E5_PORT" "$REQUEST_TIMEOUT" \
    "$ASSIGNMENT_LOG" "$ROOT" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

payload_text, bm25_port, e5_port, timeout, assignment_log, root = sys.argv[1:]
try:
    payload = json.loads(payload_text)
except json.JSONDecodeError:
    raise SystemExit(1)
expected_urls = {
    "bm25": f"http://127.0.0.1:{bm25_port}/retrieve",
    "e5": f"http://127.0.0.1:{e5_port}/retrieve",
}
valid = (
    payload.get("status") == "ok"
    and payload.get("backend") == "mixed"
    and payload.get("upstream_urls") == expected_urls
    and int(payload.get("default_topk", -1)) == 3
    and payload.get("assignment_log") == str(Path(assignment_log).resolve())
    and math.isclose(
        float(payload.get("request_timeout_seconds", -1)), float(timeout)
    )
    and payload.get("server_file_sha256")
    == hashlib.sha256(
        (Path(root) / "stackpilot" / "mixed_retriever_server.py").read_bytes()
    ).hexdigest()
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "Reusing configuration-matched mixed router on port $MIXED_PORT"
    exit 0
  fi
  echo "Existing mixed-router configuration is stale; restarting it."
  stop_managed_pid "$PID_FILE" "$PILOT_PYTHON" "$ROOT" 1
fi
rm -f "$PID_FILE"
require_free_port "$PILOT_PYTHON" "$MIXED_PORT"
LOG_FILE=$LOG_ROOT/mixed-router.log
PID=$(CUDA_VISIBLE_DEVICES='' start_managed_process \
  "$PILOT_PYTHON" "$LOG_FILE" "$PILOT_PYTHON" -m stackpilot.mixed_retriever_server \
  --bm25-url "http://127.0.0.1:${BM25_PORT}/retrieve" \
  --e5-url "http://127.0.0.1:${E5_PORT}/retrieve" \
  --topk 3 --timeout "$REQUEST_TIMEOUT" --port "$MIXED_PORT" \
  --assignment-log "$ASSIGNMENT_LOG")
echo "$PID" > "$PID_FILE"
wait_for_http "$PID" "http://127.0.0.1:${MIXED_PORT}/health" \
  "$READY_TIMEOUT" "$LOG_FILE" '"backend":"mixed"'
echo "Mixed router ready: http://127.0.0.1:${MIXED_PORT}/retrieve"

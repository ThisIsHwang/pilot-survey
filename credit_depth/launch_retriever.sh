#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
CONFIG=${CREDIT_DEPTH_CONFIG:-$ROOT/configs/credit_depth.yaml}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3
readarray -t VALUES < <("$PYTHON" - "$CONFIG" "$BACKEND" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
backend=sys.argv[2]
print(cfg['retrieval'][f'{backend}_port'])
print(cfg['retrieval']['e5_model'])
PY
)
PORT=${CREDIT_DEPTH_RETRIEVER_PORT:-${VALUES[0]}}
E5_MODEL=${E5_MODEL:-${VALUES[1]}}
CORPUS=$ROOT/work/credit_depth/benchmark/$PROFILE/corpus.jsonl
[[ -s "$CORPUS" ]] || PROFILE="$PROFILE" bash credit_depth/run_build.sh
PID_DIR=$ROOT/work/credit_depth/pids
LOG_DIR=$ROOT/logs/credit_depth
mkdir -p "$PID_DIR" "$LOG_DIR"
PID_FILE=$PID_DIR/${PROFILE}-${BACKEND}.pid
if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "CreditDepth $BACKEND retriever already running on port $PORT"
  exit 0
fi
ARGS=(--corpus "$CORPUS" --backend "$BACKEND" --port "$PORT")
if [[ "$BACKEND" == e5 ]]; then
  ARGS+=(--e5-model "$E5_MODEL" --device cuda:0)
  CUDA_VISIBLE_DEVICES=${CREDIT_DEPTH_E5_GPU:-7} nohup "$PYTHON" -m stackpilot.credit_depth_retriever "${ARGS[@]}" >"$LOG_DIR/${PROFILE}-${BACKEND}.log" 2>&1 &
else
  nohup "$PYTHON" -m stackpilot.credit_depth_retriever "${ARGS[@]}" >"$LOG_DIR/${PROFILE}-${BACKEND}.log" 2>&1 &
fi
PID=$!
echo "$PID" > "$PID_FILE"
"$PYTHON" - "$PORT" "$PID" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1]); pid=int(sys.argv[2]); deadline=time.time()+180
while time.time()<deadline:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
            payload=json.load(r)
        if payload.get('status')=='ok':
            print(json.dumps(payload,indent=2)); raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit(f'retriever pid={pid} did not become healthy')
PY

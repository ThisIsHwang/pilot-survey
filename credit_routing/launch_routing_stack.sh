#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
OBSERVATION_ROUTE=${OBSERVATION_ROUTE:?Set OBSERVATION_ROUTE=rank or utility}
PYTHON=${PILOT_PYTHON:-$ROOT/.venv-pilot/bin/python}
[[ -x "$PYTHON" ]] || bash scripts/bootstrap.sh
ARTIFACT=${UTILITY_ARTIFACT:-$ROOT/work/credit_routing/models/$PROFILE/document_utility_ridge.json}
[[ -s "$ARTIFACT" ]] || { echo "Missing utility artifact: $ARTIFACT" >&2; exit 1; }
UPSTREAM_BM25_PORT=${UPSTREAM_BM25_PORT:-8201}
UPSTREAM_E5_PORT=${UPSTREAM_E5_PORT:-8202}
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
E5_GPU=${E5_GPU:-7}
RUNTIME=$ROOT/work/credit_routing/runtime/$PROFILE/$OBSERVATION_ROUTE
mkdir -p "$RUNTIME/pids" "$RUNTIME/logs"

# Stop only processes managed by this project before changing the port layout.
bash hard_rq0/stop_retrievers.sh >/dev/null 2>&1 || true
for name in bm25 e5; do
  pid_file=$RUNTIME/pids/${name}-proxy.pid
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file" || true)
    [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    rm -f "$pid_file"
  fi
done

BM25_PORT=$UPSTREAM_BM25_PORT E5_PORT=$UPSTREAM_E5_PORT E5_GPU=$E5_GPU \
  bash hard_rq0/launch_retrievers.sh

eval "$("$PYTHON" - configs/credit_routing.yaml <<'PY'
import shlex,sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
values={
 'UPSTREAM_TOPK':cfg['labeling']['upstream_topk'],
 'OUTPUT_TOPK':cfg['labeling']['observation_k'],
 'ACTION_AGGREGATION':cfg['routing']['action_aggregation'],
 'ACTION_AGGREGATION_K':cfg['routing']['action_aggregation_k'],
}
for k,v in values.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"

launch_proxy(){
  local backend=$1 upstream_port=$2 proxy_port=$3
  local log=$RUNTIME/logs/${backend}-proxy.log pid_file=$RUNTIME/pids/${backend}-proxy.pid
  nohup "$PYTHON" -m stackpilot.credit_routing_proxy \
    --backend "$backend" \
    --upstream-url "http://127.0.0.1:${upstream_port}/retrieve" \
    --port "$proxy_port" --artifact "$ARTIFACT" \
    --observation-route "$OBSERVATION_ROUTE" \
    --upstream-topk "$UPSTREAM_TOPK" --output-topk "$OUTPUT_TOPK" \
    --action-aggregation "$ACTION_AGGREGATION" \
    --action-aggregation-k "$ACTION_AGGREGATION_K" \
    --log "$RUNTIME/${backend}-selections.jsonl" \
    >"$log" 2>&1 &
  echo $! > "$pid_file"
}
launch_proxy bm25 "$UPSTREAM_BM25_PORT" "$BM25_PORT"
launch_proxy e5 "$UPSTREAM_E5_PORT" "$E5_PORT"

for port in "$BM25_PORT" "$E5_PORT"; do
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  [[ $ready == 1 ]] || { echo "Credit-routing proxy on port $port did not become healthy" >&2; exit 1; }
done
printf '%s\n' "BM25_PORT=$BM25_PORT" "E5_PORT=$E5_PORT" "UPSTREAM_BM25_PORT=$UPSTREAM_BM25_PORT" "UPSTREAM_E5_PORT=$UPSTREAM_E5_PORT" > "$RUNTIME/ports.env"
echo "Credit-routing stack ready: observation_route=$OBSERVATION_ROUTE"

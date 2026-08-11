#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=${PROFILE:-pilot}
OBSERVATION_ROUTE=${OBSERVATION_ROUTE:-rank}
RUNTIME=$ROOT/work/credit_routing/runtime/$PROFILE/$OBSERVATION_ROUTE
for name in bm25 e5; do
  pid_file=$RUNTIME/pids/${name}-proxy.pid
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then kill "$pid" >/dev/null 2>&1 || true; fi
    rm -f "$pid_file"
  fi
done
bash "$ROOT/hard_rq0/stop_retrievers.sh" >/dev/null 2>&1 || true

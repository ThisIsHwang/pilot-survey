#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
PID_FILE=$ROOT/work/credit_depth/pids/${PROFILE}-${BACKEND}.pid
if [[ -s "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  kill "$PID" 2>/dev/null || true
  for _ in {1..30}; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
fi

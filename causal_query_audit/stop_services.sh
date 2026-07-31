#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/causal_query_audit/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/causal_query_audit}
status=0
bash "$ROOT/hard_rq0/stop_retrievers.sh" || status=1
bash "$ROOT/scripts/stop_servers.sh" || status=1
exit "$status"

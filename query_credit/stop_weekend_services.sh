#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/query_credit_weekend/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/query_credit_weekend}
profile=single
[[ -f "$STACKPILOT_RUNTIME_ROOT/hardware_profile" ]] && profile=$(cat "$STACKPILOT_RUNTIME_ROOT/hardware_profile")
status=0
if [[ "$profile" == node8 ]]; then
  bash "$ROOT/causal_query_audit/stop_services.sh" || status=1
else
  source "$ROOT/scripts/lib/runtime.sh"
  stop_managed_pid "$STACKPILOT_RUNTIME_ROOT/hard_rq0/pids/bm25.pid" \
    "stackpilot.searchr1_server" "$ROOT" 1 || status=1
  bash "$ROOT/scripts/stop_servers.sh" || status=1
fi
exit "$status"

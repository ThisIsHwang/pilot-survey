#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export STACKPILOT_RUNTIME_ROOT=${STACKPILOT_RUNTIME_ROOT:-$ROOT/work/behavior_alias_pilot/runtime}
export STACKPILOT_LOG_ROOT=${STACKPILOT_LOG_ROOT:-$ROOT/logs/behavior_alias_pilot}
export CAUSAL_QUERY_BASE_MODEL=${BEHAVIOR_ALIAS_BASE_MODEL:-${CAUSAL_QUERY_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}
export CAUSAL_QUERY_MODEL_REVISION=${BEHAVIOR_ALIAS_MODEL_REVISION:-${CAUSAL_QUERY_MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}}
# Reuse the strict 7B/vLLM/retriever launch contract from the causal audit.
bash "$ROOT/causal_query_audit/launch_services.sh"

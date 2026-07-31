#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
KEEP_SERVICES=${KEEP_SERVICES:-0}

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
  bash "$ROOT/scripts/bootstrap_vllm.sh"
fi
if [[ ${SKIP_ASSETS:-0} != 1 ]]; then
  bash "$ROOT/hard_rq0/download_assets.sh"
fi
if [[ ${SKIP_PREPARE:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash "$ROOT/causal_query_audit/prepare.sh"
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_SERVICES" != 1 && ${SKIP_SERVICES:-0} != 1 ]]; then
    bash "$ROOT/causal_query_audit/stop_services.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ${SKIP_SERVICES:-0} != 1 ]]; then
  bash "$ROOT/causal_query_audit/launch_services.sh"
fi
PROFILE="$PROFILE" bash "$ROOT/causal_query_audit/run.sh"
PROFILE="$PROFILE" bash "$ROOT/causal_query_audit/report.sh"

echo "EXP-013 complete: $ROOT/work/causal_query_audit/reports/$PROFILE/CAUSAL_QUERY_AUDIT_REPORT.md"

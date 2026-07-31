#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
KEEP_SERVICES=${KEEP_SERVICES:-0}
SERVICES_STARTED=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$SERVICES_STARTED" == 1 && "$KEEP_SERVICES" != 1 ]]; then
    bash "$ROOT/behavior_alias_pilot/stop_services.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash scripts/bootstrap.sh
  bash scripts/bootstrap_vllm.sh
fi
if [[ ${SKIP_ASSETS:-0} != 1 ]]; then
  bash hard_rq0/download_assets.sh
fi
if [[ ${SKIP_PREPARE:-0} != 1 ]]; then
  PROFILE="$PROFILE" bash behavior_alias_pilot/prepare.sh
fi
if [[ ${SKIP_SERVICES:-0} != 1 ]]; then
  bash behavior_alias_pilot/launch_services.sh
  SERVICES_STARTED=1
fi
PROFILE="$PROFILE" bash behavior_alias_pilot/run.sh
PROFILE="$PROFILE" bash behavior_alias_pilot/report.sh

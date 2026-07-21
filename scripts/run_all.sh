#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/runtime.sh"

cleanup_on_failure() {
  local status=$?
  local log_file
  trap - ERR INT TERM
  echo "Pilot failed; stopping managed servers. Logs are under $ROOT/logs/." >&2
  for log_file in "$ROOT"/logs/{bm25,e5,colbert,vllm}.log; do
    show_log_tail "$log_file"
  done
  bash "$ROOT/scripts/stop_servers.sh" || true
  exit "$status"
}
trap cleanup_on_failure ERR INT TERM

# Never rebuild an environment while an older process is importing from it.
bash "$ROOT/scripts/stop_servers.sh"

if [[ ${SKIP_BOOTSTRAP:-0} != 1 ]]; then
  bash "$ROOT/scripts/bootstrap.sh"
  bash "$ROOT/scripts/bootstrap_vllm.sh"
fi

bash "$ROOT/scripts/preflight.sh"
bash "$ROOT/scripts/prepare_data.sh" --config configs/pilot.yaml
bash "$ROOT/scripts/build_indexes.sh"
bash "$ROOT/scripts/launch_retrievers.sh"
bash "$ROOT/scripts/launch_vllm_bg.sh"
bash "$ROOT/scripts/run_retrieval_matrix.sh" --limit "${RETRIEVAL_LIMIT:-500}"
bash "$ROOT/scripts/run_agent_eval.sh" --limit "${AGENT_LIMIT:-200}"

trap - ERR INT TERM
if [[ ${KEEP_SERVERS:-0} != 1 ]]; then
  bash "$ROOT/scripts/stop_servers.sh"
fi

echo "Pilot completed. Report: $ROOT/work/results/REPORT.md"

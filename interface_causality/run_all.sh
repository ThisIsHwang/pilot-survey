#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash interface_causality/run_offline.sh
PROFILE="$PROFILE" bash interface_causality/run_expressivity.sh
if [[ ${RUN_DOCUMENT_CTU:-0} == 1 ]]; then
  PROFILE="$PROFILE" bash interface_causality/run_document_ctu.sh
  PROFILE="$PROFILE" bash interface_causality/run_granularity_with_documents.sh
fi
PROFILE="$PROFILE" bash interface_causality/report.sh

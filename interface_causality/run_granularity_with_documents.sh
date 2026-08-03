#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
DOCUMENT_CTU_FILE=${DOCUMENT_CTU_FILE:-$ROOT/work/interface_causality/document_ctu/$PROFILE/document_ctu.jsonl}
export DOCUMENT_CTU_FILE
PROFILE="$PROFILE" bash interface_causality/run_granularity.sh

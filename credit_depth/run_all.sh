#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
BACKEND=${BACKEND:-bm25}
PROFILE="$PROFILE" bash credit_depth/run_build.sh
PROFILE="$PROFILE" BACKEND="$BACKEND" bash credit_depth/run_controlled_census.sh
PROFILE="$PROFILE" BACKEND="$BACKEND" bash credit_depth/run_labels.sh
PROFILE="$PROFILE" BACKEND="$BACKEND" bash credit_depth/run_gate.sh
if [[ ${RUN_META_AUDIT:-0} == 1 ]]; then PROFILE="$PROFILE" bash credit_depth/run_meta_audit.sh; fi
if [[ ${SKIP_TRAINING:-1} != 1 ]]; then PROFILE="$PROFILE" bash credit_depth/run_factorial.sh; fi
if [[ ${RUN_GATED:-0} == 1 ]]; then PROFILE="$PROFILE" bash credit_depth/run_gated.sh; fi
if [[ ${RUN_ENDPOINT_REPORT:-0} == 1 ]]; then PROFILE="$PROFILE" bash credit_depth/run_endpoint_report.sh; fi

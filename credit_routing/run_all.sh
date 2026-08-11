#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
if [[ ${SKIP_LABELS:-0} != 1 ]]; then PROFILE="$PROFILE" bash credit_routing/prepare_labels.sh; fi
if [[ ${SKIP_ESTIMATOR:-0} != 1 ]]; then PROFILE="$PROFILE" bash credit_routing/train_estimator.sh; fi
if [[ ${SKIP_GRPO:-0} != 1 ]]; then PROFILE="$PROFILE" bash credit_routing/run_factorial.sh; fi
if [[ ${SKIP_ENDPOINT:-0} != 1 ]]; then PROFILE="$PROFILE" bash credit_routing/merge_eval.sh; fi
PROFILE="$PROFILE" bash credit_routing/report.sh

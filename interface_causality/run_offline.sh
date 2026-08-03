#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash interface_causality/run_alias.sh
PROFILE="$PROFILE" bash interface_causality/run_granularity.sh
PROFILE="$PROFILE" bash interface_causality/run_predictor.sh

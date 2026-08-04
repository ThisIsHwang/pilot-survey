#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash behavior_quotient/merge_eval.sh
PROFILE="$PROFILE" bash behavior_quotient/report.sh

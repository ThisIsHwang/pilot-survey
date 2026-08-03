#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
PROFILE="$PROFILE" bash query_attribution/run_all.sh
PROFILE="$PROFILE" bash query_attribution/run_interactive.sh

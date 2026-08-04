#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
exec bash multipositive_generalization/run_jobs.sh --experiment EXP-026 "$@"

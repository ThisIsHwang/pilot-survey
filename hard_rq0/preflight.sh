#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}

bash "$ROOT/scripts/preflight.sh"
STAGE2_MIN_DISK_GIB=60 bash "$ROOT/scripts/preflight_searchr1.sh"
PROFILE=$PROFILE bash "$ROOT/hard_rq0/preflight_storage.sh"
echo "Hard-RQ0 preflight passed."

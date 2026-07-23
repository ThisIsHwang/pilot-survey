#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Shell assignments entered on separate lines are not inherited by a child
# `bash` unless exported. Keep the supported "skip only Stage 0" resume path in
# one executable so Stage 2 and Hard-RQ0 cannot accidentally receive defaults.
export RUN_STAGE0=0
export RUN_HARD_RQ0=${RUN_HARD_RQ0:-1}
export SKIP_BOOTSTRAP=${SKIP_BOOTSTRAP:-1}
export KEEP_HARD_SOURCE_ARCHIVES=${KEEP_HARD_SOURCE_ARCHIVES:-1}

exec bash "$ROOT/scripts/run_full_pipeline.sh"

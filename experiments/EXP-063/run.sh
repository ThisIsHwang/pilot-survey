#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${PROFILE:-pilot}; BACKEND=${BACKEND:-bm25}
PROFILE="$PROFILE" bash "$ROOT/credit_depth/run_build.sh"
PROFILE="$PROFILE" BACKEND="$BACKEND" bash "$ROOT/credit_depth/run_controlled_census.sh"

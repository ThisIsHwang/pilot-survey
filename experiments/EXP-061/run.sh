#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${PROFILE:-pilot} bash "$ROOT/credit_depth/run_meta_audit.sh"

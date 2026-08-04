#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot} bash behavior_feedback/run_factorial.sh
PROFILE=${PROFILE:-pilot} bash behavior_feedback/merge_eval.sh

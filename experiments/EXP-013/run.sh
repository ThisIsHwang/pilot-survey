#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
exec bash "$ROOT/causal_query_audit/run_all.sh" "$@"

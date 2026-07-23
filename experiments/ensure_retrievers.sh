#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/hard_rq0/ensure_retrievers.sh"

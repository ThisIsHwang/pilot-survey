#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for file in "$ROOT"/work/pids/*.pid "$ROOT"/work_smoke/pids/*.pid; do
  [[ -f "$file" ]] || continue
  pid=$(cat "$file")
  kill "$pid" 2>/dev/null || true
  rm -f "$file"
done

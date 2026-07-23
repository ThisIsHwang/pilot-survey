#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BM25_PORT=${BM25_PORT:-8101}
E5_PORT=${E5_PORT:-8102}
PILOT_PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PILOT_PYTHON" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }

healthy() {
  local port=$1
  local backend=$2
  local payload
  payload=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 10 \
    "http://127.0.0.1:${port}/health" 2>/dev/null) || return 1
  "$PILOT_PYTHON" - "$payload" "$backend" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
expected = sys.argv[2]
if payload.get("status") != "ok" or payload.get("backend") != expected:
    raise SystemExit(1)
PY
}

if healthy "$BM25_PORT" bm25 && healthy "$E5_PORT" e5; then
  echo "Reusing healthy hard-RQ0 BM25/E5 retrievers."
  exit 0
fi

echo "Hard-RQ0 retrievers are absent or unhealthy; starting them now."
bash "$ROOT/hard_rq0/launch_retrievers.sh"

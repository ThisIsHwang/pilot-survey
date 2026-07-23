#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
RUNTIME_PATCH=$ROOT/searchr1_stage2/searchr1-runtime.patch

[[ -e "$SEARCH_R1/.git" ]] || {
  echo "Missing pinned Search-R1 checkout: $SEARCH_R1" >&2
  exit 1
}
if [[ "$(git -C "$SEARCH_R1" rev-parse HEAD)" != "$SEARCH_R1_COMMIT" ]]; then
  echo "Search-R1 is not at the required commit $SEARCH_R1_COMMIT." >&2
  exit 1
fi
[[ -s "$RUNTIME_PATCH" ]] || {
  echo "Missing Search-R1 runtime patch: $RUNTIME_PATCH" >&2
  exit 1
}

if git -C "$SEARCH_R1" apply --unidiff-zero --reverse --check \
  "$RUNTIME_PATCH" >/dev/null 2>&1; then
  echo "Search-R1 retrieval-timeout patch is already applied."
elif git -C "$SEARCH_R1" apply --unidiff-zero --check \
  "$RUNTIME_PATCH" >/dev/null 2>&1; then
  git -C "$SEARCH_R1" apply --unidiff-zero "$RUNTIME_PATCH"
  echo "Applied Search-R1 retrieval-timeout patch."
else
  echo "Unable to apply $RUNTIME_PATCH cleanly to pinned Search-R1." >&2
  echo "Preserving the upstream checkout; inspect its local changes." >&2
  exit 1
fi

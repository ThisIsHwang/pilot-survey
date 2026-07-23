#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}
[[ -e "$SEARCH_R1/.git" ]] || { echo "Missing Search-R1 checkout: $SEARCH_R1" >&2; exit 1; }
if [[ "$(git -C "$SEARCH_R1" rev-parse HEAD)" != "$SEARCH_R1_COMMIT" ]]; then
  echo "Search-R1 HEAD is not the pinned commit $SEARCH_R1_COMMIT" >&2
  exit 1
fi

known=(
  search_r1/llm_agent/generation.py
  verl/trainer/main_ppo.py
  verl/workers/sharding_manager/fsdp_vllm.py
)
declare -A allowed=()
for path in "${known[@]}"; do allowed[$path]=1; done
mapfile -t modified < <(
  {
    git -C "$SEARCH_R1" diff --name-only
    git -C "$SEARCH_R1" diff --cached --name-only
  } | sort -u
)
unknown=()
for path in "${modified[@]}"; do
  [[ -z "$path" ]] && continue
  if [[ -z ${allowed[$path]+x} ]]; then unknown+=("$path"); fi
done
if (( ${#unknown[@]} > 0 )); then
  printf 'Refusing to reset Search-R1 with unknown tracked edits:\n' >&2
  printf '  %s\n' "${unknown[@]}" >&2
  exit 1
fi

git -C "$SEARCH_R1" restore --source "$SEARCH_R1_COMMIT" --staged --worktree -- "${known[@]}"
for path in "${known[@]}"; do
  if ! git -C "$SEARCH_R1" diff --quiet "$SEARCH_R1_COMMIT" -- "$path"; then
    echo "Failed to reset $path to $SEARCH_R1_COMMIT" >&2
    exit 1
  fi
done
echo "Reset known Search-R1 experiment files to $SEARCH_R1_COMMIT"

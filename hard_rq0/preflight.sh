#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
PYTHON=$ROOT/.venv-pilot/bin/python

case "$PROFILE" in
  smoke) PROFILE_FREE_GIB=60 ;;
  pilot) PROFILE_FREE_GIB=120 ;;
  full) PROFILE_FREE_GIB=250 ;;
  *) echo "PROFILE must be smoke, pilot, or full; got '$PROFILE'." >&2; exit 2 ;;
esac

bash "$ROOT/scripts/preflight.sh"
STAGE2_MIN_DISK_GIB=60 bash "$ROOT/scripts/preflight_searchr1.sh"

full_asset_free_gib=${HARD_ASSET_MIN_FREE_GIB:-150}
if [[ ! "$full_asset_free_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "HARD_ASSET_MIN_FREE_GIB must be a positive integer; got '$full_asset_free_gib'." >&2
  exit 2
fi

asset_required_gib=0
if "$PYTHON" -m stackpilot.hard_assets check \
  --root "$ASSET_ROOT" >/dev/null 2>&1; then
  asset_status="verified cache"
else
  asset_required_gib=$("$PYTHON" -m stackpilot.hard_assets required-free \
    --root "$ASSET_ROOT" --full-min-gib "$full_asset_free_gib")
  if [[ ! "$asset_required_gib" =~ ^[0-9]+$ ]]; then
    echo "Unable to determine remaining hard-RQ0 asset storage; got '$asset_required_gib'." >&2
    exit 1
  fi
  if [[ $asset_required_gib -eq 0 ]]; then
    asset_status="verified components; BM25 link repair pending"
  else
    asset_status="missing components (${asset_required_gib} GiB download headroom)"
  fi
  if [[ -e "$ASSET_ROOT/.hard-rq0-assets-manifest.json" ]]; then
    echo "The hard-RQ0 cache is incomplete; download_assets.sh will reuse verified components and fetch only missing or invalid ones." >&2
  fi
fi

if [[ -n ${HARD_MIN_DISK_GIB:-} ]]; then
  min_disk_gib=$HARD_MIN_DISK_GIB
else
  min_disk_gib=$((PROFILE_FREE_GIB + asset_required_gib))
fi
min_ram_gib=${HARD_MIN_RAM_GIB:-128}
for pair in "HARD_MIN_DISK_GIB:$min_disk_gib" "HARD_MIN_RAM_GIB:$min_ram_gib"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  fi
done

available_disk_kib=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
if (( available_disk_kib < min_disk_gib * 1024 * 1024 )); then
  echo "Hard-RQ0 ($PROFILE) requires at least ${min_disk_gib} GiB free under $ROOT; found $((available_disk_kib / 1024 / 1024)) GiB." >&2
  if [[ $asset_required_gib -gt 0 ]]; then
    echo "This includes ${asset_required_gib} GiB headroom for only the missing hard-RQ0 asset components." >&2
  fi
  echo "Override HARD_MIN_DISK_GIB only after checking asset and checkpoint storage." >&2
  exit 1
fi

available_ram_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [[ ! "$available_ram_kib" =~ ^[0-9]+$ ]]; then
  echo "Unable to determine MemAvailable from /proc/meminfo." >&2
  exit 1
fi
if (( available_ram_kib < min_ram_gib * 1024 * 1024 )); then
  echo "Hard-RQ0 full-wiki retrieval requires at least ${min_ram_gib} GiB available RAM; found $((available_ram_kib / 1024 / 1024)) GiB." >&2
  exit 1
fi

echo "Hard-RQ0 preflight passed: profile=$PROFILE; assets=$asset_status; free disk=$((available_disk_kib / 1024 / 1024)) GiB; available RAM=$((available_ram_kib / 1024 / 1024)) GiB."

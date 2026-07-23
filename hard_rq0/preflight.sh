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

assets_ready=0
if "$PYTHON" -m stackpilot.hard_assets check \
  --root "$ASSET_ROOT" >/dev/null 2>&1; then
  assets_ready=1
elif [[ -e "$ASSET_ROOT/.hard-rq0-assets-manifest.json" ]]; then
  echo "The existing hard-RQ0 cache lacks current pinned provenance; download_assets.sh will rebuild it safely." >&2
fi

if [[ -n ${HARD_MIN_DISK_GIB:-} ]]; then
  min_disk_gib=$HARD_MIN_DISK_GIB
elif [[ $assets_ready -eq 1 ]]; then
  min_disk_gib=$PROFILE_FREE_GIB
else
  min_disk_gib=$((PROFILE_FREE_GIB + 150))
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
  if [[ $assets_ready -eq 0 ]]; then
    echo "This includes headroom to assemble the 64.6 GB E5 index and retain checkpoints." >&2
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

if [[ $assets_ready -eq 1 ]]; then
  asset_status="verified cache"
else
  asset_status="download required"
fi
echo "Hard-RQ0 preflight passed: profile=$PROFILE; assets=$asset_status; free disk=$((available_disk_kib / 1024 / 1024)) GiB; available RAM=$((available_ram_kib / 1024 / 1024)) GiB."

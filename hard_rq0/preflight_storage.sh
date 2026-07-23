#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${PROFILE:-pilot}
ASSET_ROOT=${HARD_ASSET_ROOT:-$ROOT/work/hard_rq0/assets/wiki18}
HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
PYTHON=$ROOT/.venv-pilot/bin/python
[[ -x "$PYTHON" ]] || {
  echo "Missing .venv-pilot; run scripts/bootstrap.sh first." >&2
  exit 1
}

case "$PROFILE" in
  smoke) PROFILE_FREE_GIB=60 ;;
  pilot) PROFILE_FREE_GIB=120 ;;
  full) PROFILE_FREE_GIB=250 ;;
  *) echo "PROFILE must be smoke, pilot, or full; got '$PROFILE'." >&2; exit 2 ;;
esac

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

root_extra_gib=${HARD_ROOT_EXTRA_GIB:-0}
hf_reserve_gib=${HARD_HF_RESERVE_GIB:-0}
min_ram_gib=${HARD_MIN_RAM_GIB:-192}
min_cpu_cores=${HARD_MIN_CPU_CORES:-22}
for pair in \
  "HARD_ROOT_EXTRA_GIB:$root_extra_gib" \
  "HARD_HF_RESERVE_GIB:$hf_reserve_gib"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer; got '$value'." >&2
    exit 2
  fi
done
for pair in \
  "HARD_MIN_RAM_GIB:$min_ram_gib" \
  "HARD_MIN_CPU_CORES:$min_cpu_cores"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  fi
done
if [[ -n ${HARD_MIN_DISK_GIB:-} && ! "$HARD_MIN_DISK_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "HARD_MIN_DISK_GIB must be a positive integer; got '$HARD_MIN_DISK_GIB'." >&2
  exit 2
fi

mkdir -p "$ASSET_ROOT"
if (( hf_reserve_gib > 0 )); then mkdir -p "$HF_HOME"; fi
root_device=$(stat -c '%d' "$ROOT")
asset_device=$(stat -c '%d' "$ASSET_ROOT")

if [[ -n ${HARD_MIN_DISK_GIB:-} ]]; then
  root_required_gib=$HARD_MIN_DISK_GIB
else
  root_required_gib=$PROFILE_FREE_GIB
fi
root_required_gib=$((root_required_gib + root_extra_gib))

declare -A required_by_device=()
declare -A path_by_device=()
declare -A labels_by_device=()
add_reservation() {
  local path=$1
  local amount=$2
  local label=$3
  local device
  (( amount > 0 )) || return
  device=$(stat -c '%d' "$path")
  required_by_device[$device]=$((${required_by_device[$device]:-0} + amount))
  path_by_device[$device]=$path
  labels_by_device[$device]="${labels_by_device[$device]:+${labels_by_device[$device]}, }$label"
}

add_reservation "$ROOT" "$root_required_gib" "work/checkpoints"
if (( asset_required_gib > 0 )); then
  # HARD_MIN_DISK_GIB historically overrides the combined requirement when the
  # assets and work tree share a filesystem. A separate asset mount must still
  # reserve its own missing-component headroom.
  if [[ -z ${HARD_MIN_DISK_GIB:-} || "$asset_device" != "$root_device" ]]; then
    add_reservation "$ASSET_ROOT" "$asset_required_gib" "hard assets"
  fi
fi
if (( hf_reserve_gib > 0 )); then
  add_reservation "$HF_HOME" "$hf_reserve_gib" "Hugging Face models"
fi

disk_summary=()
for device in "${!required_by_device[@]}"; do
  path=${path_by_device[$device]}
  required_gib=${required_by_device[$device]}
  available_disk_kib=$(df -Pk "$path" | awk 'NR==2 {print $4}')
  if [[ ! "$available_disk_kib" =~ ^[0-9]+$ ]]; then
    echo "Unable to determine free disk space for $path." >&2
    exit 1
  fi
  available_gib=$((available_disk_kib / 1024 / 1024))
  if (( available_disk_kib < required_gib * 1024 * 1024 )); then
    echo "Hard-RQ0 ($PROFILE) requires at least ${required_gib} GiB free on the filesystem for ${labels_by_device[$device]} ($path); found ${available_gib} GiB." >&2
    echo "Override capacity checks only after accounting for concurrent downloads and checkpoints." >&2
    exit 1
  fi
  disk_summary+=("${labels_by_device[$device]}=${available_gib}GiB/${required_gib}GiB-required")
done

available_ram_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [[ ! "$available_ram_kib" =~ ^[0-9]+$ ]]; then
  echo "Unable to determine MemAvailable from /proc/meminfo." >&2
  exit 1
fi
if (( available_ram_kib < min_ram_gib * 1024 * 1024 )); then
  echo "Hard-RQ0 full-wiki retrieval requires at least ${min_ram_gib} GiB available RAM; found $((available_ram_kib / 1024 / 1024)) GiB." >&2
  exit 1
fi

available_cpu_cores=$("$PYTHON" - <<'PY'
import os
from pathlib import Path

cores = set()
for cpu in os.sched_getaffinity(0):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = (topology / "physical_package_id").read_text().strip()
        core = (topology / "core_id").read_text().strip()
    except OSError:
        cores.add(("logical", str(cpu)))
    else:
        cores.add((package, core))
print(len(cores))
PY
)
if [[ ! "$available_cpu_cores" =~ ^[0-9]+$ ]]; then
  echo "Unable to determine CPU cores visible through process affinity." >&2
  exit 1
fi
if (( available_cpu_cores < min_cpu_cores )); then
  echo "Hard-RQ0 DP serving requires at least $min_cpu_cores affinity-visible physical CPU cores; found $available_cpu_cores." >&2
  echo "Increase the scheduler CPU allocation or deliberately lower DP/API concurrency." >&2
  exit 1
fi

echo "Hard-RQ0 capacity preflight passed: profile=$PROFILE; assets=$asset_status; disks=${disk_summary[*]}; available RAM=$((available_ram_kib / 1024 / 1024)) GiB; physical CPU cores=$available_cpu_cores."

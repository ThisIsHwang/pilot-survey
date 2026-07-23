#!/usr/bin/env bash

# Locate uv or install a project-local standalone binary. This deliberately
# avoids `python -m venv` and ensurepip, which are often unavailable in minimal
# Python/cluster environments.
uv_has_exact_version() {
  local candidate=$1
  local expected=$2
  local output

  [[ -x "$candidate" ]] || return 1
  output=$("$candidate" --version 2>/dev/null) || return 1
  [[ "$output" =~ ^uv[[:space:]]+([^[:space:]]+) ]] || return 1
  [[ "${BASH_REMATCH[1]}" == "$expected" ]]
}

ensure_uv() {
  local project_root=$1
  local install_dir=${UV_BOOTSTRAP_DIR:-$project_root/.bootstrap-tools}
  local bootstrap_python=${PYTHON_BIN:-python3.12}
  local uv_version=0.11.30
  local path_uv

  UV_CACHE_DIR=${UV_CACHE_DIR:-$project_root/.cache/uv}
  mkdir -p "$UV_CACHE_DIR"
  export UV_CACHE_DIR

  if uv_has_exact_version "$install_dir/uv" "$uv_version"; then
    UV_BIN=$install_dir/uv
  elif path_uv=$(command -v uv 2>/dev/null) && \
       uv_has_exact_version "$path_uv" "$uv_version"; then
    UV_BIN=$path_uv
  else
    if ! command -v "$bootstrap_python" >/dev/null 2>&1; then
      echo "uv $uv_version and the bootstrap Python are unavailable: $bootstrap_python" >&2
      return 1
    fi

    mkdir -p "$install_dir"
    echo "Installing uv $uv_version from PyPI into $install_dir"
    "$bootstrap_python" - "$uv_version" "$install_dir" <<'PY'
import json
import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

version, install_dir = sys.argv[1:]
machine = platform.machine().lower()
if machine == "amd64":
    machine = "x86_64"
elif machine == "arm64":
    machine = "aarch64"
if machine not in {"x86_64", "aarch64"}:
    raise SystemExit(f"Unsupported uv bootstrap architecture: {machine}")

with urllib.request.urlopen(f"https://pypi.org/pypi/uv/{version}/json", timeout=30) as response:
    metadata = json.load(response)
candidates = [
    item for item in metadata["urls"]
    if item["packagetype"] == "bdist_wheel"
    and "manylinux" in item["filename"]
    and item["filename"].endswith(f"{machine}.whl")
    and not item.get("yanked", False)
]
if len(candidates) != 1:
    raise SystemExit(
        f"Expected exactly one manylinux uv {version} wheel for {machine}; "
        f"found {[item['filename'] for item in candidates]}"
    )
candidate = candidates[0]
expected_sha256 = candidate["digests"]["sha256"]

os.makedirs(install_dir, exist_ok=True)
with tempfile.NamedTemporaryFile(suffix=".whl") as wheel:
    digest = hashlib.sha256()
    with urllib.request.urlopen(candidate["url"], timeout=120) as response:
        while chunk := response.read(1024 * 1024):
            wheel.write(chunk)
            digest.update(chunk)
    wheel.flush()
    if digest.hexdigest() != expected_sha256:
        raise SystemExit(
            f"uv wheel SHA256 mismatch: expected {expected_sha256}, "
            f"found {digest.hexdigest()}"
        )
    with zipfile.ZipFile(wheel.name) as archive:
        members = [name for name in archive.namelist() if name.endswith(".data/scripts/uv")]
        if len(members) != 1:
            raise SystemExit(f"Unexpected uv wheel layout: {members}")
        target = os.path.join(install_dir, "uv")
        descriptor, temporary_target = tempfile.mkstemp(prefix=".uv-", dir=install_dir)
        try:
            with archive.open(members[0]) as source, os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(
                temporary_target,
                os.stat(temporary_target).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            os.replace(temporary_target, target)
        finally:
            if os.path.exists(temporary_target):
                os.unlink(temporary_target)
PY
    UV_BIN=$install_dir/uv
  fi

  if ! uv_has_exact_version "$UV_BIN" "$uv_version"; then
    echo "uv $uv_version validation failed: $UV_BIN" >&2
    return 1
  fi

  export UV_BIN
  "$UV_BIN" --version
  echo "UV_CACHE_DIR=$UV_CACHE_DIR"
}

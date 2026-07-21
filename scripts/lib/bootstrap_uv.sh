#!/usr/bin/env bash

# Locate uv or install a project-local standalone binary. This deliberately
# avoids `python -m venv` and ensurepip, which are often unavailable in minimal
# Python/cluster environments.
ensure_uv() {
  local project_root=$1
  local install_dir=${UV_BOOTSTRAP_DIR:-$project_root/.bootstrap-tools}

  if command -v uv >/dev/null 2>&1; then
    UV_BIN=$(command -v uv)
  elif [[ -x "$install_dir/uv" ]]; then
    UV_BIN=$install_dir/uv
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "uv is not installed and curl is unavailable." >&2
      echo "Install uv or make curl available, then rerun the bootstrap." >&2
      return 1
    fi

    mkdir -p "$install_dir"
    echo "Installing a project-local uv binary into $install_dir"
    curl -LsSf https://astral.sh/uv/install.sh | \
      env UV_UNMANAGED_INSTALL="$install_dir" sh
    UV_BIN=$install_dir/uv
  fi

  if [[ ! -x "$UV_BIN" ]]; then
    echo "uv executable is missing or not executable: $UV_BIN" >&2
    return 1
  fi

  export UV_BIN
  "$UV_BIN" --version
}

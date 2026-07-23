#!/usr/bin/env bash

validate_bootstrap_flag() {
  local flag_name
  local value
  for flag_name in FORCE_BOOTSTRAP UV_OFFLINE; do
    value=${!flag_name:-0}
    if [[ "$value" != 0 && "$value" != 1 ]]; then
      echo "$flag_name must be 0 or 1; got '$value'." >&2
      return 2
    fi
  done
}

bootstrap_marker_matches() {
  local control_python=$1
  local marker=$2
  local environment=$3
  local signature=$4
  "$control_python" -m stackpilot.bootstrap_cache marker-matches \
    --marker "$marker" --environment "$environment" --signature "$signature"
}

write_bootstrap_marker() {
  local environment_python=$1
  local marker=$2
  local environment=$3
  local signature=$4
  "$environment_python" "$ROOT/stackpilot/bootstrap_cache.py" write-marker \
    --marker "$marker" --environment "$environment" --signature "$signature"
}

bootstrap_interpreter_compatible() {
  local control_python=$1
  local environment_python=$2
  local base_python=$3
  "$control_python" -m stackpilot.bootstrap_cache interpreter-compatible \
    --environment-python "$environment_python" --base-python "$base_python"
}

uv_pip_install_cached_first() {
  local uv_bin=$1
  shift
  if [[ ${UV_OFFLINE:-0} == 1 ]]; then
    echo "UV_OFFLINE=1: installing strictly from the existing uv cache."
    "$uv_bin" pip install --offline "$@"
  else
    # uv always reuses compatible wheels and source archives from UV_CACHE_DIR.
    # An online transaction therefore downloads only artifacts absent from the
    # persistent cache, without a speculative failed offline install first.
    echo "Reusing UV_CACHE_DIR; only uncached package artifacts will be downloaded."
    "$uv_bin" pip install "$@"
  fi
}

prepare_cached_venv() {
  local control_python=$1
  local base_python=$2
  local environment_python=$3
  local environment_dir=$4
  local uv_bin=$5

  if [[ ${FORCE_BOOTSTRAP:-0} == 1 ]]; then
    echo "FORCE_BOOTSTRAP=1: rebuilding $environment_dir"
    "$uv_bin" venv --clear --no-project --python "$base_python" "$environment_dir"
    return 0
  fi
  if [[ -x "$environment_python" ]] && \
     bootstrap_interpreter_compatible \
       "$control_python" "$environment_python" "$base_python"; then
    echo "Repairing the compatible environment in place: $environment_dir"
    return 0
  fi
  if [[ -e "$environment_dir" ]]; then
    echo "The existing environment has an incompatible or broken Python ABI; recreating it: $environment_dir"
  else
    echo "Creating missing environment: $environment_dir"
  fi
  "$uv_bin" venv --clear --no-project --python "$base_python" "$environment_dir"
}

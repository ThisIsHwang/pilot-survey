#!/usr/bin/env bash

_jdk_is_compatible() {
  local javac_bin=$1
  local version
  local major

  [[ -x "$javac_bin" ]] || return 1
  version=$("$javac_bin" -version 2>&1 | awk 'NR == 1 { print $2 }')
  major=${version%%.*}
  [[ "$major" =~ ^[0-9]+$ && "$major" -ge 21 ]]
}

# Export a complete JDK for Pyserini/PyJNIus. A JRE is insufficient because
# PyJNIus uses javac to locate the JVM. If no suitable system JDK is available,
# install a verified Eclipse Temurin JDK 21 under .bootstrap-tools.
ensure_java() {
  local project_root=$1
  local tools_dir=${JAVA_BOOTSTRAP_DIR:-$project_root/.bootstrap-tools}
  local install_dir=$tools_dir/jdk-21
  local javac_path=""
  local adoptium_arch
  local api_url
  local download_dir
  local archive
  local fetch_url
  local expected_sha
  local actual_sha

  if [[ -n "${JAVA_HOME:-}" ]] && _jdk_is_compatible "$JAVA_HOME/bin/javac"; then
    javac_path=$JAVA_HOME/bin/javac
  elif command -v javac >/dev/null 2>&1 && _jdk_is_compatible "$(command -v javac)"; then
    javac_path=$(readlink -f "$(command -v javac)")
    JAVA_HOME=${javac_path%/bin/javac}
  elif _jdk_is_compatible "$install_dir/bin/javac"; then
    JAVA_HOME=$install_dir
    javac_path=$JAVA_HOME/bin/javac
  else
    if [[ "$(uname -s)" != "Linux" ]]; then
      echo "Automatic JDK installation supports Linux only; found $(uname -s)." >&2
      return 1
    fi
    case "$(uname -m)" in
      x86_64) adoptium_arch=x64 ;;
      aarch64|arm64) adoptium_arch=aarch64 ;;
      *)
        echo "Unsupported architecture for the Temurin JDK: $(uname -m)" >&2
        return 1
        ;;
    esac
    for tool in curl tar sha256sum; do
      if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required JDK bootstrap tool is unavailable: $tool" >&2
        return 1
      fi
    done

    mkdir -p "$tools_dir" "$install_dir"
    download_dir=$(mktemp -d "$tools_dir/jdk-download.XXXXXX")
    archive=$download_dir/temurin-jdk.tar.gz
    api_url="https://api.adoptium.net/v3/binary/latest/21/ga/linux/${adoptium_arch}/jdk/hotspot/normal/eclipse"

    echo "Downloading Eclipse Temurin JDK 21 into $install_dir"
    fetch_url=$(curl -fsS --retry 3 \
      -o /dev/null \
      -w '%{redirect_url}' \
      "$api_url")
    if [[ -z "$fetch_url" ]]; then
      echo "Adoptium API did not return a JDK download URL." >&2
      rmdir "$download_dir"
      return 1
    fi
    curl -fsSL --retry 3 -o "$archive" "$fetch_url"
    expected_sha=$(curl -fsSL --retry 3 "${fetch_url}.sha256.txt" | awk 'NR == 1 { print $1 }')
    actual_sha=$(sha256sum "$archive" | awk '{ print $1 }')
    if [[ -z "$expected_sha" || "$actual_sha" != "$expected_sha" ]]; then
      echo "Temurin JDK checksum verification failed." >&2
      rm -f -- "$archive"
      rmdir "$download_dir"
      return 1
    fi

    tar -xzf "$archive" -C "$install_dir" --strip-components=1
    rm -f -- "$archive"
    rmdir "$download_dir"
    JAVA_HOME=$install_dir
    javac_path=$JAVA_HOME/bin/javac
  fi

  if ! _jdk_is_compatible "$javac_path"; then
    echo "A complete JDK 21+ with javac is required; JAVA_HOME=${JAVA_HOME:-unset}." >&2
    return 1
  fi

  export JAVA_HOME
  export PATH="$JAVA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$JAVA_HOME/lib/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  java -version
  javac -version
}

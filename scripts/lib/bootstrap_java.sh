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
# install a verified Amazon Corretto JDK 21 under .bootstrap-tools. Corretto's
# permanent download URLs avoid the GitHub release-asset redirects used by
# Adoptium, which are often blocked or stale behind cluster egress proxies.
ensure_java() {
  local project_root=$1
  local tools_dir=${JAVA_BOOTSTRAP_DIR:-$project_root/.bootstrap-tools}
  local install_dir=$tools_dir/jdk-21
  local javac_path=""
  local jdk_arch
  local download_url
  local checksum_url
  local download_dir
  local archive
  local install_stage
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
      x86_64) jdk_arch=x64 ;;
      aarch64|arm64) jdk_arch=aarch64 ;;
      *)
        echo "Unsupported architecture for the Corretto JDK: $(uname -m)" >&2
        return 1
        ;;
    esac
    for tool in curl tar sha256sum; do
      if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required JDK bootstrap tool is unavailable: $tool" >&2
        return 1
      fi
    done

    mkdir -p "$tools_dir"
    download_dir=$(mktemp -d "$tools_dir/jdk-download.XXXXXX")
    archive=$download_dir/corretto-jdk.tar.gz
    install_stage=$download_dir/jdk
    download_url=${JAVA_BOOTSTRAP_URL:-https://corretto.aws/downloads/latest/amazon-corretto-21-${jdk_arch}-linux-jdk.tar.gz}
    checksum_url=${JAVA_BOOTSTRAP_SHA256_URL:-https://corretto.aws/downloads/latest_sha256/amazon-corretto-21-${jdk_arch}-linux-jdk.tar.gz}

    echo "Downloading Amazon Corretto JDK 21 into $install_dir"
    if ! expected_sha=$(curl -fsSL --retry 5 --retry-delay 2 \
      --connect-timeout 30 \
      -H 'Cache-Control: no-cache' \
      -A 'pilot-survey-jdk-bootstrap/1.0' \
      "$checksum_url" | awk 'NR == 1 { print $1 }'); then
      echo "Unable to download the Corretto JDK checksum: $checksum_url" >&2
      rm -rf -- "$download_dir"
      return 1
    fi
    if [[ ! "$expected_sha" =~ ^[0-9a-fA-F]{64}$ ]]; then
      echo "The Corretto JDK checksum response was invalid." >&2
      rm -rf -- "$download_dir"
      return 1
    fi
    if ! curl -fsSL --retry 5 --retry-delay 2 \
      --connect-timeout 30 \
      -H 'Cache-Control: no-cache' \
      -A 'pilot-survey-jdk-bootstrap/1.0' \
      -o "$archive" \
      "$download_url"; then
      echo "Unable to download the Corretto JDK archive: $download_url" >&2
      rm -rf -- "$download_dir"
      return 1
    fi
    actual_sha=$(sha256sum "$archive" | awk '{ print $1 }')
    if [[ "${actual_sha,,}" != "${expected_sha,,}" ]]; then
      echo "Corretto JDK checksum verification failed." >&2
      rm -rf -- "$download_dir"
      return 1
    fi

    mkdir -p "$install_stage"
    if ! tar -xzf "$archive" -C "$install_stage" --strip-components=1; then
      echo "Unable to extract the Corretto JDK archive." >&2
      rm -rf -- "$download_dir"
      return 1
    fi
    if ! _jdk_is_compatible "$install_stage/bin/javac"; then
      echo "The downloaded archive does not contain a compatible JDK 21." >&2
      rm -rf -- "$download_dir"
      return 1
    fi

    rm -rf -- "$install_dir"
    mv "$install_stage" "$install_dir"
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

#!/usr/bin/env bash

_java_is_compatible() {
  local java_bin=$1
  local version
  local major

  [[ -x "$java_bin" ]] || return 1
  version=$("$java_bin" -version 2>&1 | awk -F'"' 'NR == 1 { print $2 }')
  major=${version%%.*}
  [[ "$major" =~ ^[0-9]+$ && "$major" -ge 21 ]]
}

# Export Java 21 for Pyserini/PyJNIus. Cluster egress policies commonly block
# direct JDK archive downloads, so the fallback is a platform-specific jdk4py
# wheel from PyPI. PyJNIus does not search for javac when JAVA_HOME is explicit.
ensure_java() {
  local project_root=$1
  local java_path=""
  local java_home=""
  local python_bin
  local uv_bin
  local bundled_home

  if [[ -n "${JAVA_HOME:-}" ]] && _java_is_compatible "$JAVA_HOME/bin/java"; then
    java_home=$JAVA_HOME
    java_path=$java_home/bin/java
  elif [[ -n "${JDK_HOME:-}" ]] && _java_is_compatible "$JDK_HOME/bin/java"; then
    java_home=$JDK_HOME
    java_path=$java_home/bin/java
  elif command -v java >/dev/null 2>&1 && _java_is_compatible "$(command -v java)"; then
    java_path=$(readlink -f "$(command -v java)")
    java_home=${java_path%/bin/java}
  else
    python_bin=${JAVA_BOOTSTRAP_PYTHON:-$project_root/.venv-pilot/bin/python}
    if [[ ! -x "$python_bin" ]]; then
      echo "Python environment required for bundled Java is missing: $python_bin" >&2
      return 1
    fi

    if ! bundled_home=$("$python_bin" -c \
      'from jdk4py import JAVA_HOME; print(JAVA_HOME)' 2>/dev/null) || \
      ! _java_is_compatible "$bundled_home/bin/java"; then
      uv_bin=${UV_BIN:-}
      if [[ -z "$uv_bin" ]] && command -v uv >/dev/null 2>&1; then
        uv_bin=$(command -v uv)
      elif [[ -z "$uv_bin" && -x "$project_root/.bootstrap-tools/uv" ]]; then
        uv_bin=$project_root/.bootstrap-tools/uv
      fi
      if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
        echo "uv is required to install the bundled Java 21 wheel." >&2
        echo "Rerun scripts/bootstrap.sh, then retry this command." >&2
        return 1
      fi

      echo "Installing bundled Java 21 from PyPI into .venv-pilot"
      UV_DEFAULT_INDEX=https://pypi.org/simple UV_LINK_MODE=copy \
        "$uv_bin" pip install \
          --python "$python_bin" \
          'jdk4py==21.0.8.2'
      bundled_home=$("$python_bin" -c \
        'from jdk4py import JAVA_HOME; print(JAVA_HOME)')
    fi

    java_home=$bundled_home
    java_path=$java_home/bin/java
  fi

  if ! _java_is_compatible "$java_path"; then
    echo "A Java 21+ runtime is required; JAVA_HOME=${JAVA_HOME:-unset}." >&2
    return 1
  fi

  JAVA_HOME=$java_home
  JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
  export JAVA_HOME JVM_PATH
  export PATH="$JAVA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$JAVA_HOME/lib/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # Pyserini 0.25 bundles Lucene 9.9 in an assembled JAR that omits Lucene's
  # Java 21 multi-release MemorySegment provider. Use Lucene's documented
  # ByteBuffer fallback. JAVA_TOOL_OPTIONS also applies to PyJNIus' embedded
  # JVM because it is read by JNI_CreateJavaVM.
  local lucene_mmap_option=-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false
  if [[ " ${JAVA_TOOL_OPTIONS:-} " != *" $lucene_mmap_option "* ]]; then
    JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }$lucene_mmap_option"
  fi
  export JAVA_TOOL_OPTIONS

  "$JAVA_HOME/bin/java" -version
  echo "JAVA_HOME=$JAVA_HOME"
  echo "JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS"
}

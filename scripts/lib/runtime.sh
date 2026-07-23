#!/usr/bin/env bash

ensure_local_no_proxy() {
  local value=${NO_PROXY:-${no_proxy:-}}
  case ",$value," in
    *,127.0.0.1,*) ;;
    *) value=${value:+$value,}127.0.0.1 ;;
  esac
  case ",$value," in
    *,localhost,*) ;;
    *) value=${value:+$value,}localhost ;;
  esac
  NO_PROXY=$value
  no_proxy=$value
  export NO_PROXY no_proxy
}

port_is_open() {
  local python_bin=$1
  local port=$2
  "$python_bin" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(0.5); rc=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(0 if rc == 0 else 1)' \
    "$port"
}

require_free_port() {
  local python_bin=$1
  local port=$2
  if port_is_open "$python_bin" "$port"; then
    echo "Port $port is already in use. Run bash scripts/stop_servers.sh or choose another port." >&2
    return 1
  fi
}

process_cmdline() {
  local pid=$1
  if [[ -r "/proc/$pid/cmdline" ]]; then
    tr '\0' ' ' < "/proc/$pid/cmdline"
  fi
}

quarantine_pid_file() {
  local pid_file=$1
  local pid=${2:-unknown}
  local quarantine
  quarantine="${pid_file}.stale.$(date +%s).${pid}"
  mv -- "$pid_file" "$quarantine"
  echo "Moved untrusted PID file to: $quarantine" >&2
}

start_managed_process() {
  local launcher_python=$1
  local log_file=$2
  shift 2
  "$launcher_python" - "$log_file" "$@" <<'PY'
import os
import subprocess
import sys

log_path = sys.argv[1]
command = sys.argv[2:]
if not command:
    raise SystemExit("managed process command is empty")
os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
with open(log_path, "wb") as log:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print(process.pid)
PY
}

stop_managed_pid() {
  local pid_file=$1
  local expected=$2
  local expected_cwd=${3:-}
  local stop_group=${4:-0}
  local pid
  local cmdline
  local actual_cwd
  local session_id
  local signal_target
  local _

  [[ -f "$pid_file" ]] || return 0
  pid=$(<"$pid_file")
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Ignoring invalid PID file: $pid_file" >&2
    quarantine_pid_file "$pid_file" invalid
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$pid_file"
    return 0
  fi
  cmdline=$(process_cmdline "$pid")
  if [[ "$cmdline" != *"$expected"* ]]; then
    echo "Refusing to kill PID $pid; its command does not contain '$expected': $cmdline" >&2
    quarantine_pid_file "$pid_file" "$pid"
    return 1
  fi
  if [[ -n "$expected_cwd" ]]; then
    expected_cwd=$(readlink -f "$expected_cwd" 2>/dev/null || printf '%s' "$expected_cwd")
    actual_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    if [[ "$actual_cwd" != "$expected_cwd" ]]; then
      echo "Refusing to kill PID $pid; cwd is '$actual_cwd', expected '$expected_cwd'." >&2
      quarantine_pid_file "$pid_file" "$pid"
      return 1
    fi
  fi

  signal_target=$pid
  if [[ "$stop_group" == 1 ]]; then
    session_id=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    if [[ "$session_id" == "$pid" ]]; then
      signal_target="-$pid"
    else
      echo "PID $pid is not a managed session leader; stopping only that legacy parent." >&2
    fi
  fi

  kill -- "$signal_target" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 -- "$signal_target" 2>/dev/null; then
      rm -f -- "$pid_file"
      return 0
    fi
    sleep 0.5
  done
  echo "Managed process $signal_target did not stop after SIGTERM; sending SIGKILL." >&2
  kill -KILL -- "$signal_target" 2>/dev/null || true
  for _ in $(seq 1 40); do
    if ! kill -0 -- "$signal_target" 2>/dev/null; then
      rm -f -- "$pid_file"
      return 0
    fi
    sleep 0.25
  done
  echo "Managed process $signal_target is still visible after SIGKILL; preserving $pid_file and refusing an immediate GPU restart." >&2
  return 1
}

show_log_tail() {
  local log_file=$1
  if [[ -f "$log_file" ]]; then
    echo "----- tail: $log_file -----" >&2
    tail -n 80 "$log_file" >&2 || true
  fi
}

wait_for_http() {
  local pid=$1
  local url=$2
  local timeout_seconds=$3
  local log_file=$4
  local expected=${5:-}
  local deadline=$((SECONDS + timeout_seconds))
  local response

  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Server process $pid exited before becoming ready: $url" >&2
      show_log_tail "$log_file"
      return 1
    fi
    if response=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "$url" 2>/dev/null); then
      if [[ -z "$expected" || "$response" == *"$expected"* ]]; then
        return 0
      fi
    fi
    sleep 2
  done
  echo "Timed out after ${timeout_seconds}s waiting for $url" >&2
  show_log_tail "$log_file"
  return 1
}

validate_gpu_list() {
  local gpu_list=$1
  local expected_count=$2
  local label=$3
  local -a ids
  local seen=,
  local id
  IFS=',' read -r -a ids <<< "$gpu_list"
  if [[ ${#ids[@]} -ne $expected_count ]]; then
    echo "$label requires $expected_count GPU IDs; got '$gpu_list'." >&2
    return 1
  fi
  for id in "${ids[@]}"; do
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU ID in $label: '$id'" >&2
      return 1
    fi
    if [[ "$seen" == *",$id,"* ]]; then
      echo "Duplicate GPU ID in $label: '$id'" >&2
      return 1
    fi
    seen+="$id,"
  done
}

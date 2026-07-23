from __future__ import annotations

import argparse
import os
from pathlib import Path


def atomic_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{pid}\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command in a tracked process group"
    )
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    if os.name != "posix":
        parser.error("session_runner requires a POSIX process-session implementation")

    # Become the process-group leader before publishing our PID, then replace
    # this process with the real job. Consequently the shell PID, process-group
    # ID, and actual job PID stay identical and no untracked child can exist.
    os.setsid()
    atomic_pid(Path(args.pid_file), os.getpid())
    os.execvpe(command[0], command, os.environ.copy())
    return 127  # pragma: no cover - a successful exec never returns


if __name__ == "__main__":
    raise SystemExit(main())

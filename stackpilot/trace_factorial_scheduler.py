from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from stackpilot.trace_common import atomic_write_json, read_jsonl


def completed(job: dict[str, Any]) -> bool:
    metrics_path = Path(job["output_dir"]) / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("job_signature") == job.get("job_signature")


def _runner_module(job: dict[str, Any]) -> str:
    module = str(job.get("runner_module", "stackpilot.trace_lora_job"))
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
    if not module.startswith("stackpilot.") or any(
        character not in allowed for character in module
    ):
        raise ValueError(f"Unsafe TRACE runner module: {module!r}")
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run positive factorial LoRA jobs across one H100 node."
    )
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--launch-stagger", type=float, default=2.0)
    args = parser.parse_args()

    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain unique non-negative GPU IDs")
    workers = min(args.workers or len(args.gpus), len(args.gpus))
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = read_jsonl(args.jobs)
    for job in jobs:
        if job.get("experiment_id") != "EXP-012":
            raise RuntimeError(
                "Factorial scheduler accepts only EXP-012; found "
                f"{job.get('experiment_id')!r}"
            )
        _runner_module(job)
    if not args.force:
        jobs = [job for job in jobs if not completed(job)]
    pending = deque(jobs)
    available = deque(args.gpus[:workers])
    running: dict[int, tuple[subprocess.Popen[bytes], dict[str, Any], Any]] = {}
    failures: list[dict[str, Any]] = []
    interrupted = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print(
            f"EXP-012 scheduler received signal {signum}; stopping workers.",
            flush=True,
        )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while (pending or running) and not interrupted:
            while pending and available and not interrupted:
                gpu = available.popleft()
                job = pending.popleft()
                log_path = Path(job["output_dir"]) / "job.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["TOKENIZERS_PARALLELISM"] = "false"
                command = [
                    args.python,
                    "-m",
                    _runner_module(job),
                    "--job",
                    str(Path(job["job_file"]).resolve()),
                ]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[gpu] = (process, job, log)
                print(
                    f"GPU {gpu}: started {job['job_id']} (PID {process.pid})",
                    flush=True,
                )
                if args.launch_stagger > 0:
                    time.sleep(args.launch_stagger)

            if not running:
                break
            time.sleep(2)
            for gpu in list(running):
                process, job, log = running[gpu]
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                available.append(gpu)
                del running[gpu]
                if return_code != 0:
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "gpu": gpu,
                            "return_code": return_code,
                            "log": str(Path(job["output_dir"]) / "job.log"),
                        }
                    )
                    print(
                        f"GPU {gpu}: FAILED {job['job_id']} ({return_code})",
                        flush=True,
                    )
                else:
                    print(f"GPU {gpu}: completed {job['job_id']}", flush=True)
    finally:
        for process, _job, _log in running.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 30.0
        for process, _job, log in running.values():
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()

    if interrupted:
        raise SystemExit(
            "EXP-012 scheduler interrupted; unfinished jobs remain resumable"
        )

    summary_path = Path(args.jobs).with_name("scheduler_summary.json")
    atomic_write_json(
        summary_path,
        {
            "requested_jobs": len(jobs),
            "failed_jobs": failures,
            "success": not failures,
        },
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} EXP-012 jobs failed; see {summary_path}"
        )
    print(f"EXP-012 scheduler completed {len(jobs)} jobs")


if __name__ == "__main__":
    main()

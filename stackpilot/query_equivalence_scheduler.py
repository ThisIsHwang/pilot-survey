from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

from stackpilot.query_equivalence_common import read_jsonl


def completed(job: dict[str, Any]) -> bool:
    path = Path(job["output_dir"]) / "metrics.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("job_signature") == job.get("job_signature")


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule EXP-015 one-GPU LoRA jobs.")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--launch-stagger", type=float, default=2.0)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--direction", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs)
    if args.variant:
        allowed = set(args.variant)
        jobs = [job for job in jobs if str(job["variant"]) in allowed]
    if args.direction:
        allowed = set(args.direction)
        jobs = [job for job in jobs if str(job["direction"]) in allowed]
    if not args.force:
        jobs = [job for job in jobs if not completed(job)]
    if not jobs:
        print("No unfinished EXP-015 jobs match the requested filters.")
        return
    workers = min(int(args.workers), len(args.gpus))
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = deque(jobs)
    available = deque(args.gpus[:workers])
    running: dict[int, tuple[subprocess.Popen[Any], dict[str, Any], Any]] = {}
    failures: list[dict[str, Any]] = []
    interrupted = False

    def stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"Scheduler received signal {signum}; stopping workers.", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
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
                    "stackpilot.query_equivalence_lora_job",
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
                print(f"GPU {gpu}: started {job['job_id']} (PID {process.pid})", flush=True)
                if args.launch_stagger > 0:
                    time.sleep(args.launch_stagger)
            if not running:
                break
            time.sleep(2.0)
            for gpu in list(running):
                process, job, log = running[gpu]
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                available.append(gpu)
                del running[gpu]
                if return_code != 0:
                    failure = {
                        "job_id": job["job_id"],
                        "gpu": gpu,
                        "return_code": return_code,
                        "log": str(Path(job["output_dir"]) / "job.log"),
                    }
                    failures.append(failure)
                    print(f"GPU {gpu}: FAILED {job['job_id']} ({return_code})", flush=True)
                else:
                    print(f"GPU {gpu}: completed {job['job_id']}", flush=True)
    finally:
        for process, _job, log in running.values():
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
        raise SystemExit("EXP-015 scheduler interrupted; unfinished jobs remain resumable")
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()

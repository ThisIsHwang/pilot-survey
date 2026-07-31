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

from stackpilot.query_equivalence_common import atomic_write_json, read_jsonl


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
    parser = argparse.ArgumentParser(description="Run EXP-014 jobs across one H100 node.")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--launch-stagger", type=float, default=2.0)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(value < 0 for value in args.gpus):
        raise ValueError("--gpus must contain unique non-negative IDs")
    workers = min(args.workers or len(args.gpus), len(args.gpus))
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = read_jsonl(args.jobs)
    if not args.force:
        jobs = [job for job in jobs if not completed(job)]
    pending = deque(jobs)
    available = deque(args.gpus[:workers])
    running: dict[int, tuple[subprocess.Popen[bytes], dict[str, Any], Any]] = {}
    failures = []
    interrupted = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"EXP-014 scheduler received signal {signum}; stopping workers.", flush=True)

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
                    args.python, "-m",
                    str(job.get("runner_module", "stackpilot.query_equivalence_lora_job")),
                    "--job", str(Path(job["job_file"]).resolve()),
                ]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[gpu] = (process, job, log)
                print(f"GPU {gpu}: started {job['job_id']} (PID {process.pid})", flush=True)
                if args.launch_stagger > 0:
                    time.sleep(args.launch_stagger)
            if not running:
                break
            time.sleep(2)
            for gpu in list(running):
                process, job, log = running[gpu]
                code = process.poll()
                if code is None:
                    continue
                log.close()
                available.append(gpu)
                del running[gpu]
                if code != 0:
                    failures.append({
                        "job_id": job["job_id"], "gpu": gpu, "return_code": code,
                        "log": str(Path(job["output_dir"]) / "job.log"),
                    })
                    print(f"GPU {gpu}: FAILED {job['job_id']} ({code})", flush=True)
                else:
                    print(f"GPU {gpu}: completed {job['job_id']}", flush=True)
    finally:
        for process, _job, _log in running.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 30
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
        raise SystemExit("EXP-014 scheduler interrupted; jobs remain resumable")
    summary = Path(args.jobs).with_name("scheduler_summary.json")
    atomic_write_json(summary, {
        "requested_jobs": len(jobs), "failed_jobs": failures, "success": not failures,
    })
    if failures:
        raise SystemExit(f"{len(failures)} EXP-014 jobs failed; see {summary}")
    print(f"EXP-014 scheduler completed {len(jobs)} jobs")


if __name__ == "__main__":
    main()

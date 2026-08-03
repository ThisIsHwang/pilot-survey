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

from stackpilot.query_attribution_common import read_jsonl


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
    parser = argparse.ArgumentParser(description="Schedule attribution-hypothesis jobs.")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--launch-stagger", type=float, default=2.0)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain unique non-negative IDs")
    jobs = read_jsonl(args.jobs)
    if args.experiment:
        allowed = set(args.experiment)
        jobs = [job for job in jobs if job["experiment_id"] in allowed]
    if args.variant:
        allowed = set(args.variant)
        jobs = [job for job in jobs if job["variant"] in allowed]
    if not args.force:
        jobs = [job for job in jobs if not completed(job)]
    if not jobs:
        print("No unfinished attribution jobs match the filters.")
        return
    workers = min(int(args.workers), len(args.gpus))
    pending = deque(jobs)
    available = deque(args.gpus[:workers])
    running: dict[int, tuple[subprocess.Popen[Any], dict[str, Any], Any]] = {}
    failures = []
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
                command = [args.python, "-m", str(job.get("runner_module", "stackpilot.query_attribution_lora_job")), "--job", str(Path(job["job_file"]).resolve())]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[gpu] = (process, job, log)
                print(f"GPU {gpu}: started {job['job_id']} (PID {process.pid})", flush=True)
                if args.launch_stagger > 0:
                    time.sleep(args.launch_stagger)
            if not running:
                break
            time.sleep(2.0)
            for gpu in list(running):
                process, job, log = running[gpu]
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[gpu]
                available.append(gpu)
                if code != 0:
                    failures.append({"job_id": job["job_id"], "gpu": gpu, "return_code": code, "log": str(Path(job["output_dir"]) / "job.log")})
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
        raise SystemExit("Attribution scheduler interrupted; jobs remain resumable")
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()

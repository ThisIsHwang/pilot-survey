from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from stackpilot.trace_common import atomic_write_json, read_jsonl
from stackpilot.trace_lora_job import main as run_trace_lora_job


def _validate_positive_job(job_path: Path) -> dict[str, Any]:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("experiment_id") != "EXP-012":
        raise RuntimeError(
            f"Positive factorial runner accepts only EXP-012, found "
            f"{job.get('experiment_id')!r}"
        )
    if job.get("weight_mode") != "positive-only":
        raise RuntimeError(
            "EXP-012 requires weight_mode='positive-only', found "
            f"{job.get('weight_mode')!r}"
        )
    rows = read_jsonl(job["train_file"])
    if not rows:
        raise RuntimeError("EXP-012 training file is empty")
    invalid = []
    for index, row in enumerate(rows):
        value = float(row.get("weight", float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            invalid.append((str(row.get("example_id", index)), row.get("weight")))
    if invalid:
        raise RuntimeError(
            "EXP-012 forbids zero or negative training weights; "
            f"first invalid rows: {invalid[:5]!r}"
        )
    return job


def _validate_outputs(job: dict[str, Any]) -> None:
    output_dir = Path(job["output_dir"])
    metrics_path = output_dir / "metrics.json"
    losses_path = output_dir / "eval_losses.jsonl"
    invalid: list[dict[str, Any]] = []
    if not metrics_path.is_file() or not losses_path.is_file():
        raise RuntimeError(
            f"EXP-012 runner produced incomplete output under {output_dir}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value):
            invalid.append(
                {"scope": "metrics", "field": name, "value": repr(value)}
            )
    training = metrics.get("training") or {}
    for name in ("mean_micro_loss", "processed_target_tokens", "elapsed_seconds"):
        value = float(training.get(name, float("nan")))
        if not math.isfinite(value):
            invalid.append(
                {"scope": "training", "field": name, "value": repr(value)}
            )
    for row in read_jsonl(losses_path):
        for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
            value = float(row.get(name, float("nan")))
            if not math.isfinite(value):
                invalid.append(
                    {
                        "scope": "example",
                        "example_id": str(row.get("example_id", "")),
                        "field": name,
                        "value": repr(value),
                    }
                )
                if len(invalid) >= 20:
                    break
        if len(invalid) >= 20:
            break
    if invalid:
        payload = {
            "schema": 1,
            "status": "invalid",
            "job_id": job["job_id"],
            "job_signature": job["job_signature"],
            "reason": "non-finite training or evaluation values",
            "examples": invalid,
            "detected_at_unix": time.time(),
        }
        atomic_write_json(output_dir / "invalid.json", payload)
        metrics_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"EXP-012 produced non-finite values for {job['job_id']}; "
            f"see {output_dir / 'invalid.json'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args, _ = parser.parse_known_args()
    job_path = Path(args.job).resolve()
    job = _validate_positive_job(job_path)
    run_trace_lora_job()
    _validate_outputs(job)


if __name__ == "__main__":
    main()

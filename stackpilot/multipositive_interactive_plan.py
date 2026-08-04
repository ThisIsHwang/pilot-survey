from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.multipositive_common import atomic_write_json, atomic_write_jsonl, file_sha256, load_config, read_jsonl, signature, stable_hash


def plan(config_path: Path, profile: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    work_root = Path(cfg["work_dir"]).resolve()
    training_root = work_root / "plans" / profile
    training_jobs_path = training_root / "jobs.jsonl"
    if not training_jobs_path.is_file():
        raise RuntimeError("Training plan is missing")
    allowed = set(map(str, cfg["interactive"]["variants"]))
    selected = [job for job in read_jsonl(training_jobs_path) if str(job["variant"]) in allowed]
    output_root = work_root / "interactive_plans" / profile
    data_root = output_root / "data"
    specs_root = output_root / "job_specs"
    results_root = work_root / "interactive_outputs" / profile
    for root in (data_root, specs_root, results_root):
        root.mkdir(parents=True, exist_ok=True)
    max_states = int(cfg["interactive"]["eval_states_per_direction"][profile])
    eval_files: dict[str, Path] = {}
    jobs = []
    for job in selected:
        direction = str(job["direction"])
        if direction not in eval_files:
            groups = read_jsonl(job["eval_file"])
            groups = sorted(groups, key=lambda row: stable_hash("interactive", direction, row["question_id"], row["state_id"]))[:max_states]
            path = data_root / f"eval__{direction}.jsonl"
            atomic_write_jsonl(path, groups)
            eval_files[direction] = path
        target_backend = str(job["metadata"]["target_backend"])
        adapter_dir = Path(job["output_dir"]) / "adapter"
        interactive = {
            "schema": 1,
            "suite_id": cfg["suite_id"],
            "experiment_id": "EXP-027",
            "profile": profile,
            "direction": direction,
            "variant": str(job["variant"]),
            "seed": int(job["seed"]),
            "base_model": str(job["base_model"]),
            "adapter_dir": str(adapter_dir.resolve()),
            "eval_file": str(eval_files[direction].resolve()),
            "eval_file_sha256": file_sha256(eval_files[direction]),
            "retrieval_url": str(cfg["interactive"][f"{target_backend}_url"]),
            "sample_budgets": [int(value) for value in cfg["interactive"]["sample_budgets"]],
            "temperature": float(cfg["interactive"]["temperature"]),
            "max_new_tokens": int(cfg["interactive"]["max_new_tokens"]),
            "output_dir": str((results_root / direction / str(job["variant"]) / f"seed-{int(job['seed']):03d}").resolve()),
            "source_job_signature": str(job["job_signature"]),
            "runner_module": "stackpilot.multipositive_interactive_job",
        }
        interactive["job_id"] = f"EXP-027__seed-{int(job['seed']):03d}__profile-{profile}__variant-{direction}__{job['variant']}"
        interactive["job_signature"] = signature(interactive)
        job_file = specs_root / f"{interactive['job_id']}.json"
        interactive["job_file"] = str(job_file.resolve())
        atomic_write_json(job_file, interactive)
        jobs.append(interactive)
    jobs_path = output_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {"schema": 1, "suite_id": cfg["suite_id"], "experiment_id": "EXP-027", "profile": profile, "jobs_path": str(jobs_path), "jobs_sha256": file_sha256(jobs_path), "job_count": len(jobs), "variants": sorted(allowed)}
    manifest["signature"] = signature({**manifest, "job_signatures": [job["job_signature"] for job in jobs]})
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan interactive multi-positive evaluation.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(plan(Path(args.config).resolve(), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

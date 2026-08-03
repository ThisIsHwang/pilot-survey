from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.query_attribution_common import SCHEMA, atomic_write_json, atomic_write_jsonl, balanced_sample, file_sha256, load_config, read_jsonl, signature


def plan(config_path: Path, profile: str) -> dict[str, Any]:
    cfg = load_config(config_path); root = Path(cfg["work_dir"]).resolve(); nll_plan_root = root / "plans" / profile; nll_jobs_path = nll_plan_root / "jobs.jsonl"; nll_manifest_path = nll_plan_root / "manifest.json"
    if not nll_jobs_path.is_file() or not nll_manifest_path.is_file():
        raise RuntimeError("NLL attribution plan is missing; run query_attribution/plan.sh")
    nll_jobs = read_jsonl(nll_jobs_path); enabled = set(map(str, cfg["interactive"]["enabled_variants"])); count = int(cfg["interactive"]["eval_states_per_direction"][profile]); output_root = root / "interactive_plans" / profile; data_root = output_root / "data"; jobs = []; eval_cache = {}
    for nll_job in nll_jobs:
        if str(nll_job["variant"]) not in enabled:
            continue
        direction = str(nll_job["direction"])
        if direction not in eval_cache:
            groups = [row for row in read_jsonl(nll_job["eval_file"]) if str(row.get("eval_scope")) == "cross"]
            selected = balanced_sample(groups, count, seed=19000 + len(eval_cache)); eval_path = data_root / f"eval__{direction}.jsonl"; atomic_write_jsonl(eval_path, selected); eval_cache[direction] = eval_path
        source_backend, target_backend = direction.split("-to-", 1); adapter_dir = Path(nll_job["output_dir"]) / "adapter"; output_dir = root / "interactive_outputs" / profile / direction / str(nll_job["variant"]) / f"seed-{int(nll_job['seed']):03d}"
        payload = {"schema": SCHEMA, "suite_id": cfg["suite_id"], "experiment_id": "EXP-019", "profile": profile, "direction": direction, "source_backend": source_backend, "target_backend": target_backend, "variant": str(nll_job["variant"]), "seed": int(nll_job["seed"]), "base_model": str(nll_job["base_model"]), "adapter_dir": str(adapter_dir.resolve()), "eval_file": str(eval_cache[direction].resolve()), "eval_file_sha256": file_sha256(eval_cache[direction]), "output_dir": str(output_dir.resolve()), "retrieval_url": str(cfg["interactive"][f"{target_backend}_url"]), "max_new_tokens": int(cfg["interactive"]["generation_max_new_tokens"]), "temperature": float(cfg["interactive"]["generation_temperature"]), "runner_module": "stackpilot.query_attribution_interactive_job", "parent_job_id": str(nll_job["job_id"]), "parent_job_signature": str(nll_job["job_signature"])}
        payload["job_id"] = f"EXP-019__seed-{int(nll_job['seed']):03d}__profile-{profile}__variant-{direction}__{nll_job['variant']}"; payload["job_signature"] = signature(payload); jobs.append(payload)
    spec_root = output_root / "job_specs"; spec_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        path = spec_root / f"{job['job_id']}.json"; job["job_file"] = str(path.resolve()); atomic_write_json(path, job)
    jobs_path = output_root / "jobs.jsonl"; atomic_write_jsonl(jobs_path, jobs)
    manifest = {"schema": SCHEMA, "experiment_id": "EXP-019", "profile": profile, "jobs": len(jobs), "jobs_path": str(jobs_path), "jobs_sha256": file_sha256(jobs_path), "config_path": str(config_path.resolve()), "config_sha256": file_sha256(config_path), "nll_manifest_path": str(nll_manifest_path.resolve()), "nll_manifest_sha256": file_sha256(nll_manifest_path)}; manifest["signature"] = signature(manifest); atomic_write_json(output_root / "manifest.json", manifest); return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan EXP-019 interactive query retrieval jobs."); parser.add_argument("--config", default="configs/query_attribution.yaml"); parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot"); args = parser.parse_args(); print(json.dumps(plan(Path(args.config).resolve(), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

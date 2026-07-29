from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.trace_common import (
    TRACE_SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    deterministic_sample,
    file_sha256,
    load_trace_config,
    read_jsonl,
    stable_hash,
)
from stackpilot.trace_curricula import (
    curriculum_token_summary,
    match_by_marginals,
    paired_portable_pool,
    portable_quantile_cells,
    recovered_vs_deep_pools,
    representative_transitions,
    unpaired_global_pool,
)


def _profile(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Unknown TRACE profile {name!r}; choose from {sorted(profiles)}")
    return dict(profiles[name])


def _write_examples(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    negative_query_weight: float = 0.25,
) -> None:
    payload = []
    for row in rows:
        evidence_gain = float(row.get("evidence_gain", 0.0))
        weight = evidence_gain if evidence_gain > 0.0 else -abs(negative_query_weight)
        payload.append(
            {
                "example_id": str(row["transition_id"]),
                "question_id": str(row["question_id"]),
                "dataset": str(row["dataset"]),
                "backend": str(row["backend"]),
                "topk": int(row["topk"]),
                "prompt": str(row["prompt"]),
                "target": str(row["target"]),
                "weight": float(weight),
            }
        )
    atomic_write_jsonl(path, payload)


def _heldout_eval(
    transitions: list[dict[str, Any]],
    *,
    backend: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = [
        row
        for row in transitions
        if row["split"] == "heldout"
        and row["backend"] == backend
        and int(row["positive_gain"]) == 1
    ]
    # One evidence-gaining reformulation per question makes all job evaluations
    # question-paired and prevents a long trajectory from dominating the probe.
    best: dict[str, dict[str, Any]] = {}
    for row in pool:
        question_id = str(row["question_id"])
        current = best.get(question_id)
        score = (float(row["evidence_gain"]), -int(row["source_turn"]))
        if current is None or score > (
            float(current["evidence_gain"]),
            -int(current["source_turn"]),
        ):
            best[question_id] = row
    candidates = list(best.values())
    if len(candidates) < count:
        raise RuntimeError(
            f"Held-out backend {backend} has only {len(candidates)} positive "
            f"reformulations; profile requests {count}"
        )
    return deterministic_sample(candidates, count, seed=seed)


def _job(
    *,
    cfg: dict[str, Any],
    profile_name: str,
    experiment_id: str,
    variant: str,
    seed: int,
    train_file: Path,
    eval_file: Path,
    output_dir: Path,
    max_steps: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    lora = dict(cfg["lora"])
    model = dict(cfg["model"])
    signature_payload = {
        "schema": TRACE_SCHEMA,
        "experiment_id": experiment_id,
        "variant": variant,
        "seed": seed,
        "profile": profile_name,
        "train_sha256": file_sha256(train_file),
        "eval_sha256": file_sha256(eval_file),
        "model": model,
        "lora": lora,
        "max_steps": max_steps,
        "metadata": metadata,
    }
    return {
        **signature_payload,
        "job_id": f"{experiment_id}__seed-{seed:03d}__profile-{profile_name}__variant-{variant}",
        "job_signature": canonical_signature(signature_payload),
        "base_model": model["base_model"],
        "max_length": int(model["max_length"]),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "output_dir": str(output_dir),
    }


def _metadata_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = [
        "portable_recovery_proxy",
        "crs",
        "turn1_recall",
        "reward_variance",
        "search_count",
        "question_difficulty",
        "total_recovery",
    ]
    result = {}
    for name in names:
        values = [float(row[name]) for row in rows]
        result[name] = float(np.mean(values)) if values else 0.0
    result.update(curriculum_token_summary(rows))
    return result


def plan_a(
    cfg: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
    transitions: list[dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    experiment_id = "EXP-009"
    jobs: list[dict[str, Any]] = []
    data_root = root / experiment_id / "data"
    job_root = root / experiment_id / "jobs"
    output_root = root / experiment_id / "outputs"
    cells_per_direction = int(profile["a_cells_per_direction"])
    examples_per_cell = int(profile["a_examples_per_cell"])
    eval_count = int(profile["eval_examples_per_direction"])
    max_steps = int(profile["a_steps"])
    seeds = [int(value) for value in profile["seeds"]]
    negative_weight = float(cfg["diagnostic"]["negative_query_weight"])

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        source_rows = representative_transitions(
            [
                row
                for row in transitions
                if row["split"] == "calibration"
                and row["backend"] == source_backend
                and int(row["paired_backend_count"])
                >= int(cfg["views"]["minimum_paired_views"])
            ]
        )
        if len(source_rows) < examples_per_cell:
            raise RuntimeError(
                f"Condition A {source_backend}->{target_backend} has only "
                f"{len(source_rows)} calibration transitions"
            )
        cells = portable_quantile_cells(
            source_rows,
            cells=cells_per_direction,
            examples_per_cell=examples_per_cell,
            seed=9010 + direction_index,
        )
        eval_rows = _heldout_eval(
            transitions,
            backend=target_backend,
            count=eval_count,
            seed=9100 + direction_index,
        )
        eval_file = data_root / f"eval__{source_backend}-to-{target_backend}.jsonl"
        _write_examples(eval_file, eval_rows, negative_query_weight=negative_weight)

        for cell_index, cell in enumerate(cells):
            train_file = data_root / (
                f"train__{source_backend}-to-{target_backend}__cell-{cell_index:02d}.jsonl"
            )
            _write_examples(train_file, cell, negative_query_weight=negative_weight)
            cell_metadata = {
                "condition": "A",
                "source_backend": source_backend,
                "target_backend": target_backend,
                "cell_index": cell_index,
                **_metadata_mean(cell),
            }
            for seed in seeds:
                variant = f"{source_backend}-to-{target_backend}__cell-{cell_index:02d}"
                output_dir = output_root / variant / f"seed-{seed:03d}"
                jobs.append(
                    _job(
                        cfg=cfg,
                        profile_name=profile_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        seed=seed,
                        train_file=train_file,
                        eval_file=eval_file,
                        output_dir=output_dir,
                        max_steps=max_steps,
                        metadata=cell_metadata,
                    )
                )
    job_root.mkdir(parents=True, exist_ok=True)
    return jobs


def plan_b(
    cfg: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
    transitions: list[dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    experiment_id = "EXP-010"
    jobs: list[dict[str, Any]] = []
    data_root = root / experiment_id / "data"
    output_root = root / experiment_id / "outputs"
    count = int(profile["bc_examples_per_variant"])
    eval_count = int(profile["eval_examples_per_direction"])
    max_steps = int(profile["bc_steps"])
    seeds = [int(value) for value in profile["seeds"]]
    negative_weight = float(cfg["diagnostic"]["negative_query_weight"])
    b_cfg = cfg["condition_b"]
    epsilon = float(cfg["recovery"]["epsilon"])

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        recovered, deep = recovered_vs_deep_pools(
            transitions,
            source_backend=source_backend,
            short_turn=int(b_cfg["short_recovery_max_turn"]),
            deep_turn=int(b_cfg["deep_unrecovered_min_turn"]),
            recovery_epsilon=epsilon,
        )
        matched_recovered, matched_deep = match_by_marginals(
            recovered,
            deep,
            count=count,
            seed=10010 + direction_index,
            group_keys=("dataset", "backend", "topk"),
        )
        eval_rows = _heldout_eval(
            transitions,
            backend=target_backend,
            count=eval_count,
            seed=10100 + direction_index,
        )
        eval_file = data_root / f"eval__{source_backend}-to-{target_backend}.jsonl"
        _write_examples(eval_file, eval_rows, negative_query_weight=negative_weight)
        curricula = {
            "short-recovered": matched_recovered,
            "deep-unrecovered": matched_deep,
        }
        for curriculum_name, curriculum_rows in curricula.items():
            train_file = data_root / (
                f"train__{source_backend}-to-{target_backend}__{curriculum_name}.jsonl"
            )
            _write_examples(train_file, curriculum_rows, negative_query_weight=negative_weight)
            metadata = {
                "condition": "B",
                "curriculum": curriculum_name,
                "source_backend": source_backend,
                "target_backend": target_backend,
                **_metadata_mean(curriculum_rows),
            }
            for seed in seeds:
                variant = f"{source_backend}-to-{target_backend}__{curriculum_name}"
                output_dir = output_root / variant / f"seed-{seed:03d}"
                jobs.append(
                    _job(
                        cfg=cfg,
                        profile_name=profile_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        seed=seed,
                        train_file=train_file,
                        eval_file=eval_file,
                        output_dir=output_dir,
                        max_steps=max_steps,
                        metadata=metadata,
                    )
                )
    return jobs


def plan_c(
    cfg: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
    transitions: list[dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    experiment_id = "EXP-011"
    jobs: list[dict[str, Any]] = []
    data_root = root / experiment_id / "data"
    output_root = root / experiment_id / "outputs"
    count = int(profile["bc_examples_per_variant"])
    eval_count = int(profile["eval_examples_per_direction"])
    max_steps = int(profile["bc_steps"])
    seeds = [int(value) for value in profile["seeds"]]
    negative_weight = float(cfg["diagnostic"]["negative_query_weight"])
    epsilon = float(cfg["recovery"]["epsilon"])

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        paired_pool = paired_portable_pool(
            transitions,
            source_backend=source_backend,
            target_backend=target_backend,
            recovery_epsilon=epsilon,
        )
        excluded = {str(row["question_id"]) for row in paired_pool}
        unpaired_pool = unpaired_global_pool(
            transitions,
            source_backend=source_backend,
            excluded_questions=excluded,
            recovery_epsilon=epsilon,
        )
        matched_paired, matched_unpaired = match_by_marginals(
            paired_pool,
            unpaired_pool,
            count=count,
            seed=11010 + direction_index,
            group_keys=("dataset", "backend", "topk"),
        )
        eval_rows = _heldout_eval(
            transitions,
            backend=target_backend,
            count=eval_count,
            seed=11100 + direction_index,
        )
        eval_file = data_root / f"eval__{source_backend}-to-{target_backend}.jsonl"
        _write_examples(eval_file, eval_rows, negative_query_weight=negative_weight)
        curricula = {
            "paired": matched_paired,
            "unpaired": matched_unpaired,
        }
        for curriculum_name, curriculum_rows in curricula.items():
            train_file = data_root / (
                f"train__{source_backend}-to-{target_backend}__{curriculum_name}.jsonl"
            )
            _write_examples(train_file, curriculum_rows, negative_query_weight=negative_weight)
            metadata = {
                "condition": "C",
                "curriculum": curriculum_name,
                "source_backend": source_backend,
                "target_backend": target_backend,
                **_metadata_mean(curriculum_rows),
            }
            for seed in seeds:
                variant = f"{source_backend}-to-{target_backend}__{curriculum_name}"
                output_dir = output_root / variant / f"seed-{seed:03d}"
                jobs.append(
                    _job(
                        cfg=cfg,
                        profile_name=profile_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        seed=seed,
                        train_file=train_file,
                        eval_file=eval_file,
                        output_dir=output_dir,
                        max_steps=max_steps,
                        metadata=metadata,
                    )
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan TRACE go/no-go LoRA jobs.")
    parser.add_argument("--config", default="configs/trace_go.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--bank-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("EXP-009", "EXP-010", "EXP-011"),
        default=("EXP-009", "EXP-010", "EXP-011"),
    )
    args = parser.parse_args()

    cfg = load_trace_config(args.config)
    if args.base_model:
        cfg["model"]["base_model"] = args.base_model
    profile = _profile(cfg, args.profile)
    work_root = Path(args.output_root or cfg["work_dir"]).resolve()
    bank_root = Path(args.bank_root or work_root / "bank")
    transitions = read_jsonl(bank_root / "transitions.jsonl")
    manifest_path = bank_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing TRACE bank manifest: {manifest_path}")

    plan_root = work_root / "plans" / args.profile
    all_jobs: list[dict[str, Any]] = []
    if "EXP-009" in args.experiments:
        all_jobs.extend(plan_a(cfg, args.profile, profile, transitions, plan_root))
    if "EXP-010" in args.experiments:
        all_jobs.extend(plan_b(cfg, args.profile, profile, transitions, plan_root))
    if "EXP-011" in args.experiments:
        all_jobs.extend(plan_c(cfg, args.profile, profile, transitions, plan_root))

    if not all_jobs:
        raise RuntimeError("TRACE planner produced no jobs")
    spec_root = plan_root / "job_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    for job in all_jobs:
        spec_path = spec_root / f"{job['job_id']}.json"
        job["job_file"] = str(spec_path.resolve())
        atomic_write_json(spec_path, job)
    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, all_jobs)
    plan_manifest = {
        "schema": TRACE_SCHEMA,
        "profile": args.profile,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "bank_manifest_sha256": file_sha256(manifest_path),
        "experiments": list(args.experiments),
        "job_count": len(all_jobs),
        "jobs_sha256": file_sha256(jobs_path),
        "signature": canonical_signature(
            {
                "profile": args.profile,
                "config": cfg,
                "bank_manifest_sha256": file_sha256(manifest_path),
                "job_signatures": [job["job_signature"] for job in all_jobs],
            }
        ),
    }
    atomic_write_json(plan_root / "manifest.json", plan_manifest)
    print(json.dumps(plan_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

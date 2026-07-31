from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from stackpilot.query_equivalence_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    deterministic_order,
    file_sha256,
    load_config,
    read_jsonl,
    signature,
    stable_hash,
)

EXPERIMENT_ID = "EXP-015"
VARIANTS = ("first-exposure", "random-member", "equivalence-pool", "all-direct-pool")


def _balanced_sample(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    buckets: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["source_turn"]),
            str(row["policy_tag"]),
            int(row["policy_seed"]),
        )
        buckets[key].append(dict(row))
    for key in buckets:
        buckets[key] = deterministic_order(buckets[key], seed, key)
    keys = sorted(buckets, key=repr)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while keys and len(selected) < count:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
            cursor += 1
        else:
            keys.remove(key)
            if keys:
                cursor %= len(keys)
    if len(selected) < count:
        sizes = {repr(key): len(value) for key, value in buckets.items()}
        raise RuntimeError(
            f"Only {len(selected)} eligible equivalence states were available; requested {count}. "
            f"Remaining strata={sizes}"
        )
    return selected


def _best_class(state: dict[str, Any]) -> list[dict[str, Any]]:
    ids = set(map(str, state["best_class_ids"]))
    rows = [row for row in state["candidates"] if str(row["candidate_id"]) in ids]
    if len(rows) != len(ids):
        raise RuntimeError(f"State {state['state_id']} best-class IDs are incomplete")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _direct_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in state["candidates"]
            if int(row.get("direct", 0)) == 1 and int(row.get("protocol_failure", 0)) == 0
        ],
        key=lambda row: str(row["candidate_id"]),
    )


def _targets_for_variant(
    state: dict[str, Any],
    *,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    best = _best_class(state)
    if variant == "first-exposure":
        selected = [row for row in best if int(row.get("factual", 0)) == 1]
        if len(selected) != 1:
            raise RuntimeError(
                f"State {state['state_id']} requires one factual best-class member; found {len(selected)}"
            )
    elif variant == "random-member":
        generator = random.Random(int(stable_hash(seed, state["state_id"], length=8), 16))
        selected = [best[generator.randrange(len(best))]]
    elif variant == "equivalence-pool":
        selected = best
    elif variant == "all-direct-pool":
        selected = _direct_candidates(state)
    else:
        raise ValueError(f"Unknown variant {variant}")
    if not selected:
        raise RuntimeError(f"State {state['state_id']} produced no targets for {variant}")
    weight = 1.0 / len(selected)
    return [
        {
            "target_id": str(row["candidate_id"]),
            "text": str(row["query"]),
            "weight": weight,
            "style": str(row["style"]),
            "origin": str(row["origin"]),
            "best_class_member": int(row.get("best_class_member", 0)),
            "direct": int(row.get("direct", 0)),
        }
        for row in selected
    ]


def _training_groups(
    states: Sequence[dict[str, Any]],
    *,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    groups = []
    for state in states:
        targets = _targets_for_variant(state, variant=variant, seed=seed)
        total_weight = sum(float(row["weight"]) for row in targets)
        if abs(total_weight - 1.0) > 1e-8:
            raise RuntimeError(f"State {state['state_id']} target weights sum to {total_weight}")
        groups.append(
            {
                "group_id": str(state["state_id"]),
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "dataset": str(state["dataset"]),
                "backend": str(state["backend"]),
                "prompt": str(state["prompt"]),
                "targets": targets,
                "best_class_size": int(state["best_class_size"]),
                "direct_candidate_count": int(state["direct_candidate_count"]),
            }
        )
    return groups


def _evaluation_groups(states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for state in states:
        targets = []
        for row in state["candidates"]:
            if int(row.get("protocol_failure", 0)) != 0:
                continue
            targets.append(
                {
                    "target_id": str(row["candidate_id"]),
                    "text": str(row["query"]),
                    "weight": 1.0,
                    "style": str(row["style"]),
                    "origin": str(row["origin"]),
                    "best_class_member": int(row.get("best_class_member", 0)),
                    "direct": int(row.get("direct", 0)),
                    "immediate_support_gain": float(row.get("immediate_support_gain", 0.0)),
                    "final_support_recall": float(row.get("final_support_recall", 0.0)),
                }
            )
        output.append(
            {
                "group_id": str(state["state_id"]),
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "dataset": str(state["dataset"]),
                "backend": str(state["backend"]),
                "prompt": str(state["prompt"]),
                "targets": targets,
                "best_class_size": int(state["best_class_size"]),
                "direct_candidate_count": int(state["direct_candidate_count"]),
            }
        )
    return output


def _job(
    *,
    cfg: dict[str, Any],
    profile: str,
    direction: str,
    variant: str,
    seed: int,
    train_file: Path,
    eval_file: Path,
    output_dir: Path,
    max_steps: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "direction": direction,
        "variant": variant,
        "seed": seed,
        "base_model": str(cfg["model"]["base_model"]),
        "max_length": int(cfg["model"]["max_length"]),
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", False)),
        "minimum_parameters": int(cfg["model"]["minimum_parameters"]),
        "maximum_parameters": int(cfg["model"]["maximum_parameters"]),
        "lora": dict(cfg["lora"]),
        "max_steps": max_steps,
        "train_file": str(train_file.resolve()),
        "eval_file": str(eval_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_file_sha256": file_sha256(train_file),
        "eval_file_sha256": file_sha256(eval_file),
        "runner_module": "stackpilot.query_equivalence_lora_job",
        "metadata": metadata,
    }
    payload["job_id"] = (
        f"{EXPERIMENT_ID}__seed-{seed:03d}__profile-{profile}__"
        f"variant-{direction}__{variant}"
    )
    payload["job_signature"] = signature(payload)
    return payload


def plan(
    *,
    config_path: Path,
    profile: str,
    base_model: str | None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if base_model:
        cfg["model"]["base_model"] = base_model
    if profile not in cfg["profiles"]:
        raise KeyError(f"Unknown profile {profile}")
    profile_cfg = cfg["profiles"][profile]
    prepared_root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    manifest_path = prepared_root / "manifest.json"
    eligible_path = prepared_root / "eligible_states.jsonl"
    if not manifest_path.is_file() or not eligible_path.is_file():
        raise RuntimeError("Prepared equivalence states are missing; run query_equivalence/prepare.sh")
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prepared_manifest.get("eligible_sha256") != file_sha256(eligible_path):
        raise RuntimeError("Prepared eligible states do not match their manifest")
    states = read_jsonl(eligible_path)
    plan_root = Path(cfg["work_dir"]).resolve() / "plans" / profile
    data_root = plan_root / "data"
    output_root = plan_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    direction_diagnostics = {}
    seeds = [int(value) for value in profile_cfg["seeds"]]

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        direction = f"{source_backend}-to-{target_backend}"
        train_pool = [
            row for row in states if row["backend"] == source_backend and row["split"] == "train"
        ]
        eval_pool = [
            row for row in states if row["backend"] == target_backend and row["split"] == "heldout"
        ]
        train_states = _balanced_sample(
            train_pool,
            count=int(profile_cfg["train_states_per_direction"]),
            seed=15000 + direction_index,
        )
        eval_states = _balanced_sample(
            eval_pool,
            count=int(profile_cfg["eval_states_per_direction"]),
            seed=15100 + direction_index,
        )
        eval_file = data_root / f"eval__{direction}.jsonl"
        atomic_write_jsonl(eval_file, _evaluation_groups(eval_states))
        direction_diagnostics[direction] = {
            "train_pool": len(train_pool),
            "eval_pool": len(eval_pool),
            "train_states": len(train_states),
            "eval_states": len(eval_states),
            "mean_train_class_size": sum(int(row["best_class_size"]) for row in train_states)
            / len(train_states),
            "mean_eval_class_size": sum(int(row["best_class_size"]) for row in eval_states)
            / len(eval_states),
        }
        for variant in VARIANTS:
            for seed in seeds:
                train_file = data_root / f"train__{direction}__{variant}__seed-{seed:03d}.jsonl"
                atomic_write_jsonl(
                    train_file,
                    _training_groups(train_states, variant=variant, seed=seed),
                )
                output_dir = output_root / direction / variant / f"seed-{seed:03d}"
                metadata = {
                    "source_backend": source_backend,
                    "target_backend": target_backend,
                    "direction": direction,
                    "variant": variant,
                    "train_states": len(train_states),
                    "eval_states": len(eval_states),
                    "state_normalized_credit": True,
                    "prepared_manifest_signature": prepared_manifest.get("signature", ""),
                }
                jobs.append(
                    _job(
                        cfg=cfg,
                        profile=profile,
                        direction=direction,
                        variant=variant,
                        seed=seed,
                        train_file=train_file,
                        eval_file=eval_file,
                        output_dir=output_dir,
                        max_steps=int(profile_cfg["steps"]),
                        metadata=metadata,
                    )
                )

    spec_root = plan_root / "job_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job_file = spec_root / f"{job['job_id']}.json"
        job["job_file"] = str(job_file.resolve())
        atomic_write_json(job_file, job)
    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "job_count": len(jobs),
        "jobs_path": str(jobs_path),
        "jobs_sha256": file_sha256(jobs_path),
        "directions": direction_diagnostics,
        "model": cfg["model"],
    }
    manifest["signature"] = signature(
        {**manifest, "job_signatures": [job["job_signature"] for job in jobs]}
    )
    atomic_write_json(plan_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan EXP-015 equivalence-aware LoRA jobs.")
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()
    payload = plan(
        config_path=Path(args.config).resolve(),
        profile=args.profile,
        base_model=args.base_model,
    )
    print(canonical_json(payload))


if __name__ == "__main__":
    main()

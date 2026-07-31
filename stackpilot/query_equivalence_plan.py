from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stackpilot.query_equivalence_common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    load_equivalence_config,
    read_jsonl,
    stable_hash,
)
from stackpilot.trace_common import file_sha256

EXPERIMENT_ID = "EXP-014"
VARIANTS = ("factual-onehot", "random-onehot", "equivalence-normalized")


def _profile(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    profile = cfg.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise KeyError(f"Unknown profile {name!r}")
    return dict(profile)


def _candidate_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {str(row["candidate_id"]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise RuntimeError("Duplicate candidate IDs in prepared data")
    return output


def _classes_by_state(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["state_id"])].append(dict(row))
    return output


def eligible_training_states(states, classes, *, backend: str) -> list[dict[str, Any]]:
    by_state = _classes_by_state(classes)
    output = []
    for state in states:
        if state.get("split") != "train" or state.get("backend") != backend:
            continue
        if not state.get("factual_direct") or int(state.get("factual_class_size", 0)) < 2:
            continue
        factual_class = next((
            row for row in by_state.get(str(state["state_id"]), [])
            if row["class_id"] == state["factual_class_id"]
        ), None)
        if factual_class is not None and factual_class.get("contains_direct"):
            output.append({**state, "selected_class": factual_class})
    return output


def eligible_eval_classes(states, classes, *, backend: str) -> list[dict[str, Any]]:
    state_by_id = {str(row["state_id"]): row for row in states}
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classes:
        state = state_by_id[str(row["state_id"])]
        if state.get("split") != "heldout" or state.get("backend") != backend:
            continue
        if int(row.get("class_size", 0)) >= 2 and row.get("contains_direct"):
            by_state[str(row["state_id"])].append({**row, "state": state})
    return [
        max(rows, key=lambda row: (
            float(row.get("final_support_recall", 0.0)),
            int(row.get("direct_member_count", 0)), int(row.get("class_size", 0)),
            float(row.get("mean_answer_f1", 0.0)), -float(row.get("mean_search_count", 0.0)),
        ))
        for rows in by_state.values()
    ]


def _balanced_select(rows, count: int, *, seed: int, keys) -> list[dict[str, Any]]:
    if count > len(rows):
        raise RuntimeError(f"Requested {count} rows from pool of {len(rows)}")
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(dict(row))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    active = sorted(buckets, key=repr)
    selected = []
    cursor = 0
    while len(selected) < count and active:
        key = active[cursor % len(active)]
        selected.append(buckets[key].pop())
        if not buckets[key]:
            active.remove(key)
            if active:
                cursor %= len(active)
        else:
            cursor += 1
    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)}/{count} rows")
    return selected


def _example(candidate, *, state_id: str, class_id: str) -> dict[str, Any]:
    return {
        "example_id": str(candidate["candidate_id"]), "state_id": state_id,
        "class_id": class_id, "style": str(candidate["style"]),
        "origin": str(candidate["origin"]), "prompt": str(candidate["prompt"]),
        "target": str(candidate["query"]), "weight": 1.0,
    }


def _training_rows(selected_states, candidate_by_id, *, variant: str, seed: int):
    rows = []
    for state in selected_states:
        class_row = state["selected_class"]
        members = [candidate_by_id[str(value)] for value in class_row["member_candidate_ids"]]
        factual = next((
            row for row in members
            if row.get("origin") == "factual" or row.get("style") == "factual"
        ), None)
        if factual is None:
            raise RuntimeError(f"State {state['state_id']} has no factual class member")
        if variant == "factual-onehot":
            chosen = [factual]
        elif variant == "random-onehot":
            rng = random.Random(int(stable_hash(seed, state["state_id"], length=16), 16))
            chosen = [members[rng.randrange(len(members))]]
        elif variant == "equivalence-normalized":
            chosen = members
        else:
            raise ValueError(variant)
        rows.extend(
            _example(member, state_id=str(state["state_id"]), class_id=str(class_row["class_id"]))
            for member in chosen
        )
    return rows


def _evaluation_rows(selected_classes, candidate_by_id):
    rows = []
    for class_row in selected_classes:
        for candidate_id in class_row["member_candidate_ids"]:
            rows.append(_example(
                candidate_by_id[str(candidate_id)], state_id=str(class_row["state_id"]),
                class_id=str(class_row["class_id"]),
            ))
    return rows


def _job(cfg, *, profile_name, direction, variant, seed, train_file, eval_file, output_dir, max_steps, metadata):
    payload = {
        "schema": 1, "experiment_id": EXPERIMENT_ID, "profile": profile_name,
        "direction": direction, "variant": variant, "seed": seed,
        "base_model": str(cfg["model"]["base_model"]),
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", False)),
        "max_length": int(cfg["model"]["max_length"]), "lora": dict(cfg["lora"]),
        "max_steps": max_steps, "train_file": str(train_file), "eval_file": str(eval_file),
        "output_dir": str(output_dir), "runner_module": "stackpilot.query_equivalence_lora_job",
        "metadata": metadata,
    }
    signature_payload = {**payload, "train_sha256": file_sha256(train_file), "eval_sha256": file_sha256(eval_file)}
    payload["job_id"] = f"{EXPERIMENT_ID}__seed-{seed:03d}__profile-{profile_name}__variant-{direction}__{variant}"
    payload["job_signature"] = canonical_signature(signature_payload)
    return payload


def plan(config_path: str, profile_name: str, base_model: str | None = None) -> dict[str, Any]:
    cfg = load_equivalence_config(config_path)
    if base_model:
        cfg["model"]["base_model"] = base_model
    profile = _profile(cfg, profile_name)
    work_root = Path(cfg["work_dir"]).resolve()
    prepared = work_root / "prepared"
    states = read_jsonl(prepared / "states.jsonl")
    classes = read_jsonl(prepared / "classes.jsonl")
    candidates = read_jsonl(prepared / "candidates.jsonl")
    candidate_by_id = _candidate_index(candidates)
    plan_root = work_root / "plans" / profile_name
    data_root, output_root = plan_root / "data", plan_root / "outputs"
    jobs, diagnostics = [], {}
    count = int(profile["train_states_per_direction"])
    eval_count = int(profile["eval_classes_per_direction"])
    max_steps = int(profile["steps"])
    seeds = [int(value) for value in profile["seeds"]]

    for direction_index, (source, target) in enumerate((("bm25", "e5"), ("e5", "bm25"))):
        direction = f"{source}-to-{target}"
        train_pool = eligible_training_states(states, classes, backend=source)
        eval_pool = eligible_eval_classes(states, classes, backend=target)
        selected_train = _balanced_select(
            train_pool, count, seed=14000 + direction_index,
            keys=("dataset", "topk", "source_turn", "policy_tag", "policy_seed"),
        )
        selected_eval = _balanced_select(
            eval_pool, eval_count, seed=14100 + direction_index,
            keys=("dataset", "topk", "source_turn", "policy_tag", "policy_seed"),
        )
        eval_file = data_root / f"eval__{direction}.jsonl"
        atomic_write_jsonl(eval_file, _evaluation_rows(selected_eval, candidate_by_id))
        diagnostics[direction] = {
            "train_pool": len(train_pool), "eval_pool": len(eval_pool),
            "selected_train_states": len(selected_train),
            "selected_eval_classes": len(selected_eval),
            "mean_train_class_size": float(np.mean([row["selected_class"]["class_size"] for row in selected_train])),
            "mean_eval_class_size": float(np.mean([row["class_size"] for row in selected_eval])),
        }
        for seed in seeds:
            for variant in VARIANTS:
                train_file = data_root / f"train__{direction}__{variant}__seed-{seed:03d}.jsonl"
                train_rows = _training_rows(selected_train, candidate_by_id, variant=variant, seed=seed)
                atomic_write_jsonl(train_file, train_rows)
                output_dir = output_root / direction / variant / f"seed-{seed:03d}"
                jobs.append(_job(
                    cfg, profile_name=profile_name, direction=direction, variant=variant,
                    seed=seed, train_file=train_file, eval_file=eval_file,
                    output_dir=output_dir, max_steps=max_steps,
                    metadata={
                        "source_backend": source, "target_backend": target,
                        "train_states": len(selected_train), "train_examples": len(train_rows),
                        "eval_classes": len(selected_eval), "credit_rule": variant,
                    },
                ))

    specs = plan_root / "job_specs"
    specs.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job_file = specs / f"{job['job_id']}.json"
        job["job_file"] = str(job_file.resolve())
        atomic_write_json(job_file, job)
    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {
        "schema": 1, "experiment_id": EXPERIMENT_ID, "profile": profile_name,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path), "jobs_sha256": file_sha256(jobs_path),
        "job_count": len(jobs), "diagnostics": diagnostics,
    }
    manifest["signature"] = canonical_signature({**manifest, "job_signatures": [job["job_signature"] for job in jobs]})
    atomic_write_json(plan_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan EXP-014 equivalence-aware LoRA jobs.")
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()
    print(json.dumps(plan(args.config, args.profile, args.base_model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

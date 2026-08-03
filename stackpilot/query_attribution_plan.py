from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from stackpilot.query_attribution_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    balanced_sample,
    candidate_valid,
    class_members,
    factual_candidate,
    file_sha256,
    load_config,
    query_jaccard,
    read_jsonl,
    relative_imbalance,
    select_targets,
    signature,
    strict_class_ids,
    target_token_total,
)


def profile_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get("profiles", {}).get(name)
    if not isinstance(value, dict):
        raise KeyError(f"Unknown profile {name!r}")
    return dict(value)


def variant_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for experiment_id, experiment in cfg["experiments"].items():
        for variant, raw in experiment["variants"].items():
            if variant in output:
                raise RuntimeError(f"Duplicate variant name {variant}")
            output[variant] = {**dict(raw), "variant": variant, "experiment_id": experiment_id, "family": str(experiment["name"])}
    return output


def pool_name(spec: dict[str, Any]) -> str:
    if spec["experiment_id"] in {"EXP-016", "EXP-017"}:
        return "attribution"
    return str(spec.get("pool", "definition"))


def pool_variants(specs: dict[str, dict[str, Any]], pool: str) -> list[dict[str, Any]]:
    return [spec for spec in specs.values() if pool_name(spec) == pool]


def state_supports_selector(state: dict[str, Any], selector: str) -> bool:
    try:
        select_targets(state, selector=selector, seed=13, maximum_random_query_jaccard=0.95)
        return True
    except (RuntimeError, ValueError):
        return False


def pool_eligible(state: dict[str, Any], pool: str, specs: dict[str, dict[str, Any]]) -> bool:
    if state.get("split") != "train" or len(class_members(state, "strict")) < 2:
        return False
    if pool == "attribution":
        selectors = {spec["selector"] for spec in pool_variants(specs, pool)}
    elif pool == "immediate":
        selectors = {"strict", "immediate_only"}
    elif pool == "final":
        selectors = {"strict", "final_only"}
    else:
        return False
    return all(state_supports_selector(state, selector) for selector in selectors)


def token_balanced(state: dict[str, Any], *, pool: str, specs: dict[str, dict[str, Any]], seeds: Sequence[int], tokenizer: Any, cfg: dict[str, Any]) -> tuple[bool, dict[str, int], float]:
    totals = {}
    for spec in pool_variants(specs, pool):
        for seed in seeds:
            key = f"{spec['variant']}::seed-{seed}"
            rows = select_targets(state, selector=str(spec["selector"]), seed=seed, maximum_random_query_jaccard=float(cfg["selection"]["maximum_random_query_jaccard"]))
            totals[key] = target_token_total(tokenizer, rows)
    imbalance = relative_imbalance(list(totals.values()))
    return imbalance <= float(cfg["selection"]["maximum_relative_target_token_imbalance"]), totals, imbalance


def target_payload(row: dict[str, Any], weight: float) -> dict[str, Any]:
    return {"target_id": str(row["candidate_id"]), "text": str(row["query"]), "weight": weight, "style": str(row.get("style", "unknown")), "origin": str(row.get("origin", "unknown")), "direct": int(row.get("direct", 0)), "factual": int(row.get("factual", 0)), "immediate_support_set": list(row.get("immediate_support_set", [])), "final_support_set": list(row.get("final_support_set", [])), "answer_em": float(row.get("answer_em", 0.0)), "query_length": len(str(row["query"]).split())}


def training_groups(states: Sequence[dict[str, Any]], *, spec: dict[str, Any], seed: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for state in states:
        selected = select_targets(state, selector=str(spec["selector"]), seed=seed, maximum_random_query_jaccard=float(cfg["selection"]["maximum_random_query_jaccard"]))
        weight = 1.0 / len(selected)
        strict_rows = class_members(state, "strict")
        pair_jaccards = [query_jaccard(str(left["query"]), str(right["query"])) for index, left in enumerate(strict_rows) for right in strict_rows[index + 1 :]]
        output.append({"group_id": str(state["state_id"]), "state_id": str(state["state_id"]), "question_id": str(state["question_id"]), "dataset": str(state["dataset"]), "backend": str(state["backend"]), "prompt": str(state["prompt"]), "targets": [target_payload(row, weight) for row in selected], "class_definition": "immediate" if spec["selector"] == "immediate_only" else "final" if spec["selector"] == "final_only" else "strict", "strict_class_size": len(strict_rows), "strict_min_jaccard": min(pair_jaccards or [1.0])})
    return output


def evaluation_groups(states: Sequence[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
    output = []
    for state in states:
        strict_ids = strict_class_ids(state)
        targets = []
        for row in state["candidates"]:
            if not candidate_valid(row):
                continue
            payload = target_payload(row, 1.0)
            payload.update({"best_class_member": int(str(row["candidate_id"]) in strict_ids), "synthetic": int(str(row.get("origin", "")) != "factual")})
            targets.append(payload)
        if len([row for row in targets if row["best_class_member"]]) < 2:
            continue
        strict_rows = class_members(state, "strict")
        pair_jaccards = [query_jaccard(str(left["query"]), str(right["query"])) for index, left in enumerate(strict_rows) for right in strict_rows[index + 1 :]]
        output.append({"group_id": f"{scope}:{state['state_id']}", "state_id": str(state["state_id"]), "question_id": str(state["question_id"]), "dataset": str(state["dataset"]), "backend": str(state["backend"]), "eval_scope": scope, "prompt": str(state["prompt"]), "targets": targets, "strict_class_size": len(strict_ids), "strict_min_jaccard": min(pair_jaccards or [1.0]), "support_titles": list(state.get("support_titles", [])), "prefix_observed_titles": list(state.get("prefix_observed_titles", [])), "topk": int(state["topk"])})
    return output


def make_job(cfg: dict[str, Any], *, profile: str, spec: dict[str, Any], direction: str, seed: int, train_file: Path, eval_file: Path, output_dir: Path, max_steps: int, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema": SCHEMA, "suite_id": cfg["suite_id"], "experiment_id": str(spec["experiment_id"]), "family": str(spec["family"]), "profile": profile, "direction": direction, "variant": str(spec["variant"]), "seed": seed, "base_model": str(cfg["model"]["base_model"]), "max_length": int(cfg["model"]["max_length"]), "trust_remote_code": bool(cfg["model"].get("trust_remote_code", False)), "minimum_parameters": int(cfg["model"]["minimum_parameters"]), "maximum_parameters": int(cfg["model"]["maximum_parameters"]), "lora": dict(cfg["lora"]), "objective": dict(cfg["objectives"][str(spec["objective"])]), "max_steps": max_steps, "train_file": str(train_file.resolve()), "eval_file": str(eval_file.resolve()), "output_dir": str(output_dir.resolve()), "train_file_sha256": file_sha256(train_file), "eval_file_sha256": file_sha256(eval_file), "runner_module": "stackpilot.query_attribution_lora_job", "metadata": metadata}
    payload["job_id"] = f"{spec['experiment_id']}__seed-{seed:03d}__profile-{profile}__variant-{direction}__{spec['variant']}"
    payload["job_signature"] = signature(payload)
    return payload


def plan(config_path: Path, profile: str, base_model: str | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    if base_model:
        cfg["model"]["base_model"] = base_model
    profile_cfg = profile_config(cfg, profile)
    prepared_root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    manifest_path = prepared_root / "manifest.json"
    states_path = prepared_root / "states.jsonl"
    if not manifest_path.is_file() or not states_path.is_file():
        raise RuntimeError("Prepared attribution states are missing; run query_attribution/prepare.sh")
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prepared_manifest["states_sha256"] != file_sha256(states_path):
        raise RuntimeError("Prepared attribution states do not match their manifest")
    states = read_jsonl(states_path)
    specs = variant_specs(cfg)
    seeds = [int(value) for value in profile_cfg["seeds"]]
    from transformers import AutoTokenizer
    model_ref = str(cfg["model"]["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)), local_files_only=Path(model_ref).is_dir())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    plan_root = Path(cfg["work_dir"]).resolve() / "plans" / profile
    data_root = plan_root / "data"
    output_root = plan_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    diagnostics = {}
    eval_by_backend = {}
    for backend_index, backend in enumerate(("bm25", "e5")):
        eval_pool = [row for row in states if row.get("split") == "heldout" and row["backend"] == backend and len(class_members(row, "strict")) >= int(cfg["selection"]["minimum_eval_class_size"])]
        eval_by_backend[backend] = balanced_sample(eval_pool, int(profile_cfg["eval_states_per_backend"]), seed=16100 + backend_index)
    pool_state_cache = {}
    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        direction = f"{source_backend}-to-{target_backend}"
        eval_rows = []
        if bool(cfg["views"].get("evaluate_seen", True)):
            eval_rows.extend(evaluation_groups(eval_by_backend[source_backend], scope="seen"))
        if bool(cfg["views"].get("evaluate_cross", True)):
            eval_rows.extend(evaluation_groups(eval_by_backend[target_backend], scope="cross"))
        eval_file = data_root / f"eval__{direction}.jsonl"
        atomic_write_jsonl(eval_file, eval_rows)
        for pool in ("attribution", "immediate", "final"):
            pool_specs = pool_variants(specs, pool)
            if not pool_specs:
                continue
            experiment_ids = sorted({str(spec["experiment_id"]) for spec in pool_specs})
            required_count = max(int(profile_cfg["train_states_per_direction"][experiment_id]) for experiment_id in experiment_ids)
            candidates = []
            imbalance_by_state = {}
            for state in states:
                if state.get("split") != "train" or state["backend"] != source_backend or not pool_eligible(state, pool, specs):
                    continue
                okay, _totals, imbalance = token_balanced(state, pool=pool, specs=specs, seeds=seeds, tokenizer=tokenizer, cfg=cfg)
                if okay:
                    candidates.append(state)
                    imbalance_by_state[state["state_id"]] = imbalance
            selected = balanced_sample(candidates, required_count, seed=16200 + direction_index * 10 + (0 if pool == "attribution" else 1 if pool == "immediate" else 2))
            pool_state_cache[(direction, pool)] = selected
            diagnostics[f"{direction}:{pool}"] = {"candidate_states": len(candidates), "selected_states": len(selected), "maximum_token_imbalance": max([imbalance_by_state[row["state_id"]] for row in selected] or [0.0])}
        for variant, spec in specs.items():
            pool = pool_name(spec)
            count = int(profile_cfg["train_states_per_direction"][spec["experiment_id"]])
            selected_states = pool_state_cache[(direction, pool)][:count]
            for seed in seeds:
                train_file = data_root / f"train__{direction}__{variant}__seed-{seed:03d}.jsonl"
                groups = training_groups(selected_states, spec=spec, seed=seed, cfg=cfg)
                atomic_write_jsonl(train_file, groups)
                output_dir = output_root / direction / variant / f"seed-{seed:03d}"
                jobs.append(make_job(cfg, profile=profile, spec=spec, direction=direction, seed=seed, train_file=train_file, eval_file=eval_file, output_dir=output_dir, max_steps=int(profile_cfg["steps"]), metadata={"source_backend": source_backend, "target_backend": target_backend, "pool": pool, "selector": spec["selector"], "objective_name": spec["objective"], "train_states": len(groups), "eval_groups": len(eval_rows), "prepared_manifest_signature": prepared_manifest["signature"]}))
    spec_root = plan_root / "job_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job_file = spec_root / f"{job['job_id']}.json"
        job["job_file"] = str(job_file.resolve())
        atomic_write_json(job_file, job)
    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {"schema": SCHEMA, "suite_id": cfg["suite_id"], "profile": profile, "config_path": str(config_path.resolve()), "config_sha256": file_sha256(config_path), "prepared_manifest_path": str(manifest_path.resolve()), "prepared_manifest_sha256": file_sha256(manifest_path), "jobs_path": str(jobs_path), "jobs_sha256": file_sha256(jobs_path), "job_count": len(jobs), "variant_count": len(specs), "diagnostics": diagnostics, "model": cfg["model"]}
    manifest["signature"] = signature({**manifest, "job_signatures": [job["job_signature"] for job in jobs]})
    atomic_write_json(plan_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan the multi-hypothesis query-attribution matrix.")
    parser.add_argument("--config", default="configs/query_attribution.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()
    print(json.dumps(plan(Path(args.config).resolve(), args.profile, args.base_model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

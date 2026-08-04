from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from stackpilot.multipositive_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    external_query_map,
    file_sha256,
    load_config,
    read_jsonl,
    select_pair,
    signature,
    state_supports,
)
from stackpilot.query_attribution_common import (
    balanced_sample,
    candidate_valid,
    class_members,
    query_jaccard,
    strict_class_ids,
    target_token_total,
)


def core_specs() -> dict[str, dict[str, Any]]:
    return {
        "factual-replicated-uniform": {"experiment_id": "EXP-025", "selector": "factual_replicated", "objective": "uniform", "family": "consistency-attribution"},
        "random-uniform": {"experiment_id": "EXP-025", "selector": "random", "objective": "uniform", "family": "consistency-attribution"},
        "random-consistency": {"experiment_id": "EXP-025", "selector": "random", "objective": "consistency", "family": "consistency-attribution"},
        "diversity-uniform": {"experiment_id": "EXP-025", "selector": "diversity", "objective": "uniform", "family": "consistency-attribution"},
        "diversity-consistency": {"experiment_id": "EXP-025", "selector": "diversity", "objective": "consistency", "family": "consistency-attribution"},
        "all-direct-uniform": {"experiment_id": "EXP-025", "selector": "all_direct", "objective": "uniform", "family": "consistency-attribution"},
        "all-direct-consistency": {"experiment_id": "EXP-025", "selector": "all_direct", "objective": "consistency", "family": "consistency-attribution"},
        "strict-uniform": {"experiment_id": "EXP-025", "selector": "strict", "objective": "uniform", "family": "consistency-attribution"},
        "strict-consistency": {"experiment_id": "EXP-025", "selector": "strict", "objective": "consistency", "family": "consistency-attribution"},
        "all-direct-hardmax": {"experiment_id": "EXP-026", "selector": "all_direct", "objective": "hardmax", "family": "set-objectives"},
        "all-direct-setmass": {"experiment_id": "EXP-026", "selector": "all_direct", "objective": "setmass", "family": "set-objectives"},
        "all-direct-setmass-consistency": {"experiment_id": "EXP-026", "selector": "all_direct", "objective": "setmass_consistency", "family": "set-objectives"},
    }


def style_specs(styles: Sequence[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for style in styles:
        for stem, selector, objective in (
            ("all-direct-uniform", "all_direct", "uniform"),
            ("all-direct-consistency", "all_direct", "consistency"),
            ("strict-consistency", "strict", "consistency"),
        ):
            name = f"{stem}--holdout-{style}"
            output[name] = {
                "experiment_id": "EXP-024",
                "selector": selector,
                "objective": objective,
                "family": "style-heldout",
                "heldout_style": style,
            }
    return output


def variant_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {**core_specs(), **style_specs([str(value) for value in cfg["selection"]["styles"]])}


def target_payload(row: dict[str, Any], *, weight: float) -> dict[str, Any]:
    return {
        "target_id": str(row["candidate_id"]),
        "text": str(row["query"]),
        "weight": float(weight),
        "style": str(row.get("style", "unknown")),
        "origin": str(row.get("origin", "unknown")),
        "direct": int(row.get("direct", 0)),
        "factual": int(row.get("factual", 0) or row.get("origin") == "factual"),
        "immediate_support_gain": float(row.get("immediate_support_gain", 0.0)),
        "final_support_recall": float(row.get("final_support_recall", 0.0)),
        "answer_em": float(row.get("answer_em", 0.0)),
    }


def training_groups(states: Sequence[dict[str, Any]], *, spec: dict[str, Any], seed: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    heldout_style = spec.get("heldout_style")
    for state in states:
        selected = select_pair(
            state,
            selector=str(spec["selector"]),
            seed=seed,
            excluded_style=str(heldout_style) if heldout_style else None,
            maximum_random_query_jaccard=float(cfg["selection"]["maximum_random_query_jaccard"]),
        )
        weight = 1.0 / len(selected)
        strict_rows = class_members(state, "strict")
        pair_jaccards = [query_jaccard(str(left["query"]), str(right["query"])) for index, left in enumerate(strict_rows) for right in strict_rows[index + 1 :]]
        output.append({
            "group_id": str(state["state_id"]),
            "state_id": str(state["state_id"]),
            "question_id": str(state["question_id"]),
            "dataset": str(state["dataset"]),
            "backend": str(state["backend"]),
            "prompt": str(state["prompt"]),
            "targets": [target_payload(row, weight=weight) for row in selected],
            "heldout_style": str(heldout_style or ""),
            "strict_class_size": len(strict_rows),
            "strict_min_jaccard": min(pair_jaccards or [1.0]),
        })
    return output


def evaluation_groups(states: Sequence[dict[str, Any]], *, external: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for state in states:
        strict_ids = strict_class_ids(state)
        targets = []
        for row in state["candidates"]:
            if not candidate_valid(row):
                continue
            payload = target_payload(row, weight=1.0)
            payload.update({"best_class_member": int(str(row["candidate_id"]) in strict_ids), "synthetic": int(str(row.get("origin", "")) != "factual"), "external_generator": 0})
            targets.append(payload)
        external_row = external.get(str(state["state_id"]))
        if external_row:
            targets.append({
                "target_id": f"external::{state['state_id']}",
                "text": str(external_row["query"]),
                "weight": 1.0,
                "style": "external-generator",
                "origin": "external",
                "direct": 0,
                "factual": 0,
                "immediate_support_gain": 0.0,
                "final_support_recall": 0.0,
                "answer_em": 0.0,
                "best_class_member": 0,
                "synthetic": 1,
                "external_generator": 1,
            })
        if len([row for row in targets if int(row["best_class_member"]) == 1]) < 2:
            continue
        output.append({
            "group_id": f"cross:{state['state_id']}",
            "state_id": str(state["state_id"]),
            "question_id": str(state["question_id"]),
            "dataset": str(state["dataset"]),
            "backend": str(state["backend"]),
            "eval_scope": "cross",
            "prompt": str(state["prompt"]),
            "targets": targets,
            "strict_class_size": len(strict_ids),
            "support_titles": list(state.get("support_titles", [])),
            "prefix_observed_titles": list(state.get("prefix_observed_titles", [])),
            "topk": int(state["topk"]),
        })
    return output


def token_balanced(state: dict[str, Any], *, specs: Sequence[dict[str, Any]], seeds: Sequence[int], tokenizer: Any, cfg: dict[str, Any]) -> tuple[bool, float]:
    totals = []
    for spec in specs:
        for seed in seeds:
            selected = select_pair(
                state,
                selector=str(spec["selector"]),
                seed=seed,
                excluded_style=str(spec.get("heldout_style")) if spec.get("heldout_style") else None,
                maximum_random_query_jaccard=float(cfg["selection"]["maximum_random_query_jaccard"]),
            )
            totals.append(target_token_total(tokenizer, selected))
    if not totals or min(totals) <= 0:
        return False, float("inf")
    imbalance = (max(totals) - min(totals)) / min(totals)
    return imbalance <= float(cfg["selection"]["maximum_relative_target_token_imbalance"]), imbalance


def make_job(cfg: dict[str, Any], *, profile: str, variant: str, spec: dict[str, Any], direction: str, seed: int, train_file: Path, eval_file: Path, output_dir: Path, steps: int, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": 2,
        "suite_id": cfg["suite_id"],
        "experiment_id": str(spec["experiment_id"]),
        "family": str(spec["family"]),
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
        "objective": dict(cfg["objectives"][str(spec["objective"])]),
        "max_steps": steps,
        "train_file": str(train_file.resolve()),
        "eval_file": str(eval_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_file_sha256": file_sha256(train_file),
        "eval_file_sha256": file_sha256(eval_file),
        "runner_module": "stackpilot.multipositive_lora_job",
        "metadata": metadata,
    }
    payload["job_id"] = f"{spec['experiment_id']}__seed-{seed:03d}__profile-{profile}__variant-{direction}__{variant}"
    payload["job_signature"] = signature(payload)
    return payload


def plan(config_path: Path, profile: str, base_model: str | None, external_path: str | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    if base_model:
        cfg["model"]["base_model"] = base_model
    profile_cfg = dict(cfg["profiles"][profile])
    prepared_root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    states_path = prepared_root / "states.jsonl"
    manifest_path = prepared_root / "manifest.json"
    if not states_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Prepared states are missing; run multipositive_generalization/prepare.sh")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["states_sha256"] != file_sha256(states_path):
        raise RuntimeError("Prepared states do not match their manifest")
    states = read_jsonl(states_path)
    specs = variant_specs(cfg)
    seeds = [int(value) for value in profile_cfg["seeds"]]
    external = external_query_map(external_path or os.environ.get("MULTIPOSITIVE_EXTERNAL_QUERIES") or None)

    from transformers import AutoTokenizer

    model_ref = str(cfg["model"]["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)), local_files_only=Path(model_ref).is_dir())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    plan_root = Path(cfg["work_dir"]).resolve() / "plans" / profile
    data_root = plan_root / "data"
    output_root = plan_root / "outputs"
    spec_root = plan_root / "job_specs"
    for root in (data_root, output_root, spec_root):
        root.mkdir(parents=True, exist_ok=True)

    eval_by_backend: dict[str, list[dict[str, Any]]] = {}
    for backend_index, backend in enumerate(("bm25", "e5")):
        pool = [row for row in states if row.get("split") == "heldout" and str(row["backend"]) == backend and len(class_members(row, "strict")) >= int(cfg["selection"]["minimum_eval_strict_class_size"])]
        eval_by_backend[backend] = balanced_sample(pool, int(profile_cfg["eval_states_per_backend"]), seed=17100 + backend_index)

    jobs = []
    diagnostics: dict[str, Any] = {}
    core = [dict(value, variant=key) for key, value in core_specs().items()]
    style_families = {style: [dict(spec, variant=name) for name, spec in style_specs([style]).items()] for style in cfg["selection"]["styles"]}

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        direction = f"{source_backend}-to-{target_backend}"
        eval_file = data_root / f"eval__{direction}.jsonl"
        eval_groups = evaluation_groups(eval_by_backend[target_backend], external=external)
        atomic_write_jsonl(eval_file, eval_groups)

        family_states: dict[str, list[dict[str, Any]]] = {}
        families = {"core": core, **{f"style-{style}": values for style, values in style_families.items()}}
        for family_index, (family_name, family_specs) in enumerate(families.items()):
            selectors = sorted({str(spec["selector"]) for spec in family_specs})
            excluded_style = None if family_name == "core" else family_name.removeprefix("style-")
            candidates = []
            imbalance_by_state = {}
            for state in states:
                if state.get("split") != "train" or str(state["backend"]) != source_backend:
                    continue
                if not state_supports(state, selectors=selectors, excluded_style=excluded_style, maximum_random_query_jaccard=float(cfg["selection"]["maximum_random_query_jaccard"])):
                    continue
                okay, imbalance = token_balanced(state, specs=family_specs, seeds=seeds, tokenizer=tokenizer, cfg=cfg)
                if okay:
                    candidates.append(state)
                    imbalance_by_state[str(state["state_id"])] = imbalance
            selected = balanced_sample(candidates, int(profile_cfg["train_states_per_direction"]), seed=17200 + direction_index * 20 + family_index)
            family_states[family_name] = selected
            diagnostics[f"{direction}:{family_name}"] = {
                "candidate_states": len(candidates),
                "selected_states": len(selected),
                "maximum_token_imbalance": max((imbalance_by_state[str(row["state_id"])] for row in selected), default=0.0),
            }

        for variant, spec in specs.items():
            family_key = "core" if spec["family"] != "style-heldout" else f"style-{spec['heldout_style']}"
            selected_states = family_states[family_key]
            for seed in seeds:
                train_file = data_root / f"train__{direction}__{variant}__seed-{seed:03d}.jsonl"
                groups = training_groups(selected_states, spec=spec, seed=seed, cfg=cfg)
                atomic_write_jsonl(train_file, groups)
                output_dir = output_root / direction / variant / f"seed-{seed:03d}"
                job = make_job(cfg, profile=profile, variant=variant, spec=spec, direction=direction, seed=seed, train_file=train_file, eval_file=eval_file, output_dir=output_dir, steps=int(profile_cfg["steps"]), metadata={
                    "source_backend": source_backend,
                    "target_backend": target_backend,
                    "selector": spec["selector"],
                    "objective_name": spec["objective"],
                    "heldout_style": str(spec.get("heldout_style", "")),
                    "train_states": len(groups),
                    "eval_states": len(eval_groups),
                    "external_queries": len(external),
                    "prepared_manifest_signature": manifest["signature"],
                })
                job_file = spec_root / f"{job['job_id']}.json"
                job["job_file"] = str(job_file.resolve())
                atomic_write_json(job_file, job)
                jobs.append(job)

    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    output_manifest = {
        "schema": SCHEMA,
        "suite_id": cfg["suite_id"],
        "profile": profile,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "prepared_manifest_path": str(manifest_path.resolve()),
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "jobs_path": str(jobs_path),
        "jobs_sha256": file_sha256(jobs_path),
        "job_count": len(jobs),
        "variant_count": len(specs),
        "external_query_count": len(external),
        "diagnostics": diagnostics,
        "model": cfg["model"],
    }
    output_manifest["signature"] = signature({**output_manifest, "job_signatures": [job["job_signature"] for job in jobs]})
    atomic_write_json(plan_root / "manifest.json", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan multi-positive generalization jobs.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--external-queries", default=None)
    args = parser.parse_args()
    print(json.dumps(plan(Path(args.config).resolve(), args.profile, args.base_model, args.external_queries), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

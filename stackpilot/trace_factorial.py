from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from stackpilot.trace_common import (
    TRACE_SCHEMA,
    approximate_tokens,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    deterministic_sample,
    file_sha256,
    load_trace_config,
    read_jsonl,
)

EXPERIMENT_ID = "EXP-012"
VARIANTS = (
    "short-recovered",
    "short-unrecovered",
    "deep-recovered",
    "deep-unrecovered",
)


def _profile(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Unknown factorial profile {name!r}; choose from {sorted(profiles)}")
    return dict(profiles[name])


def _one_per_question(
    rows: Sequence[dict[str, Any]], *, recovered: bool
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        current = selected.get(question_id)
        if recovered:
            score = (
                float(row.get("evidence_gain", 0.0)),
                float(row.get("crs", 0.0)),
                -int(row.get("source_turn", 0)),
            )
        else:
            score = (
                int(row.get("source_turn", 0)),
                int(row.get("search_count", 0)),
                -approximate_tokens(str(row.get("target", ""))),
            )
        if current is None:
            selected[question_id] = row
            continue
        if recovered:
            current_score = (
                float(current.get("evidence_gain", 0.0)),
                float(current.get("crs", 0.0)),
                -int(current.get("source_turn", 0)),
            )
        else:
            current_score = (
                int(current.get("source_turn", 0)),
                int(current.get("search_count", 0)),
                -approximate_tokens(str(current.get("target", ""))),
            )
        if score > current_score:
            selected[question_id] = row
    return list(selected.values())


def factorial_pools(
    transitions: Sequence[dict[str, Any]],
    *,
    source_backend: str,
    short_max_turn: int,
    deep_min_turn: int,
    recovery_epsilon: float,
) -> dict[str, list[dict[str, Any]]]:
    if short_max_turn >= deep_min_turn:
        raise ValueError("short_max_turn must be strictly smaller than deep_min_turn")
    source = [
        row
        for row in transitions
        if row.get("split") == "train" and row.get("backend") == source_backend
    ]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_episode[str(row["episode_id"])].append(row)

    pools: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows, key=lambda row: int(row["source_turn"]))
        first = episode_rows[0]
        episode_class = str(first.get("episode_class", ""))
        search_count = int(
            first.get(
                "search_count", max(int(row["source_turn"]) for row in episode_rows)
            )
        )
        total_recovery = float(first.get("total_recovery", 0.0))
        positive = [
            row
            for row in episode_rows
            if float(row.get("evidence_gain", 0.0)) > recovery_epsilon
        ]

        if episode_class == "recoverable" and positive:
            first_positive = min(positive, key=lambda row: int(row["source_turn"]))
            turn = int(first_positive["source_turn"])
            if turn <= short_max_turn:
                pools["short-recovered"].append(first_positive)
            elif turn >= deep_min_turn:
                pools["deep-recovered"].append(first_positive)
            continue

        if episode_class != "unrecoverable" or total_recovery > recovery_epsilon:
            continue
        reformulations = [row for row in episode_rows if int(row["source_turn"]) >= 2]
        if not reformulations:
            continue
        final_transition = max(reformulations, key=lambda row: int(row["source_turn"]))
        final_turn = int(final_transition["source_turn"])
        if search_count <= short_max_turn and final_turn <= short_max_turn:
            pools["short-unrecovered"].append(final_transition)
        elif search_count >= deep_min_turn and final_turn >= deep_min_turn:
            pools["deep-unrecovered"].append(final_transition)

    pools["short-recovered"] = _one_per_question(
        pools["short-recovered"], recovered=True
    )
    pools["deep-recovered"] = _one_per_question(
        pools["deep-recovered"], recovered=True
    )
    pools["short-unrecovered"] = _one_per_question(
        pools["short-unrecovered"], recovered=False
    )
    pools["deep-unrecovered"] = _one_per_question(
        pools["deep-unrecovered"], recovered=False
    )
    return pools


def _enrich(row: dict[str, Any], difficulty_bins: int) -> dict[str, Any]:
    copy = dict(row)
    difficulty = min(
        1.0, max(0.0, float(copy.get("question_difficulty", 0.0)))
    )
    copy["difficulty_bin"] = min(
        difficulty_bins - 1, int(math.floor(difficulty * difficulty_bins))
    )
    copy["approx_tokens"] = approximate_tokens(
        str(copy["prompt"])
    ) + approximate_tokens(str(copy["target"]))
    return copy


def match_factorial_pools(
    pools: dict[str, Sequence[dict[str, Any]]],
    *,
    count: int,
    seed: int,
    group_keys: Sequence[str],
    difficulty_bins: int = 5,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    missing = set(VARIANTS) - set(pools)
    if missing:
        raise ValueError(f"Missing factorial pools: {sorted(missing)}")
    rng = random.Random(seed)
    buckets: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {}
    for variant in VARIANTS:
        variant_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in pools[variant]:
            enriched = _enrich(dict(row), difficulty_bins)
            key = tuple(enriched.get(name) for name in group_keys)
            variant_buckets[key].append(enriched)
        for bucket in variant_buckets.values():
            rng.shuffle(bucket)
        buckets[variant] = variant_buckets

    shared_keys = set(buckets[VARIANTS[0]])
    for variant in VARIANTS[1:]:
        shared_keys &= set(buckets[variant])
    ordered_keys = sorted(shared_keys, key=repr)
    if not ordered_keys:
        sizes = {name: len(pools[name]) for name in VARIANTS}
        raise RuntimeError(f"No shared factorial strata exist; pool sizes={sizes}")

    selected: dict[str, list[dict[str, Any]]] = {
        name: [] for name in VARIANTS
    }
    cursor = 0
    while len(selected[VARIANTS[0]]) < count and ordered_keys:
        key = ordered_keys[cursor % len(ordered_keys)]
        if any(not buckets[variant][key] for variant in VARIANTS):
            ordered_keys.remove(key)
            if ordered_keys:
                cursor %= len(ordered_keys)
            continue

        anchor = buckets["short-recovered"][key].pop()
        quartet = {"short-recovered": anchor}
        target_difficulty = float(anchor.get("question_difficulty", 0.0))
        target_tokens = int(anchor["approx_tokens"])
        for variant in VARIANTS[1:]:
            candidates = buckets[variant][key]
            nearest = min(
                range(len(candidates)),
                key=lambda index: (
                    abs(
                        float(candidates[index].get("question_difficulty", 0.0))
                        - target_difficulty
                    ),
                    abs(int(candidates[index]["approx_tokens"]) - target_tokens),
                    str(candidates[index].get("question_id", "")),
                ),
            )
            quartet[variant] = candidates.pop(nearest)
        for variant in VARIANTS:
            selected[variant].append(quartet[variant])
        cursor += 1

    if len(selected[VARIANTS[0]]) < count:
        sizes = {name: len(pools[name]) for name in VARIANTS}
        matched = len(selected[VARIANTS[0]])
        raise RuntimeError(
            f"Only {matched} complete factorial quartets were available; "
            f"requested {count}; pool sizes={sizes}; "
            f"shared strata={len(shared_keys)}"
        )

    diagnostics = {
        "requested_count": count,
        "matched_count": len(selected[VARIANTS[0]]),
        "pool_sizes": {name: len(pools[name]) for name in VARIANTS},
        "shared_strata": len(shared_keys),
        "group_keys": list(group_keys),
        "mean_question_difficulty": {
            name: float(
                np.mean(
                    [
                        float(row.get("question_difficulty", 0.0))
                        for row in rows
                    ]
                )
            )
            for name, rows in selected.items()
        },
        "mean_approx_tokens": {
            name: float(np.mean([int(row["approx_tokens"]) for row in rows]))
            for name, rows in selected.items()
        },
    }
    return selected, diagnostics


def _heldout_eval(
    transitions: Sequence[dict[str, Any]],
    *,
    backend: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in transitions
        if row.get("split") == "heldout"
        and row.get("backend") == backend
        and float(row.get("evidence_gain", 0.0)) > 0.0
    ]
    best: dict[str, dict[str, Any]] = {}
    for row in candidates:
        question_id = str(row["question_id"])
        current = best.get(question_id)
        score = (float(row["evidence_gain"]), -int(row["source_turn"]))
        if current is None or score > (
            float(current["evidence_gain"]),
            -int(current["source_turn"]),
        ):
            best[question_id] = row
    if len(best) < count:
        raise RuntimeError(
            f"Held-out backend {backend} has only {len(best)} positive "
            f"reformulations; profile requests {count}"
        )
    return deterministic_sample(list(best.values()), count, seed=seed)


def _write_positive_examples(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    payload = []
    for row in rows:
        payload.append(
            {
                "example_id": str(row["transition_id"]),
                "question_id": str(row["question_id"]),
                "dataset": str(row["dataset"]),
                "backend": str(row["backend"]),
                "topk": int(row["topk"]),
                "prompt": str(row["prompt"]),
                "target": str(row["target"]),
                "weight": 1.0,
                "source_turn": int(row["source_turn"]),
                "evidence_gain": float(row.get("evidence_gain", 0.0)),
                "episode_class": str(row.get("episode_class", "")),
            }
        )
    atomic_write_jsonl(path, payload)


def _job(
    *,
    cfg: dict[str, Any],
    profile_name: str,
    direction: str,
    variant: str,
    seed: int,
    train_file: Path,
    eval_file: Path,
    output_dir: Path,
    max_steps: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model = dict(cfg["model"])
    lora = dict(cfg["lora"])
    signature_payload = {
        "schema": TRACE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "direction": direction,
        "variant": variant,
        "seed": seed,
        "profile": profile_name,
        "train_sha256": file_sha256(train_file),
        "eval_sha256": file_sha256(eval_file),
        "model": model,
        "lora": lora,
        "max_steps": max_steps,
        "weight_mode": "positive-only",
        "runner_module": "stackpilot.trace_factorial_lora_job",
        "metadata": metadata,
    }
    return {
        **signature_payload,
        "job_id": (
            f"{EXPERIMENT_ID}__seed-{seed:03d}__profile-{profile_name}__"
            f"variant-{direction}__{variant}"
        ),
        "job_signature": canonical_signature(signature_payload),
        "base_model": model["base_model"],
        "max_length": int(model["max_length"]),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "output_dir": str(output_dir),
    }


def plan(
    cfg: dict[str, Any], profile_name: str, *, base_model: str | None = None
) -> dict[str, Any]:
    if base_model:
        cfg["model"]["base_model"] = base_model
    profile = _profile(cfg, profile_name)
    work_root = Path(cfg["work_dir"]).resolve()
    bank_root = Path(cfg.get("bank_root", "./work/trace_go/bank")).resolve()
    transitions_path = bank_root / "transitions.jsonl"
    manifest_path = bank_root / "manifest.json"
    if not transitions_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Missing TRACE trajectory bank under {bank_root}; "
            "run trace_go/prepare_bank.sh"
        )
    transitions = read_jsonl(transitions_path)
    plan_root = work_root / "plans" / profile_name
    data_root = plan_root / "data"
    output_root = plan_root / "outputs"
    jobs: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    factorial_cfg = cfg["factorial"]
    count = int(profile["examples_per_variant"])
    eval_count = int(profile["eval_examples_per_direction"])
    max_steps = int(profile["steps"])
    seeds = [int(seed) for seed in profile["seeds"]]
    group_keys = tuple(str(value) for value in factorial_cfg["group_keys"])

    for direction_index, pair in enumerate(cfg["views"]["directions"]):
        source_backend, target_backend = map(str, pair)
        direction = f"{source_backend}-to-{target_backend}"
        pools = factorial_pools(
            transitions,
            source_backend=source_backend,
            short_max_turn=int(factorial_cfg["short_max_turn"]),
            deep_min_turn=int(factorial_cfg["deep_min_turn"]),
            recovery_epsilon=float(factorial_cfg["recovery_epsilon"]),
        )
        matched, direction_diagnostics = match_factorial_pools(
            pools,
            count=count,
            seed=120120 + direction_index,
            group_keys=group_keys,
            difficulty_bins=int(factorial_cfg.get("difficulty_bins", 5)),
        )
        diagnostics[direction] = direction_diagnostics
        eval_rows = _heldout_eval(
            transitions,
            backend=target_backend,
            count=eval_count,
            seed=120200 + direction_index,
        )
        eval_file = data_root / f"eval__{direction}.jsonl"
        _write_positive_examples(eval_file, eval_rows)

        for variant in VARIANTS:
            train_file = data_root / f"train__{direction}__{variant}.jsonl"
            _write_positive_examples(train_file, matched[variant])
            metadata = {
                "condition": "positive-factorial",
                "curriculum": variant,
                "source_backend": source_backend,
                "target_backend": target_backend,
                "direction": direction,
                "weight_mode": "positive-only",
                "examples": len(matched[variant]),
                "mean_evidence_gain": float(
                    np.mean(
                        [
                            float(row.get("evidence_gain", 0.0))
                            for row in matched[variant]
                        ]
                    )
                ),
                "mean_search_count": float(
                    np.mean(
                        [
                            float(row.get("search_count", 0.0))
                            for row in matched[variant]
                        ]
                    )
                ),
                "matching": direction_diagnostics,
            }
            for seed in seeds:
                output_dir = (
                    output_root / direction / variant / f"seed-{seed:03d}"
                )
                jobs.append(
                    _job(
                        cfg=cfg,
                        profile_name=profile_name,
                        direction=direction,
                        variant=variant,
                        seed=seed,
                        train_file=train_file,
                        eval_file=eval_file,
                        output_dir=output_dir,
                        max_steps=max_steps,
                        metadata=metadata,
                    )
                )

    spec_root = plan_root / "job_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        spec_path = spec_root / f"{job['job_id']}.json"
        job["job_file"] = str(spec_path.resolve())
        atomic_write_json(spec_path, job)
    jobs_path = plan_root / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    config_path = Path(cfg["_config_path"]).resolve()
    plan_manifest = {
        "schema": TRACE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "bank_manifest_sha256": file_sha256(manifest_path),
        "job_count": len(jobs),
        "jobs_sha256": file_sha256(jobs_path),
        "diagnostics": diagnostics,
    }
    plan_manifest["signature"] = canonical_signature(
        {
            **plan_manifest,
            "job_signatures": [job["job_signature"] for job in jobs],
        }
    )
    atomic_write_json(plan_root / "manifest.json", plan_manifest)
    return plan_manifest


def _require_finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {label}: {value!r}")
    return number


def _load_loss_grid(
    plan_root: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    jobs_path = plan_root / "jobs.jsonl"
    manifest_path = plan_root / "manifest.json"
    if not jobs_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing factorial plan under {plan_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("jobs_sha256") != file_sha256(jobs_path):
        raise RuntimeError("Factorial jobs.jsonl does not match its plan manifest")
    jobs = read_jsonl(jobs_path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in jobs:
        output_dir = Path(job["output_dir"])
        metrics_path = output_dir / "metrics.json"
        losses_path = output_dir / "eval_losses.jsonl"
        if not metrics_path.is_file() or not losses_path.is_file():
            missing.append(job["job_id"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("job_signature") != job.get("job_signature"):
            raise RuntimeError(f"Stale metrics for {job['job_id']}")
        for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
            _require_finite(metrics[name], label=f"{job['job_id']} metrics.{name}")
        for row in read_jsonl(losses_path):
            baseline = _require_finite(
                row["baseline_nll"],
                label=(
                    f"{job['job_id']}:{row.get('example_id')} baseline_nll"
                ),
            )
            adapted = _require_finite(
                row["adapted_nll"],
                label=(
                    f"{job['job_id']}:{row.get('example_id')} adapted_nll"
                ),
            )
            gain = _require_finite(
                row["heldout_gain"],
                label=(
                    f"{job['job_id']}:{row.get('example_id')} heldout_gain"
                ),
            )
            rows.append(
                {
                    "direction": str(job["direction"]),
                    "variant": str(job["variant"]),
                    "seed": int(job["seed"]),
                    "example_id": str(row["example_id"]),
                    "baseline_nll": baseline,
                    "adapted_nll": adapted,
                    "heldout_gain": gain,
                }
            )
    if missing:
        raise RuntimeError(
            f"Factorial analysis is missing {len(missing)} jobs; "
            f"first missing: {missing[:5]}"
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No factorial evaluation losses were found")
    return jobs, frame


def build_factorial_effect_rows(
    frame: pd.DataFrame,
    *,
    baseline_tolerance: float,
) -> pd.DataFrame:
    required = set(VARIANTS)
    output: list[dict[str, Any]] = []
    for (direction, seed, example_id), group in frame.groupby(
        ["direction", "seed", "example_id"], sort=True
    ):
        variants = set(group["variant"].astype(str))
        if variants != required or len(group) != len(required):
            raise RuntimeError(
                f"Incomplete factorial grid for {direction}, seed={seed}, "
                f"example={example_id}: found {sorted(variants)}"
            )
        baselines = group["baseline_nll"].to_numpy(dtype=np.float64)
        baseline_range = float(baselines.max() - baselines.min())
        if baseline_range > baseline_tolerance:
            raise RuntimeError(
                f"Base-model NLL mismatch across variants for {direction}, "
                f"seed={seed}, example={example_id}: range={baseline_range:.6g}"
            )
        values = {
            str(row.variant): float(row.heldout_gain)
            for row in group.itertuples(index=False)
        }
        sr = values["short-recovered"]
        su = values["short-unrecovered"]
        dr = values["deep-recovered"]
        du = values["deep-unrecovered"]
        output.append(
            {
                "direction": str(direction),
                "seed": int(seed),
                "example_id": str(example_id),
                "recovery_effect": 0.5 * ((sr - su) + (dr - du)),
                "depth_effect": 0.5 * ((dr - sr) + (du - su)),
                "interaction": (dr - du) - (sr - su),
                "recovered_mean_gain": 0.5 * (sr + dr),
                "unrecovered_mean_gain": 0.5 * (su + du),
            }
        )
    return pd.DataFrame(output)


def hierarchical_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    samples: int,
    random_seed: int,
) -> dict[str, float]:
    if frame.empty:
        raise RuntimeError(f"No rows available for {value_column}")
    by_seed = {
        int(seed): group[value_column].to_numpy(dtype=np.float64)
        for seed, group in frame.groupby("seed")
    }
    if any(not np.isfinite(values).all() for values in by_seed.values()):
        raise RuntimeError(f"Non-finite values reached bootstrap for {value_column}")
    seeds = sorted(by_seed)
    estimate = float(np.mean([values.mean() for values in by_seed.values()]))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = by_seed[int(seed)]
            sampled = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_rows": float(len(frame)),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value) for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            headers[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |",
        "| "
        + " | ".join("-" * widths[index] for index in range(len(headers)))
        + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def analyze(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile = _profile(cfg, profile_name)
    work_root = Path(cfg["work_dir"]).resolve()
    plan_root = work_root / "plans" / profile_name
    output_dir = work_root / "reports" / profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, loss_frame = _load_loss_grid(plan_root)
    effects = build_factorial_effect_rows(
        loss_frame,
        baseline_tolerance=float(cfg["analysis"]["baseline_tolerance"]),
    )
    effects.to_csv(output_dir / "factorial_effect_rows.csv", index=False)

    cell_seed = loss_frame.groupby(
        ["direction", "variant", "seed"], as_index=False
    )["heldout_gain"].mean()
    cell_summary = (
        cell_seed.groupby(["direction", "variant"])["heldout_gain"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    cell_summary.to_csv(output_dir / "cell_summary.csv", index=False)

    summaries: list[dict[str, Any]] = []
    metrics = (
        "recovery_effect",
        "depth_effect",
        "interaction",
        "recovered_mean_gain",
    )
    for direction_index, (direction, group) in enumerate(
        effects.groupby("direction", sort=True)
    ):
        for metric_index, metric in enumerate(metrics):
            summaries.append(
                {
                    "scope": str(direction),
                    "metric": metric,
                    **hierarchical_mean(
                        group,
                        value_column=metric,
                        samples=int(profile["bootstrap_samples"]),
                        random_seed=(
                            121200 + 10 * direction_index + metric_index
                        ),
                    ),
                }
            )
    for metric_index, metric in enumerate(metrics):
        summaries.append(
            {
                "scope": "combined",
                "metric": metric,
                **hierarchical_mean(
                    effects,
                    value_column=metric,
                    samples=int(profile["bootstrap_samples"]),
                    random_seed=121300 + metric_index,
                ),
            }
        )
    effect_summary = pd.DataFrame(summaries)
    effect_summary.to_csv(output_dir / "factorial_effects.csv", index=False)

    combined_recovery = effect_summary[
        (effect_summary["scope"] == "combined")
        & (effect_summary["metric"] == "recovery_effect")
    ].iloc[0]
    combined_recovered_gain = effect_summary[
        (effect_summary["scope"] == "combined")
        & (effect_summary["metric"] == "recovered_mean_gain")
    ].iloc[0]
    direction_recovery = effect_summary[
        (effect_summary["scope"] != "combined")
        & (effect_summary["metric"] == "recovery_effect")
    ]
    gate_cfg = cfg["gate"]
    decision = bool(
        float(combined_recovery["estimate"])
        >= float(gate_cfg["minimum_recovery_effect"])
        and float(combined_recovery["ci_low"]) > 0.0
        and float(combined_recovered_gain["estimate"])
        > float(gate_cfg.get("minimum_recovered_mean_gain", 0.0))
        and (
            not bool(gate_cfg.get("require_direction_nonnegative", True))
            or bool((direction_recovery["estimate"] >= 0.0).all())
        )
    )
    decision_payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "status": "valid",
        "go": decision,
        "primary_metric": "recovery_effect",
        "minimum_recovery_effect": float(
            gate_cfg["minimum_recovery_effect"]
        ),
        "combined_recovery_effect": {
            key: float(combined_recovery[key])
            for key in ("estimate", "ci_low", "ci_high", "n_seeds", "n_rows")
        },
        "combined_recovered_mean_gain": {
            key: float(combined_recovered_gain[key])
            for key in ("estimate", "ci_low", "ci_high")
        },
    }
    atomic_write_json(output_dir / "decision.json", decision_payload)

    report = [
        "# EXP-012 Positive-only recovery × depth factorial report",
        "",
        f"Profile: `{profile_name}`. Base model: `{jobs[0]['base_model']}`.",
        "",
        "All four curricula use positive imitation weight `+1.0`; no zero-gain query receives negative likelihood weight. The same held-out query grid is evaluated for every cell.",
        "",
        "## Cell means",
        "",
        markdown_table(cell_summary),
        "",
        "## Factorial effects",
        "",
        markdown_table(effect_summary),
        "",
        "The primary contrast is the recovery main effect:",
        "",
        "```text",
        "0.5 * [(short recovered - short unrecovered)",
        "     + (deep recovered - deep unrecovered)]",
        "```",
        "",
        f"Decision: **{'GO' if decision else 'NO-GO'}**.",
        "",
        "GO requires a combined recovery effect at or above the configured threshold, a hierarchical-bootstrap lower bound above zero, positive absolute mean gain for recovered curricula, and non-negative point estimates in both transfer directions.",
        "",
        "This remains a query-NLL micro-update diagnostic. A GO must be confirmed with interactive held-out retrieval under matched search-call budgets.",
        "",
    ]
    (output_dir / "EXP012_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return decision_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or analyze the EXP-012 positive-only recovery-by-depth "
            "factorial."
        )
    )
    parser.add_argument("command", choices=("plan", "analyze"))
    parser.add_argument("--config", default="configs/trace_factorial.yaml")
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_trace_config(config_path)
    cfg["_config_path"] = str(config_path)
    if args.command == "plan":
        payload = plan(cfg, args.profile, base_model=args.base_model)
    else:
        payload = analyze(cfg, args.profile)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

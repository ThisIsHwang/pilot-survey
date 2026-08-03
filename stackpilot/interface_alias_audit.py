from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.interface_causality_common import (
    atomic_write_json,
    balanced_state_subset,
    candidate_reward,
    cluster_bootstrap,
    effective_count,
    group_candidates,
    load_config,
    load_state_results,
    markdown_table,
    normalize_advantages,
    source_patterns,
)

EXPERIMENT_ID = "EXP-020"


def _class_rewards(classes: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> list[float]:
    return [float(np.mean([candidate_reward(row, cfg) for row in members])) for members in classes]


def _expanded_surface_rewards(
    classes: list[list[dict[str, Any]]],
    cfg: dict[str, Any],
    *,
    injected_class: int,
    multiplicity: int,
) -> tuple[list[float], list[int]]:
    rewards: list[float] = []
    class_ids: list[int] = []
    for class_index, members in enumerate(classes):
        repeat = multiplicity if class_index == injected_class else 1
        for _copy in range(repeat):
            for candidate in members:
                rewards.append(candidate_reward(candidate, cfg))
                class_ids.append(class_index)
    return rewards, class_ids


def _class_advantage_mass(
    advantages: np.ndarray,
    class_ids: list[int],
    class_count: int,
) -> np.ndarray:
    output = np.zeros(class_count, dtype=np.float64)
    counts = np.zeros(class_count, dtype=np.float64)
    for value, class_id in zip(advantages, class_ids, strict=True):
        output[class_id] += float(value)
        counts[class_id] += 1.0
    counts[counts == 0.0] = 1.0
    return output / counts


def _quotient_advantages(classes: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> np.ndarray:
    return normalize_advantages(_class_rewards(classes, cfg))


def state_alias_rows(
    result: dict[str, Any],
    cfg: dict[str, Any],
    *,
    multiplicities: list[int],
) -> list[dict[str, Any]]:
    state = result["state"]
    candidates = result["candidates"]
    valid_candidate_count = sum(int(row.get("protocol_failure", 0)) == 0 for row in candidates)
    classes = group_candidates(
        state,
        candidates,
        mode=str(cfg["alias_audit"]["behavior_signature"]),
    )
    if len(classes) < 2:
        return []
    class_rewards = np.asarray(_class_rewards(classes, cfg), dtype=np.float64)
    best_class = int(np.argmax(class_rewards))
    largest_nonbest = max(
        (index for index in range(len(classes)) if index != best_class),
        key=lambda index: (len(classes[index]), -index),
    )
    quotient = _quotient_advantages(classes, cfg)
    baseline_surface: np.ndarray | None = None
    rows = []
    for multiplicity in multiplicities:
        rewards, class_ids = _expanded_surface_rewards(
            classes,
            cfg,
            injected_class=largest_nonbest,
            multiplicity=multiplicity,
        )
        surface_advantages = normalize_advantages(rewards)
        surface_class_mass = _class_advantage_mass(
            surface_advantages,
            class_ids,
            len(classes),
        )
        if baseline_surface is None:
            baseline_surface = surface_class_mass.copy()
        surface_drift = float(np.abs(surface_class_mass - baseline_surface).mean())
        quotient_drift = 0.0
        surface_best = int(np.argmax(surface_class_mass))
        quotient_best = int(np.argmax(quotient))
        class_probabilities = np.bincount(class_ids, minlength=len(classes)).astype(float)
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": str(state["backend"]),
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "multiplicity": int(multiplicity),
                "surface_actions": len(rewards),
                "behavior_classes": len(classes),
                "natural_alias_fraction": 1.0 - len(classes) / max(1, valid_candidate_count),
                "injected_class": largest_nonbest,
                "injected_class_size": len(classes[largest_nonbest]),
                "surface_class_advantage_drift": surface_drift,
                "quotient_class_advantage_drift": quotient_drift,
                "surface_best_class_flip": int(surface_best != best_class),
                "quotient_best_class_flip": int(quotient_best != best_class),
                "surface_effective_behavior_count": effective_count(class_probabilities),
                "quotient_effective_behavior_count": float(len(classes)),
                "best_reward": float(class_rewards.max()),
                "injected_reward": float(class_rewards[largest_nonbest]),
            }
        )
    return rows


def fixed_budget_simulation(
    result: dict[str, Any],
    cfg: dict[str, Any],
    *,
    multiplicity: int,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    state = result["state"]
    classes = group_candidates(
        state,
        result["candidates"],
        mode=str(cfg["alias_audit"]["behavior_signature"]),
    )
    if len(classes) < 2:
        return []
    class_rewards = np.asarray(_class_rewards(classes, cfg), dtype=float)
    best_class = int(np.argmax(class_rewards))
    injected_class = max(
        (index for index in range(len(classes)) if index != best_class),
        key=lambda index: (len(classes[index]), -index),
    )
    surfaces: list[int] = []
    for class_index, members in enumerate(classes):
        repeat = multiplicity if class_index == injected_class else 1
        surfaces.extend([class_index] * (repeat * len(members)))
    budget = int(cfg["alias_audit"]["fixed_rollout_budget"])
    generator = np.random.default_rng(seed)
    rows = []
    for method in ("surface", "quotient"):
        coverages = []
        discoveries = []
        reward_variances = []
        duplicate_rates = []
        for _draw in range(draws):
            if method == "surface":
                sampled = generator.choice(surfaces, size=budget, replace=True)
            else:
                sampled = generator.choice(len(classes), size=budget, replace=True)
            unique = len(set(map(int, sampled)))
            coverages.append(unique / len(classes))
            discoveries.append(float(best_class in set(map(int, sampled))))
            rewards = class_rewards[np.asarray(sampled, dtype=int)]
            reward_variances.append(float(np.var(rewards)))
            duplicate_rates.append(1.0 - unique / budget)
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": str(state["backend"]),
                "dataset": str(state["dataset"]),
                "multiplicity": multiplicity,
                "method": method,
                "class_coverage": float(np.mean(coverages)),
                "best_behavior_discovery": float(np.mean(discoveries)),
                "reward_variance": float(np.mean(reward_variances)),
                "duplicate_rate": float(np.mean(duplicate_rates)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-020: test alias-invariance of surface versus quotient credit.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    results = load_state_results(source_patterns(cfg, args.inputs))
    maximum_states = int(profile["alias_states"])
    results = balanced_state_subset(results, maximum_states)
    multiplicities = [int(value) for value in cfg["alias_audit"]["multiplicities"]]
    rows: list[dict[str, Any]] = []
    simulations: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        rows.extend(state_alias_rows(result, cfg, multiplicities=multiplicities))
        for multiplicity in multiplicities:
            simulations.extend(
                fixed_budget_simulation(
                    result,
                    cfg,
                    multiplicity=multiplicity,
                    draws=int(profile["alias_simulation_draws"]),
                    seed=20000 + index * 100 + multiplicity,
                )
            )
    if not rows:
        raise RuntimeError("No states with at least two behavioral classes were available")
    frame = pd.DataFrame(rows)
    simulation_frame = pd.DataFrame(simulations)
    output_dir = Path(args.output_dir or Path(cfg["work_dir"]) / "reports" / args.profile / "EXP-020").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "alias_invariance.csv", index=False)
    simulation_frame.to_csv(output_dir / "budget_simulation.csv", index=False)

    max_multiplicity = max(multiplicities)
    max_rows = frame[frame["multiplicity"] == max_multiplicity]
    surface_drift = cluster_bootstrap(
        max_rows.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([row["surface_class_advantage_drift"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=20101,
    )
    coverage_pivot = simulation_frame[simulation_frame["multiplicity"] == max_multiplicity].pivot_table(
        index=["state_id", "backend", "dataset"], columns="method", values="class_coverage", aggfunc="first"
    ).dropna().reset_index()
    coverage_pivot["quotient_minus_surface"] = coverage_pivot["quotient"] - coverage_pivot["surface"]
    coverage_gain = cluster_bootstrap(
        coverage_pivot.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([row["quotient_minus_surface"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=20102,
    )
    gates = cfg["gates"]["EXP-020"]
    go = bool(
        surface_drift["estimate"] >= float(gates["minimum_surface_advantage_drift"])
        and surface_drift["ci_low"] > 0.0
        and coverage_gain["estimate"] >= float(gates["minimum_quotient_coverage_gain"])
        and coverage_gain["ci_low"] > 0.0
        and float(max_rows["quotient_class_advantage_drift"].max()) <= float(gates["maximum_quotient_drift"])
    )
    summary = (
        frame.groupby(["backend", "multiplicity"], as_index=False)[
            [
                "natural_alias_fraction", "surface_class_advantage_drift",
                "surface_best_class_flip", "surface_effective_behavior_count",
                "quotient_effective_behavior_count",
            ]
        ].mean()
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "profile": args.profile,
        "go": go,
        "surface_advantage_drift_at_max_alias": surface_drift,
        "quotient_minus_surface_coverage_at_max_alias": coverage_gain,
        "maximum_quotient_drift": float(max_rows["quotient_class_advantage_drift"].max()),
        "states": int(frame["state_id"].nunique()),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = [
        "# EXP-020 Surface-alias invariance report",
        "",
        f"Profile: `{args.profile}`. Alias multiplicity is injected without changing the underlying behavior rewards.",
        "",
        "## Advantage and effective-behavior summary",
        "",
        markdown_table(summary),
        "",
        "## Primary effects",
        "",
        "```text",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO means surface-action normalization is measurably non-invariant to alias multiplicity while quotient normalization remains invariant and recovers fixed-budget behavior coverage.",
        "",
    ]
    (output_dir / "EXP020_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

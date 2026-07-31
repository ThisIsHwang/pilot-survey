from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.behavior_alias_common import (
    build_injected_pool,
    select_queries,
    selected_metrics,
    stable_seed,
)


def load_results(result_root: Path, expected_states: int) -> list[dict[str, Any]]:
    paths = sorted(result_root.glob("*/*.json"))
    if len(paths) != expected_states:
        raise RuntimeError(
            f"Expected {expected_states} state results under {result_root}; found {len(paths)}"
        )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    signatures = {str(row.get("run_signature", "")) for row in rows}
    if len(signatures) != 1 or not next(iter(signatures)):
        raise RuntimeError(f"Inconsistent behavior-alias run signatures: {signatures}")
    return rows


def natural_rows(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "state_id": result["state_id"],
                "backend": result["backend"],
                "dataset": result["dataset"],
                "source_turn": int(result["source_turn"]),
                "eligible_for_injection": int(result["eligible_for_injection"]),
                **dict(result["natural_alias_metrics"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.select_dtypes(include=["number"])).all().all():
        raise RuntimeError("Natural alias metrics are empty or non-finite")
    return frame


def simulate_state(
    state: dict[str, Any],
    *,
    methods: list[str],
    multiplicities: list[int],
    budget: int,
    draws: int,
) -> list[dict[str, Any]]:
    if not int(state["eligible_for_injection"]):
        return []
    output: list[dict[str, Any]] = []
    for multiplicity in multiplicities:
        pool = build_injected_pool(state, multiplicity=multiplicity)
        exact_copy_fraction = float(np.mean([row["is_exact_copy"] for row in pool]))
        for method in methods:
            draw_metrics: list[dict[str, float]] = []
            for draw in range(draws):
                selected = select_queries(
                    pool,
                    method=method,
                    budget=budget,
                    seed=stable_seed(
                        "alias-selection",
                        state["state_id"],
                        multiplicity,
                        method,
                        draw,
                    ),
                )
                draw_metrics.append(selected_metrics(state, selected))
            means = {
                name: float(np.mean([row[name] for row in draw_metrics]))
                for name in draw_metrics[0]
            }
            output.append(
                {
                    "state_id": state["state_id"],
                    "question_id": state["question_id"],
                    "question": state["question"],
                    "backend": state["backend"],
                    "dataset": state["dataset"],
                    "source_turn": int(state["source_turn"]),
                    "method": method,
                    "multiplicity": int(multiplicity),
                    "budget": int(budget),
                    "draws": int(draws),
                    "available_classes": len(state["classes"]),
                    "natural_target_aliases": next(
                        int(row["natural_alias_count"])
                        for row in state["classes"]
                        if str(row["class_id"]) == str(state["injection_class_id"])
                    ),
                    "injected_pool_size": len(pool),
                    "injected_exact_copy_fraction": exact_copy_fraction,
                    **means,
                }
            )
    return output


def _group_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        state_key = str(row.get("_bootstrap_state") or row["state_id"])
        key = (state_key, str(row["method"]), int(row["multiplicity"]))
        if key in lookup:
            raise RuntimeError(f"Duplicate simulation cell: {key}")
        lookup[key] = row
    return lookup


def contrast_stat(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    method_a: str,
    multiplicity_a: int,
    method_b: str,
    multiplicity_b: int,
) -> float:
    lookup = _group_lookup(rows)
    states = sorted({key[0] for key in lookup})
    values = []
    for state in states:
        left = lookup.get((state, method_a, multiplicity_a))
        right = lookup.get((state, method_b, multiplicity_b))
        if left is None or right is None:
            raise RuntimeError(
                f"Incomplete paired grid for {state}: "
                f"{method_a}@{multiplicity_a} vs {method_b}@{multiplicity_b}"
            )
        values.append(float(left[metric]) - float(right[metric]))
    return float(np.mean(values))


def bootstrap_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    method_a: str,
    multiplicity_a: int,
    method_b: str,
    multiplicity_b: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    lookup = _group_lookup(frame.to_dict("records"))
    states = sorted({key[0] for key in lookup})
    differences = []
    for state in states:
        left = lookup.get((state, method_a, multiplicity_a))
        right = lookup.get((state, method_b, multiplicity_b))
        if left is None or right is None:
            raise RuntimeError(
                f"Incomplete paired grid for {state}: "
                f"{method_a}@{multiplicity_a} vs {method_b}@{multiplicity_b}"
            )
        differences.append(float(left[metric]) - float(right[metric]))
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_states": float(len(states)),
        "n_rows": float(len(frame)),
    }


def natural_summary(frame: pd.DataFrame) -> pd.DataFrame:
    scopes = [("combined", frame)] + [
        (str(backend), group) for backend, group in frame.groupby("backend")
    ]
    rows = []
    for scope, group in scopes:
        rows.append(
            {
                "scope": scope,
                "states": len(group),
                "states_with_aliases": int((group["alias_fraction"] > 0).sum()),
                "state_alias_rate": float((group["alias_fraction"] > 0).mean()),
                "mean_alias_fraction": float(group["alias_fraction"].mean()),
                "mean_largest_class_share": float(group["largest_class_share"].mean()),
                "mean_within_class_entropy": float(group["within_class_entropy"].mean()),
                "mean_effective_behavior_count": float(
                    group["effective_behavior_count"].mean()
                ),
                "eligible_injection_rate": float(
                    group["eligible_for_injection"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "class_coverage",
        "unique_classes",
        "effective_behavior_count",
        "duplicate_call_fraction",
        "union_support_gain",
        "best_immediate_gain",
        "best_reward",
        "reward_variance",
    ]
    return (
        frame.groupby(["backend", "method", "multiplicity"], as_index=False)[metrics]
        .mean()
        .sort_values(["backend", "method", "multiplicity"])
    )


def qualitative_examples(
    results: list[dict[str, Any]],
    simulation: pd.DataFrame,
    *,
    maximum_multiplicity: int,
    count: int,
) -> pd.DataFrame:
    quotient = simulation[
        (simulation["method"] == "quotient")
        & (simulation["multiplicity"] == maximum_multiplicity)
    ][["state_id", "union_support_gain", "class_coverage"]].rename(
        columns={
            "union_support_gain": "quotient_union_gain",
            "class_coverage": "quotient_coverage",
        }
    )
    surface = simulation[
        (simulation["method"] == "surface")
        & (simulation["multiplicity"] == maximum_multiplicity)
    ][["state_id", "union_support_gain", "class_coverage"]].rename(
        columns={
            "union_support_gain": "surface_union_gain",
            "class_coverage": "surface_coverage",
        }
    )
    merged = quotient.merge(surface, on="state_id", validate="one_to_one")
    merged["utility_gain"] = merged["quotient_union_gain"] - merged["surface_union_gain"]
    merged["coverage_gain"] = merged["quotient_coverage"] - merged["surface_coverage"]
    by_id = {str(row["state_id"]): row for row in results}
    rows = []
    ranked = merged.sort_values(["utility_gain", "coverage_gain"], ascending=False)
    for row in ranked.head(count).itertuples(index=False):
        state = by_id[str(row.state_id)]
        target = next(
            item
            for item in state["classes"]
            if str(item["class_id"]) == str(state["injection_class_id"])
        )
        rows.append(
            {
                "backend": state["backend"],
                "dataset": state["dataset"],
                "question": state["question"],
                "target_alias_queries": " | ".join(target["queries"][:6]),
                "target_reward": float(target["reward"]),
                "available_classes": len(state["classes"]),
                "surface_coverage": float(row.surface_coverage),
                "quotient_coverage": float(row.quotient_coverage),
                "surface_union_gain": float(row.surface_union_gain),
                "quotient_union_gain": float(row.quotient_union_gain),
            }
        )
    return pd.DataFrame(rows)

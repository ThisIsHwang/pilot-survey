from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    balanced_subset,
    candidate_reward,
    cluster_bootstrap,
    gold_support_set,
    jaccard,
    load_config,
    load_state_results,
    markdown_table,
    ranked_transition,
    source_patterns,
    stable_hash,
    token_set,
)

METHODS = ("surface-random", "text-diverse", "quotient-balanced", "oracle-reward")


def _stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    integer = int(stable_hash(seed, *parts, length=16), 16) % (2**63 - 1)
    return np.random.default_rng(integer)


def _text_diverse_indices(candidates: Sequence[dict[str, Any]], budget: int, seed: int, state_id: str) -> list[int]:
    if budget >= len(candidates):
        return list(range(len(candidates)))
    rng = _stable_rng(seed, state_id, "text-diverse")
    start = int(rng.integers(0, len(candidates)))
    selected = [start]
    remaining = set(range(len(candidates))) - {start}
    query_tokens = [token_set(candidate.get("query", "")) for candidate in candidates]
    while remaining and len(selected) < budget:
        next_index = max(
            remaining,
            key=lambda index: (
                min(1.0 - jaccard(query_tokens[index], query_tokens[chosen]) for chosen in selected),
                stable_hash(seed, state_id, index),
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def _quotient_indices(candidates: Sequence[dict[str, Any]], budget: int, seed: int, state_id: str) -> list[int]:
    if budget >= len(candidates):
        return list(range(len(candidates)))
    classes: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        classes[ranked_transition(candidate)].append(index)
    ordered_classes = sorted(classes, key=lambda value: stable_hash(seed, state_id, value))
    for class_name in ordered_classes:
        classes[class_name].sort(key=lambda index: stable_hash(seed, state_id, class_name, index))
    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for class_name in ordered_classes:
            members = classes[class_name]
            if depth < len(members):
                selected.append(members[depth])
                added = True
                if len(selected) == budget:
                    break
        if not added:
            break
        depth += 1
    return selected


def select_indices(
    candidates: Sequence[dict[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
    state_id: str,
    config: dict[str, Any],
) -> list[int]:
    if not candidates:
        return []
    budget = min(max(1, int(budget)), len(candidates))
    if method == "surface-random":
        rng = _stable_rng(seed, state_id, method)
        return sorted(map(int, rng.choice(len(candidates), size=budget, replace=False)))
    if method == "text-diverse":
        return _text_diverse_indices(candidates, budget, seed, state_id)
    if method == "quotient-balanced":
        return _quotient_indices(candidates, budget, seed, state_id)
    if method == "oracle-reward":
        return sorted(
            range(len(candidates)),
            key=lambda index: (
                -candidate_reward(candidates[index], config),
                stable_hash(seed, state_id, candidates[index].get("candidate_id", index)),
            ),
        )[:budget]
    raise ValueError(f"Unknown selection method: {method}")


def _mean_pairwise_diversity(queries: Sequence[str]) -> float:
    if len(queries) < 2:
        return 0.0
    values = []
    tokens = [token_set(query) for query in queries]
    for left in range(len(tokens)):
        for right in range(left + 1, len(tokens)):
            values.append(1.0 - jaccard(tokens[left], tokens[right]))
    return float(np.mean(values)) if values else 0.0


def state_metrics(
    result: dict[str, Any],
    selected_indices: Sequence[int],
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    state = result["state"]
    candidates = [candidate for candidate in result["candidates"] if int(candidate.get("protocol_failure", 0)) == 0]
    selected = [candidates[index] for index in selected_indices]
    all_classes = {ranked_transition(candidate) for candidate in candidates}
    selected_classes = {ranked_transition(candidate) for candidate in selected}
    gold = {str(value).strip().lower() for value in state.get("support_titles", [])}
    selected_support: set[str] = set()
    for candidate in selected:
        selected_support.update(gold_support_set(candidate, state, final=False))
    union_recall = len(selected_support) / max(1, len(gold))
    rewards = [candidate_reward(candidate, config) for candidate in selected]
    immediate = [float(candidate.get("immediate_support_gain", 0.0)) for candidate in selected]
    final_recalls = [float(candidate.get("final_support_recall", 0.0)) for candidate in selected]
    return {
        "behavior_coverage": len(selected_classes) / max(1, min(len(selected), len(all_classes))),
        "absolute_behavior_coverage": len(selected_classes) / max(1, len(all_classes)),
        "duplicate_rate": 1.0 - len(selected_classes) / max(1, len(selected)),
        "effective_selected_behaviors": float(len(selected_classes)),
        "union_immediate_support_recall": float(union_recall),
        "best_immediate_gain": max(immediate, default=0.0),
        "best_final_support_recall": max(final_recalls, default=0.0),
        "best_reward": max(rewards, default=0.0),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_variance": float(np.var(rewards)) if rewards else 0.0,
        "query_diversity": _mean_pairwise_diversity([str(candidate.get("query", "")) for candidate in selected]),
    }


def run(config: dict[str, Any], profile_name: str, provided_inputs: Sequence[str] | None = None) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    results = balanced_subset(
        load_state_results(source_patterns(config, provided_inputs)),
        int(profile["offline_states"]),
    )
    budgets = [int(value) for value in config["fixed_budget"]["budgets"]]
    draws = int(profile["selection_draws"])
    output_rows: list[dict[str, Any]] = []
    for result in results:
        state = result["state"]
        candidates = [candidate for candidate in result["candidates"] if int(candidate.get("protocol_failure", 0)) == 0]
        if len(candidates) < 2:
            continue
        for budget in budgets:
            actual_budget = min(budget, len(candidates))
            for draw in range(draws):
                for method in METHODS:
                    indices = select_indices(
                        candidates,
                        method=method,
                        budget=actual_budget,
                        seed=draw,
                        state_id=str(state["state_id"]),
                        config=config,
                    )
                    output_rows.append(
                        {
                            "state_id": str(state["state_id"]),
                            "question_id": str(state["question_id"]),
                            "backend": str(state["backend"]),
                            "dataset": str(state["dataset"]),
                            "budget": actual_budget,
                            "draw": draw,
                            "method": method,
                            **state_metrics(result, indices, config=config),
                        }
                    )
    if not output_rows:
        raise RuntimeError("Fixed-budget audit produced no rows")
    frame = pd.DataFrame(output_rows)
    work_root = Path(config["work_dir"]).resolve()
    output_dir = work_root / "reports" / profile_name / "EXP-025"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "selection_rows.csv", index=False)
    summary = (
        frame.groupby(["backend", "budget", "method"], as_index=False)
        .agg(
            behavior_coverage=("behavior_coverage", "mean"),
            duplicate_rate=("duplicate_rate", "mean"),
            union_support_recall=("union_immediate_support_recall", "mean"),
            best_reward=("best_reward", "mean"),
            query_diversity=("query_diversity", "mean"),
        )
    )
    summary.to_csv(output_dir / "selection_summary.csv", index=False)

    maximum_budget = max(frame["budget"])
    paired = frame[frame["budget"] == maximum_budget]
    contrasts: list[dict[str, Any]] = []
    for backend in sorted(paired["backend"].unique()):
        backend_frame = paired[paired["backend"] == backend]
        pivot = backend_frame.pivot_table(
            index=["state_id", "draw"], columns="method", values=["behavior_coverage", "union_immediate_support_recall"]
        )
        for metric in ("behavior_coverage", "union_immediate_support_recall"):
            rows = []
            for index_value, row in pivot[metric].dropna().iterrows():
                rows.append(
                    {
                        "cluster": str(index_value[0]),
                        "difference": float(row["quotient-balanced"] - row["surface-random"]),
                    }
                )
            estimate = cluster_bootstrap(
                rows,
                cluster_key="cluster",
                statistic=lambda values: float(np.mean([item["difference"] for item in values])),
                samples=int(profile["bootstrap_samples"]),
                seed=25025 + len(contrasts),
            )
            contrasts.append({"backend": backend, "metric": metric, **estimate})
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output_dir / "paired_contrasts.csv", index=False)

    gate = config["gates"]["EXP-025"]
    coverage_rows = contrast_frame[contrast_frame["metric"] == "behavior_coverage"]
    recall_rows = contrast_frame[contrast_frame["metric"] == "union_immediate_support_recall"]
    decision = bool(
        len(coverage_rows) >= 2
        and len(recall_rows) >= 2
        and (coverage_rows["estimate"] >= float(gate["minimum_coverage_gain"])).all()
        and (coverage_rows["ci_low"] > 0).all()
        and (recall_rows["estimate"] >= float(gate["minimum_union_recall_gain"])).all()
        and (recall_rows["ci_low"] > 0).all()
    )
    payload = {
        "schema": 1,
        "experiment_id": "EXP-025",
        "profile": profile_name,
        "states": int(frame["state_id"].nunique()),
        "rows": int(len(frame)),
        "go": decision,
        "maximum_budget": int(maximum_budget),
        "contrasts": contrasts,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-025 Fixed-budget behavior selection",
        "",
        f"Profile: `{profile_name}`. Every method sees the same candidate queries and retrieval results; only the K-row selection rule changes.",
        "",
        "## Means",
        "",
        markdown_table(summary),
        "",
        "## Quotient-balanced minus surface-random",
        "",
        markdown_table(contrast_frame),
        "",
        f"Decision: **{'GO' if decision else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP025_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_quotient.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, args.input)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    balanced_subset,
    candidate_reward,
    jaccard,
    load_config,
    load_state_results,
    markdown_table,
    normalize_title,
    full_trajectory_signature,
    ranked_transition,
    source_patterns,
    token_set,
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _partition_from_keys(keys: Sequence[Any]) -> list[int]:
    mapping: dict[Any, int] = {}
    labels = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        labels.append(mapping[key])
    return labels


def _jaccard_partition(candidates: Sequence[dict[str, Any]], threshold: float) -> list[int]:
    uf = UnionFind(len(candidates))
    title_tokens = [
        set().union(
            *(token_set(title) for turn in full_trajectory_signature(candidate) for title in turn)
        )
        for candidate in candidates
    ]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if jaccard(title_tokens[left], title_tokens[right]) >= threshold:
                uf.union(left, right)
    roots = [uf.find(index) for index in range(len(candidates))]
    return _partition_from_keys(roots)


def partition(candidates: Sequence[dict[str, Any]], mode: str) -> list[int]:
    trajectories = [full_trajectory_signature(candidate) for candidate in candidates]
    if mode == "exact-ranked":
        return _partition_from_keys(trajectories)
    if mode == "unordered-set":
        return _partition_from_keys(
            [tuple(tuple(sorted(set(turn))) for turn in trajectory) for trajectory in trajectories]
        )
    if mode == "top1":
        return _partition_from_keys(
            [tuple(tuple(turn[:1]) for turn in trajectory) for trajectory in trajectories]
        )
    if mode == "top3-ranked":
        return _partition_from_keys(
            [tuple(tuple(turn[:3]) for turn in trajectory) for trajectory in trajectories]
        )
    if mode == "title-token-jaccard-050":
        return _jaccard_partition(candidates, 0.50)
    if mode == "title-token-jaccard-075":
        return _jaccard_partition(candidates, 0.75)
    if mode == "title-token-jaccard-090":
        return _jaccard_partition(candidates, 0.90)
    raise ValueError(f"Unknown signature mode: {mode}")


def _pair_metrics(exact: Sequence[int], approximate: Sequence[int]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for left in range(len(exact)):
        for right in range(left + 1, len(exact)):
            exact_same = exact[left] == exact[right]
            approx_same = approximate[left] == approximate[right]
            if exact_same and approx_same:
                tp += 1
            elif not exact_same and approx_same:
                fp += 1
            elif exact_same and not approx_same:
                fn += 1
            else:
                tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "false_merge_rate": fp / max(1, fp + tn),
        "false_split_rate": fn / max(1, fn + tp),
        "pairs": float(tp + fp + fn + tn),
    }


def _class_members(labels: Sequence[int]) -> dict[int, list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        output[int(label)].append(index)
    return output


def state_signature_metrics(
    candidates: Sequence[dict[str, Any]],
    *,
    mode: str,
    config: dict[str, Any],
) -> dict[str, float]:
    exact = partition(candidates, "exact-ranked")
    approximate = partition(candidates, mode)
    metrics = _pair_metrics(exact, approximate)
    exact_classes = _class_members(exact)
    approximate_classes = _class_members(approximate)
    rewards = [candidate_reward(candidate, config) for candidate in candidates]

    exact_best = max(
        exact_classes,
        key=lambda label: float(np.mean([rewards[index] for index in exact_classes[label]])),
    )
    approximate_best_members = max(
        approximate_classes.values(),
        key=lambda members: float(np.mean([rewards[index] for index in members])),
    )
    exact_best_members = set(exact_classes[exact_best])
    metrics.update(
        {
            "exact_class_count": float(len(exact_classes)),
            "approximate_class_count": float(len(approximate_classes)),
            "class_count_ratio": len(approximate_classes) / max(1, len(exact_classes)),
            "best_class_overlap": len(exact_best_members & set(approximate_best_members)) / max(1, len(exact_best_members | set(approximate_best_members))),
            "best_class_hit": float(bool(exact_best_members & set(approximate_best_members))),
            "mean_within_class_reward_variance": float(
                np.mean(
                    [
                        np.var([rewards[index] for index in members])
                        for members in approximate_classes.values()
                    ]
                )
            ),
        }
    )
    return metrics


def run(config: dict[str, Any], profile_name: str, provided_inputs: Sequence[str] | None = None) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    results = balanced_subset(
        load_state_results(source_patterns(config, provided_inputs)),
        int(profile["signature_states"]),
    )
    modes = [str(value) for value in config["signature_audit"]["modes"]]
    rows: list[dict[str, Any]] = []
    for result in results:
        state = result["state"]
        candidates = [candidate for candidate in result["candidates"] if int(candidate.get("protocol_failure", 0)) == 0]
        if len(candidates) < 2 or len(set(full_trajectory_signature(candidate) for candidate in candidates)) < 2:
            continue
        for mode in modes:
            rows.append(
                {
                    "state_id": str(state["state_id"]),
                    "backend": str(state["backend"]),
                    "dataset": str(state["dataset"]),
                    "mode": mode,
                    **state_signature_metrics(candidates, mode=mode, config=config),
                }
            )
    if not rows:
        raise RuntimeError("Signature audit produced no eligible rows")
    frame = pd.DataFrame(rows)
    output_dir = Path(config["work_dir"]).resolve() / "reports" / profile_name / "EXP-026"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "signature_rows.csv", index=False)
    summary = (
        frame.groupby(["backend", "mode"], as_index=False)
        .agg(
            pair_precision=("pair_precision", "mean"),
            pair_recall=("pair_recall", "mean"),
            pair_f1=("pair_f1", "mean"),
            false_merge_rate=("false_merge_rate", "mean"),
            false_split_rate=("false_split_rate", "mean"),
            class_count_ratio=("class_count_ratio", "mean"),
            best_class_hit=("best_class_hit", "mean"),
            reward_variance=("mean_within_class_reward_variance", "mean"),
        )
    )
    summary.to_csv(output_dir / "signature_summary.csv", index=False)
    gate = config["gates"]["EXP-026"]
    eligible_modes = []
    for mode, group in summary.groupby("mode"):
        if (
            (group["pair_precision"] >= float(gate["minimum_precision"])).all()
            and (group["pair_recall"] >= float(gate["minimum_recall"])).all()
            and (group["best_class_hit"] >= float(gate["minimum_best_class_hit"])).all()
        ):
            eligible_modes.append(str(mode))
    payload = {
        "schema": 1,
        "experiment_id": "EXP-026",
        "profile": profile_name,
        "states": int(frame["state_id"].nunique()),
        "go": bool(eligible_modes),
        "eligible_signature_modes": eligible_modes,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-026 Response-signature robustness",
        "",
        "Exact ranked full-trajectory retrieval transitions are the reference. Approximate signatures are judged by pair precision/recall and whether they preserve the highest-reward exact class.",
        "",
        markdown_table(summary),
        "",
        f"Eligible modes: `{eligible_modes}`.",
        "",
        f"Decision: **{'GO' if eligible_modes else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP026_REPORT.md").write_text("\n".join(report), encoding="utf-8")
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

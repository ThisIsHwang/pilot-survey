from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from stackpilot.trace_common import approximate_tokens


def difficulty_bin(value: float, bins: int = 5) -> int:
    clipped = min(1.0, max(0.0, float(value)))
    return min(bins - 1, int(math.floor(clipped * bins)))


def episode_last_transition(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        current = selected.get(episode_id)
        if current is None or int(row["source_turn"]) > int(current["source_turn"]):
            selected[episode_id] = row
    return list(selected.values())


def representative_transitions(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one policy-gradient transition per episode.

    Evidence-gaining turns outrank zero-gain turns; ties prefer the earliest
    reformulation. Recoverable episodes therefore contribute their first most
    useful query, while zero-gain episodes contribute the earliest comparable
    reformulation. Condition B separately selects the final deep failed query.
    """
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        current = selected.get(episode_id)
        score = (float(row["evidence_gain"]), -int(row["source_turn"]))
        if current is None:
            selected[episode_id] = row
            continue
        current_score = (
            float(current["evidence_gain"]),
            -int(current["source_turn"]),
        )
        if score > current_score:
            selected[episode_id] = row
    return list(selected.values())


def match_by_marginals(
    positive: Sequence[dict[str, Any]],
    negative: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    group_keys: Sequence[str] = ("dataset", "backend", "topk"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily form equal-size matched curricula without replacement."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)

    def enriched(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            copy = dict(row)
            copy.setdefault("difficulty_bin", difficulty_bin(copy["question_difficulty"]))
            copy["approx_tokens"] = approximate_tokens(copy["prompt"]) + approximate_tokens(
                copy["target"]
            )
            result.append(copy)
        return result

    left = enriched(positive)
    right = enriched(negative)
    left_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    right_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in left:
        left_buckets[tuple(row[key] for key in group_keys)].append(row)
    for row in right:
        right_buckets[tuple(row[key] for key in group_keys)].append(row)
    for bucket in list(left_buckets.values()) + list(right_buckets.values()):
        rng.shuffle(bucket)

    matched_left: list[dict[str, Any]] = []
    matched_right: list[dict[str, Any]] = []
    shared_keys = sorted(set(left_buckets) & set(right_buckets), key=repr)
    if not shared_keys:
        raise RuntimeError("No shared marginal groups exist between the two curricula")

    # Round-robin prevents a large dataset/view stratum from dominating.
    cursor = 0
    while len(matched_left) < count and shared_keys:
        key = shared_keys[cursor % len(shared_keys)]
        left_bucket = left_buckets[key]
        right_bucket = right_buckets[key]
        if left_bucket and right_bucket:
            candidate_left = left_bucket.pop()
            # Match prompt/target length inside the same scientific stratum.
            target_length = candidate_left["approx_tokens"]
            target_difficulty = float(candidate_left["question_difficulty"])
            nearest = min(
                range(len(right_bucket)),
                key=lambda index: (
                    abs(float(right_bucket[index]["question_difficulty"]) - target_difficulty),
                    abs(right_bucket[index]["approx_tokens"] - target_length),
                ),
            )
            candidate_right = right_bucket.pop(nearest)
            matched_left.append(candidate_left)
            matched_right.append(candidate_right)
        if not left_bucket or not right_bucket:
            shared_keys.remove(key)
            if shared_keys:
                cursor %= len(shared_keys)
        else:
            cursor += 1

    if len(matched_left) < count:
        raise RuntimeError(
            f"Only {len(matched_left)} matched examples were available; requested {count}"
        )
    return matched_left, matched_right


def recovered_vs_deep_pools(
    transitions: Sequence[dict[str, Any]],
    *,
    source_backend: str,
    short_turn: int,
    deep_turn: int,
    recovery_epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = [
        row
        for row in transitions
        if row["split"] == "train" and row["backend"] == source_backend
    ]
    recovered = [
        row
        for row in source
        if row["episode_class"] == "recoverable"
        and int(row["source_turn"]) <= short_turn
        and float(row["evidence_gain"]) > recovery_epsilon
    ]
    deep_candidates = [
        row
        for row in source
        if row["episode_class"] == "unrecoverable"
        and int(row["search_count"]) >= deep_turn
        and float(row["total_recovery"]) <= recovery_epsilon
    ]
    deep = [
        row
        for row in episode_last_transition(deep_candidates)
        if int(row["source_turn"]) >= deep_turn
    ]
    return recovered, deep


def paired_portable_pool(
    transitions: Sequence[dict[str, Any]],
    *,
    source_backend: str,
    target_backend: str,
    recovery_epsilon: float,
) -> list[dict[str, Any]]:
    target_recoverable_questions = {
        str(row["question_id"])
        for row in transitions
        if row["split"] == "train"
        and row["backend"] == target_backend
        and float(row["evidence_gain"]) > recovery_epsilon
        and float(row["crs"]) > 0.0
    }
    source_rows = [
        row
        for row in transitions
        if row["split"] == "train"
        and row["backend"] == source_backend
        and float(row["evidence_gain"]) > recovery_epsilon
        and float(row["crs"]) > 0.0
        and str(row["question_id"]) in target_recoverable_questions
    ]
    best: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        question_id = str(row["question_id"])
        score = (
            float(row["portable_recovery_proxy"]),
            float(row["evidence_gain"]),
            -int(row["source_turn"]),
        )
        current = best.get(question_id)
        if current is None:
            best[question_id] = row
            continue
        current_score = (
            float(current["portable_recovery_proxy"]),
            float(current["evidence_gain"]),
            -int(current["source_turn"]),
        )
        if score > current_score:
            best[question_id] = row
    return list(best.values())


def unpaired_global_pool(
    transitions: Sequence[dict[str, Any]],
    *,
    source_backend: str,
    excluded_questions: set[str],
    recovery_epsilon: float,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in transitions
        if row["split"] == "train"
        and row["backend"] == source_backend
        and float(row["evidence_gain"]) > recovery_epsilon
        and float(row["crs"]) > 0.0
        and str(row["question_id"]) not in excluded_questions
    ]
    # One transition per question avoids silently giving repeated questions more
    # update weight in the unpaired condition.
    best: dict[str, dict[str, Any]] = {}
    for row in candidates:
        question_id = str(row["question_id"])
        current = best.get(question_id)
        if current is None or (
            float(row["crs"]), float(row["evidence_gain"])
        ) > (float(current["crs"]), float(current["evidence_gain"])):
            best[question_id] = row
    return sorted(
        best.values(),
        key=lambda row: (
            -float(row["crs"]),
            -float(row["evidence_gain"]),
            str(row["question_id"]),
        ),
    )


def portable_quantile_cells(
    rows: Sequence[dict[str, Any]],
    *,
    cells: int,
    examples_per_cell: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if cells <= 0 or examples_per_cell <= 0:
        raise ValueError("cells and examples_per_cell must be positive")
    if len(rows) < examples_per_cell:
        raise RuntimeError(
            f"Only {len(rows)} source transitions are available for {examples_per_cell} examples"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["portable_recovery_proxy"]),
            float(row["crs"]),
            str(row["transition_id"]),
        ),
    )
    rng = np.random.default_rng(seed)
    cells_out: list[list[dict[str, Any]]] = []
    # Centers sweep the complete recovery distribution. Local jitter yields
    # overlapping cells whose controls vary continuously rather than forming
    # only three hand-picked bins.
    centers = np.linspace(0.05, 0.95, cells)
    for index, center in enumerate(centers):
        center_index = int(round(center * (len(ordered) - 1)))
        half = max(examples_per_cell, len(ordered) // max(4, cells))
        low = max(0, center_index - half)
        high = min(len(ordered), center_index + half + 1)
        window = ordered[low:high]
        if len(window) < examples_per_cell:
            window = ordered
        chosen = rng.choice(len(window), size=examples_per_cell, replace=False)
        cell = [dict(window[int(position)]) for position in chosen]
        for row in cell:
            row["cell_index"] = index
        cells_out.append(cell)
    return cells_out


def curriculum_token_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    tokens = np.asarray(
        [
            approximate_tokens(str(row["prompt"]))
            + approximate_tokens(str(row["target"]))
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "examples": float(len(rows)),
        "approx_tokens": float(tokens.sum()),
        "mean_tokens": float(tokens.mean()) if len(tokens) else 0.0,
    }

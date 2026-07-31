from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class EquivalenceThresholds:
    support_recall_tolerance: float = 1e-8
    answer_f1_tolerance: float = 0.05
    search_count_tolerance: int = 0
    require_same_support_set: bool = True
    require_same_answer_em: bool = True
    require_same_protocol_status: bool = True


def load_equivalence_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != 1:
        raise ValueError(f"Unsupported query-equivalence config: {config_path}")
    return payload


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def stable_hash(*parts: object, length: int = 24) -> str:
    text = "\n".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def canonical_signature(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object at {path}:{line_number}")
            output.append(value)
    return output


def support_set(state: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, ...]:
    gold = {
        normalize_title(value)
        for value in state.get("support_titles", [])
        if str(value).strip()
    }
    if not gold:
        raise RuntimeError(f"State {state.get('state_id')} has no support titles")
    observed: set[str] = set()
    for turn in state.get("prior_turns", []):
        for title in turn.get("observed_titles", []) or []:
            observed.add(normalize_title(title))
    branch_turns = (
        candidate.get("branch_turns")
        or candidate.get("turns")
        or candidate.get("records")
        or []
    )
    for turn in branch_turns:
        for title in turn.get("observed_titles", []) or []:
            observed.add(normalize_title(title))
    return tuple(sorted(gold & observed))


def final_support_recall(state: dict[str, Any], candidate: dict[str, Any]) -> float:
    if "final_support_recall" in candidate:
        return float(candidate["final_support_recall"])
    found = support_set(state, candidate)
    gold_count = len({normalize_title(x) for x in state.get("support_titles", [])})
    if gold_count <= 0:
        raise RuntimeError(f"State {state.get('state_id')} has no support titles")
    return len(found) / gold_count


def total_search_count(candidate: dict[str, Any]) -> int:
    if "total_search_count" in candidate:
        return int(candidate["total_search_count"])
    turns = candidate.get("branch_turns") or candidate.get("turns") or []
    return len(turns)


def _protocol_status(candidate: dict[str, Any]) -> tuple[int, int]:
    return (
        int(candidate.get("protocol_failure", 0)),
        int(candidate.get("invalid_action_count", 0) > 0),
    )


def candidates_equivalent(
    state: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    thresholds: EquivalenceThresholds,
) -> bool:
    if thresholds.require_same_support_set:
        if support_set(state, left) != support_set(state, right):
            return False
    elif abs(final_support_recall(state, left) - final_support_recall(state, right)) > thresholds.support_recall_tolerance:
        return False
    if thresholds.require_same_answer_em and int(left.get("answer_em", 0)) != int(
        right.get("answer_em", 0)
    ):
        return False
    if abs(float(left.get("answer_f1", 0.0)) - float(right.get("answer_f1", 0.0))) > thresholds.answer_f1_tolerance:
        return False
    if abs(total_search_count(left) - total_search_count(right)) > thresholds.search_count_tolerance:
        return False
    if thresholds.require_same_protocol_status and _protocol_status(left) != _protocol_status(right):
        return False
    return True


def equivalence_classes(
    state: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    thresholds: EquivalenceThresholds,
) -> list[list[int]]:
    """Return connected components of the symmetric equivalence graph."""

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if candidates_equivalent(state, candidates[left], candidates[right], thresholds):
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda row: (-len(row), row))


def equivalence_edges(
    state: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    thresholds: EquivalenceThresholds,
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if not candidates_equivalent(state, candidates[left], candidates[right], thresholds):
                continue
            left_style = str(candidates[left].get("style", candidates[left].get("candidate_id", left)))
            right_style = str(candidates[right].get("style", candidates[right].get("candidate_id", right)))
            edges.add(tuple(sorted((left_style, right_style))))
    return edges


def direct(candidate: dict[str, Any], epsilon: float) -> bool:
    return float(candidate.get("immediate_support_gain", 0.0)) > epsilon


def class_summary(
    state: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    member_indices: Sequence[int],
    *,
    epsilon: float,
) -> dict[str, Any]:
    members = [candidates[index] for index in member_indices]
    member_styles = [str(row.get("style", row.get("candidate_id", ""))) for row in members]
    factual_indices = [
        index
        for index in member_indices
        if str(candidates[index].get("origin", "")) == "factual"
        or str(candidates[index].get("style", "")) == "factual"
    ]
    direct_indices = [index for index in member_indices if direct(candidates[index], epsilon)]
    return {
        "class_id": stable_hash(
            state.get("state_id"),
            *sorted(str(candidates[index].get("candidate_id", index)) for index in member_indices),
        ),
        "member_indices": list(member_indices),
        "member_candidate_ids": [
            str(candidates[index].get("candidate_id", index)) for index in member_indices
        ],
        "member_styles": member_styles,
        "class_size": len(member_indices),
        "factual_member": bool(factual_indices),
        "factual_candidate_id": (
            str(candidates[factual_indices[0]].get("candidate_id", factual_indices[0]))
            if factual_indices
            else ""
        ),
        "direct_member_count": len(direct_indices),
        "contains_direct": bool(direct_indices),
        "final_support_set": list(support_set(state, members[0])),
        "final_support_recall": final_support_recall(state, members[0]),
        "answer_em": int(members[0].get("answer_em", 0)),
        "mean_answer_f1": float(np.mean([float(row.get("answer_f1", 0.0)) for row in members])),
        "mean_search_count": float(np.mean([total_search_count(row) for row in members])),
        "exclusive_credit_overallocation": 1.0 - 1.0 / len(member_indices),
        "expected_onehot_credit_tv": 1.0 - 1.0 / len(member_indices),
    }


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def hash_split(question_id: str, *, seed: int, train_ratio: float) -> str:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    value = int(stable_hash(seed, question_id, length=16), 16) / float(16**16)
    return "train" if value < train_ratio else "heldout"


def hierarchical_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    *,
    positive_variant: str,
    negative_variant: str,
    value_key: str,
    seed_key: str,
    example_key: str,
    samples: int,
    random_seed: int,
) -> dict[str, float]:
    by_key: dict[tuple[Any, Any], dict[str, float]] = {}
    for row in rows:
        value = float(row[value_key])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {value_key}: {value}")
        key = (row[seed_key], row[example_key])
        by_key.setdefault(key, {})[str(row["variant"])] = value
    paired = {
        key: values[positive_variant] - values[negative_variant]
        for key, values in by_key.items()
        if positive_variant in values and negative_variant in values
    }
    if not paired:
        raise RuntimeError(
            f"No paired rows for {positive_variant} versus {negative_variant}"
        )
    seeds = sorted({key[0] for key in paired})
    values_by_seed = {
        seed: np.asarray(
            [value for (row_seed, _), value in paired.items() if row_seed == seed],
            dtype=np.float64,
        )
        for seed in seeds
    }
    estimate = float(np.mean([values.mean() for values in values_by_seed.values()]))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = values_by_seed[int(seed)]
            sampled = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_pairs": float(len(paired)),
    }

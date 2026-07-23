from __future__ import annotations

import math
import re
from collections.abc import Sequence

RESULT_SCHEMA = 3
BASE_POLICY_TAG = "base-qwen"
SPECIALIST_TAGS = ("bm25-specialist", "e5-specialist")
POLICY_TAGS = (BASE_POLICY_TAG, *SPECIALIST_TAGS)
RETRIEVER_BACKENDS = ("bm25", "e5")
DEFAULT_TOPKS = (3, 5, 10)
DEFAULT_DATASETS = ("2wikimultihopqa", "musique")

METRICS = (
    "em",
    "f1",
    "support_recall",
    "turn1_support_recall",
    "turn2_support_recall",
    "turn3_support_recall",
    "turn2_evidence_gain",
    "turn3_evidence_gain",
    "recovery_at_2",
    "recovery_at_3",
    "full_recovery_at_2",
    "full_recovery_at_3",
)


def validate_policy_selection(
    tag: str,
    limit: int | None,
    backends: Sequence[str],
    topks: Sequence[int],
) -> None:
    if re.fullmatch(r"[A-Za-z0-9._-]+", tag) is None:
        raise ValueError(f"Invalid policy tag: {tag!r}")
    if tag not in POLICY_TAGS:
        raise ValueError(
            f"Hard-RQ0 policy tag must be one of {list(POLICY_TAGS)}; got {tag!r}"
        )
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError(f"--limit must be a positive integer; got {limit!r}")
    if not backends:
        raise ValueError("At least one retrieval backend is required")
    if len(set(backends)) != len(backends):
        raise ValueError(f"Duplicate backends are not allowed: {list(backends)}")
    unknown = set(backends) - set(RETRIEVER_BACKENDS)
    if unknown:
        raise ValueError(f"Unknown retrieval backends: {sorted(unknown)}")
    if not topks:
        raise ValueError("At least one top-k value is required")
    if len(set(topks)) != len(topks):
        raise ValueError(f"Duplicate top-k values are not allowed: {list(topks)}")
    invalid = [value for value in topks if isinstance(value, bool) or value < 1]
    if invalid:
        raise ValueError(f"Top-k values must be positive integers; got {invalid}")


def validate_policy_seed(tag: str, seed: int, specialist_seeds: Sequence[int]) -> None:
    if isinstance(seed, bool):
        raise ValueError(f"Policy seed must be an integer; got {seed!r}")
    if tag == BASE_POLICY_TAG:
        if seed != 0:
            raise ValueError(f"{BASE_POLICY_TAG} must use seed 0; got {seed}")
        return
    allowed = list(specialist_seeds)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError(f"Specialist seeds must be nonempty and unique: {allowed}")
    if any(isinstance(value, bool) or value < 1 for value in allowed):
        raise ValueError(f"Specialist seeds must be positive integers: {allowed}")
    if seed not in allowed:
        raise ValueError(
            f"{tag} seed must be one of the configured specialist seeds {allowed}; got {seed}"
        )


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def episode_validation_error(
    row: dict[str, object], max_search_turns: int
) -> str | None:
    if max_search_turns < 1:
        return "max_search_turns must be positive"
    turns = row.get("turns")
    queries = row.get("queries")
    if not isinstance(turns, list) or not isinstance(queries, list):
        return "turns and queries must be lists"
    search_count = row.get("search_count")
    if not finite_number(search_count) or not float(search_count).is_integer():
        return "search_count must be an integer"
    if int(float(search_count)) != len(turns) or len(turns) != len(queries):
        return "search_count, turns, and queries have different lengths"
    if len(turns) > max_search_turns:
        return f"episode exceeds the {max_search_turns}-turn search budget"
    if (
        not isinstance(row.get("answers"), list)
        or not str(row.get("question", "")).strip()
    ):
        return "question or answers are missing"

    recalls: list[float] = []
    gains: list[float] = []
    previous_recall = 0.0
    bounded_features = (
        "query_question_overlap",
        "query_has_quotes",
        "query_capitalized_ratio",
        "query_numeric_ratio",
        "query_lexical_change",
    )
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            return f"turn {index} is not an object"
        if turn.get("turn") != index:
            return f"turn numbers must be sequential from 1; got {turn.get('turn')!r}"
        query = str(turn.get("query", "")).strip()
        if not query or query != str(queries[index - 1]).strip():
            return f"turn {index} query does not match queries"
        if not isinstance(turn.get("retrieved_titles"), list) or not isinstance(
            turn.get("new_support_titles"), list
        ):
            return f"turn {index} retrieval-title fields must be lists"
        recall = turn.get("support_recall")
        gain = turn.get("evidence_gain")
        if not finite_number(recall) or not 0.0 <= float(recall) <= 1.0:
            return f"turn {index} support recall is invalid"
        if not finite_number(gain) or not 0.0 <= float(gain) <= 1.0:
            return f"turn {index} evidence gain is invalid"
        recall_value = float(recall)
        gain_value = float(gain)
        if recall_value + 1e-9 < previous_recall:
            return f"turn {index} support recall decreases"
        if abs(gain_value - (recall_value - previous_recall)) > 1e-9:
            return f"turn {index} evidence gain is inconsistent"
        token_count = turn.get("query_token_count")
        if (
            not finite_number(token_count)
            or float(token_count) < 0
            or not float(token_count).is_integer()
        ):
            return f"turn {index} query_token_count is invalid"
        for feature in bounded_features:
            value = turn.get(feature)
            if not finite_number(value) or not 0.0 <= float(value) <= 1.0:
                return f"turn {index} {feature} is invalid"
        recalls.append(recall_value)
        gains.append(gain_value)
        previous_recall = recall_value

    def recall_at(index: int) -> float:
        if not recalls:
            return 0.0
        return recalls[index] if len(recalls) > index else recalls[-1]

    def gain_at(index: int) -> float:
        return gains[index] if len(gains) > index else 0.0

    turn1 = recall_at(0)
    turn2 = recall_at(1)
    turn3 = recall_at(2)
    first_miss = turn1 < 1.0
    expected = {
        "support_recall": recalls[-1] if recalls else 0.0,
        "turn1_support_recall": turn1,
        "turn2_support_recall": turn2,
        "turn3_support_recall": turn3,
        "turn2_evidence_gain": gain_at(1),
        "turn3_evidence_gain": gain_at(2),
        "recovery_at_2": float(first_miss and turn2 > turn1),
        "recovery_at_3": float(first_miss and turn3 > turn1),
        "full_recovery_at_2": float(first_miss and turn2 >= 1.0),
        "full_recovery_at_3": float(first_miss and turn3 >= 1.0),
    }
    for field, expected_value in expected.items():
        actual = row.get(field)
        if not finite_number(actual) or abs(float(actual) - expected_value) > 1e-9:
            return f"{field} is inconsistent with the recorded turns"
    return None

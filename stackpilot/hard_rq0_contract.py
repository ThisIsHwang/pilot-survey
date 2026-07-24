from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence

from stackpilot.common import answer_em, answer_f1

RESULT_SCHEMA = 4
NUMBERED_EVALUATION_MANIFEST_SCHEMA = 3
BASE_POLICY_TAG = "base-qwen"
SPECIALIST_TAGS = ("bm25-specialist", "e5-specialist")
POLICY_TAGS = (BASE_POLICY_TAG, *SPECIALIST_TAGS)
RETRIEVER_BACKENDS = ("bm25", "e5")
DEFAULT_TOPKS = (3, 5, 10)
DEFAULT_DATASETS = ("2wikimultihopqa", "musique")

METRICS = (
    "em",
    "f1",
    "raw_text_em",
    "raw_text_f1",
    "protocol_failure",
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


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("_", " ").strip()
    text = text.strip("\"'")
    return " ".join(text.lower().split())


def _answer_scores(prediction: str, answers: Sequence[str]) -> tuple[float, float]:
    return (
        max(answer_em(prediction, answer) for answer in answers),
        max(answer_f1(prediction, answer) for answer in answers),
    )


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
    if (
        isinstance(search_count, bool)
        or not finite_number(search_count)
        or not float(search_count).is_integer()
    ):
        return "search_count must be an integer"
    if int(float(search_count)) != len(turns) or len(turns) != len(queries):
        return "search_count, turns, and queries have different lengths"
    if len(turns) > max_search_turns:
        return f"episode exceeds the {max_search_turns}-turn search budget"
    topk = row.get("topk")
    if (
        isinstance(topk, bool)
        or not finite_number(topk)
        or not float(topk).is_integer()
        or float(topk) < 1
    ):
        return "topk must be a positive integer"
    raw_answers = row.get("answers")
    if not isinstance(raw_answers, list) or not str(row.get("question", "")).strip():
        return "question or answers are missing"
    if not raw_answers or any(
        not isinstance(answer, str) or not answer.strip() for answer in raw_answers
    ):
        return "answers must be nonempty strings"
    answers = [answer.strip() for answer in raw_answers]
    raw_support_titles = row.get("support_titles")
    if not isinstance(raw_support_titles, list) or not raw_support_titles or any(
        not isinstance(title, str) or not title.strip()
        for title in raw_support_titles
    ):
        return "support_titles must be nonempty strings"
    support_titles = [title.strip() for title in raw_support_titles]
    gold_titles = {normalize_title(title) for title in support_titles}
    if not isinstance(row.get("prediction"), str) or not isinstance(
        row.get("raw_text_prediction"), str
    ):
        return "prediction fields must be strings"
    prediction = str(row["prediction"])
    raw_text_prediction = str(row["raw_text_prediction"])
    protocol_failure = row.get("protocol_failure")
    if isinstance(protocol_failure, bool) or protocol_failure not in (0, 1, 0.0, 1.0):
        return "protocol_failure must be 0 or 1"
    invalid_action_count = row.get("invalid_action_count")
    if (
        isinstance(invalid_action_count, bool)
        or not finite_number(invalid_action_count)
        or not float(invalid_action_count).is_integer()
        or float(invalid_action_count) < 0
    ):
        return "invalid_action_count is invalid"
    failed = bool(int(float(protocol_failure)))
    counted_actions = int(float(search_count)) + int(float(invalid_action_count))
    counted_action_limit = max_search_turns + int(failed)
    if (
        int(float(invalid_action_count)) > max_search_turns + 1
        or counted_actions > counted_action_limit
    ):
        return "invalid/search action counts exceed the rollout budget"
    if failed and prediction:
        return "protocol failures must have an empty primary prediction"
    if not failed and not prediction.strip():
        return "successful protocol episodes must have a primary prediction"
    for metric in ("em", "f1", "raw_text_em", "raw_text_f1"):
        value = row.get(metric)
        if not finite_number(value) or not 0.0 <= float(value) <= 1.0:
            return f"{metric} is invalid"
    expected_em, expected_f1 = (
        (0.0, 0.0) if failed else _answer_scores(prediction, answers)
    )
    expected_raw_em, expected_raw_f1 = _answer_scores(raw_text_prediction, answers)
    expected_answer_metrics = {
        "em": expected_em,
        "f1": expected_f1,
        "raw_text_em": expected_raw_em,
        "raw_text_f1": expected_raw_f1,
    }
    for metric, expected_value in expected_answer_metrics.items():
        if abs(float(row[metric]) - expected_value) > 1e-9:
            return (
                f"{metric} is inconsistent with its prediction and nonempty answers"
            )

    recalls: list[float] = []
    gains: list[float] = []
    previous_recall = 0.0
    previously_seen_titles: set[str] = set()
    cumulative_titles: set[str] = set()
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
        query_value = turn.get("query")
        recorded_query = queries[index - 1]
        if (
            not isinstance(query_value, str)
            or not isinstance(recorded_query, str)
            or not query_value.strip()
            or query_value.strip() != recorded_query.strip()
        ):
            return f"turn {index} query does not match queries"
        retrieved_titles = turn.get("retrieved_titles")
        new_support_titles = turn.get("new_support_titles")
        if not isinstance(retrieved_titles, list) or not isinstance(
            new_support_titles, list
        ):
            return f"turn {index} retrieval-title fields must be lists"
        if any(
            not isinstance(title, str) or not title.strip()
            for title in retrieved_titles
        ):
            return f"turn {index} retrieved titles must be nonempty strings"
        if len(retrieved_titles) > int(float(topk)):
            return f"turn {index} has more retrieved titles than topk"
        normalized_retrieved = {
            normalize_title(title) for title in retrieved_titles
        }
        expected_new_support = sorted(
            (gold_titles & normalized_retrieved) - previously_seen_titles
        )
        if new_support_titles != expected_new_support:
            return f"turn {index} new support titles are inconsistent"
        cumulative_titles.update(normalized_retrieved)
        expected_recall = len(gold_titles & cumulative_titles) / len(gold_titles)
        recall = turn.get("support_recall")
        gain = turn.get("evidence_gain")
        if not finite_number(recall) or not 0.0 <= float(recall) <= 1.0:
            return f"turn {index} support recall is invalid"
        if not finite_number(gain) or not 0.0 <= float(gain) <= 1.0:
            return f"turn {index} evidence gain is invalid"
        recall_value = float(recall)
        gain_value = float(gain)
        if abs(recall_value - expected_recall) > 1e-9:
            return f"turn {index} support recall is inconsistent with titles"
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
        previously_seen_titles.update(normalized_retrieved)
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

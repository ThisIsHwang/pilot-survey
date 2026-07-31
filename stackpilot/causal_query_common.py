from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
LINE_QUERY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def load_causal_query_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError(f"Unsupported causal-query config: {config_path}")
    return payload


def canonical_signature(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_hash(*parts: object, length: int = 24) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts, length=8), 16) % (2**31)


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def word_tokens(value: str, *, content_only: bool = False) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value))]
    if content_only:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def token_set(value: str, *, content_only: bool = False) -> set[str]:
    return set(word_tokens(value, content_only=content_only))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(value) for value in left}
    right_set = {str(value) for value in right}
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def support_recall(gold_titles: Sequence[str], observed_titles: Iterable[str]) -> float:
    gold = {normalize_title(value) for value in gold_titles if str(value).strip()}
    observed = {normalize_title(value) for value in observed_titles if str(value).strip()}
    if not gold:
        raise ValueError("support_recall requires at least one gold title")
    return len(gold & observed) / len(gold)


def average_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("average_rank expects a one-dimensional sequence")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank = average_rank(left)
    right_rank = average_rank(right)
    if float(left_rank.std()) <= 1e-12 or float(right_rank.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def require_finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}: {value}")


def _clean_query(value: Any) -> str:
    query = str(value or "").strip().strip("`\"'").strip()
    query = re.sub(r"^[-*]\s*", "", query)
    query = re.sub(r"\s+", " ", query)
    if "<" in query or ">" in query or "\n" in query:
        return ""
    return query


def parse_alternative_queries(text: str, styles: Sequence[str]) -> dict[str, str]:
    """Parse JSON or ``STYLE: query`` output from the alternative generator."""

    expected = [str(style).lower() for style in styles]
    candidates: dict[str, str] = {}
    stripped = str(text).strip()
    decoder_candidates = [stripped]
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        decoder_candidates.append(stripped[first_brace : last_brace + 1])
    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        decoder_candidates.append(stripped[first_bracket : last_bracket + 1])

    for candidate_text in decoder_candidates:
        try:
            payload = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            for key, value in payload.items():
                style = str(key).lower().replace("-", "_")
                if style in expected:
                    query = _clean_query(value)
                    if query:
                        candidates[style] = query
        elif isinstance(payload, list):
            for style, value in zip(expected, payload, strict=False):
                query = _clean_query(value)
                if query:
                    candidates[style] = query
        if len(candidates) >= len(expected):
            return {style: candidates[style] for style in expected}

    for line in stripped.splitlines():
        match = LINE_QUERY_RE.match(line)
        if not match:
            continue
        style = match.group(1).lower().replace("-", "_")
        if style not in expected:
            continue
        query = _clean_query(match.group(2))
        if query:
            candidates[style] = query
    return {style: candidates[style] for style in expected if style in candidates}


def validate_alternatives(
    alternatives: dict[str, str],
    *,
    styles: Sequence[str],
    factual_query: str,
    question: str,
    observed_titles: Sequence[str],
    minimum_tokens: int,
    length_ratio_low: float,
    length_ratio_high: float,
) -> dict[str, str]:
    factual_normalized = " ".join(word_tokens(factual_query))
    factual_length = max(1, len(word_tokens(factual_query)))
    known_content = token_set(question, content_only=True)
    known_content.update(token_set(" ".join(observed_titles), content_only=True))
    output: dict[str, str] = {}
    seen = {factual_normalized}
    for style in styles:
        query = _clean_query(alternatives.get(str(style).lower(), ""))
        normalized = " ".join(word_tokens(query))
        token_count = len(word_tokens(query))
        if not query or not normalized or normalized in seen:
            continue
        if token_count < minimum_tokens:
            continue
        ratio = token_count / factual_length
        if ratio < length_ratio_low or ratio > length_ratio_high:
            continue
        query_content = token_set(query, content_only=True)
        if known_content and not (query_content & known_content):
            continue
        seen.add(normalized)
        output[str(style).lower()] = query
    return output


def leave_one_out_effect(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("leave_one_out_effect requires at least two values")
    total = float(array.sum())
    return [float(value - (total - value) / (len(array) - 1)) for value in array]


def attach_query_effects(
    branches: Sequence[dict[str, Any]],
    *,
    answer_weight: float,
    search_cost: float,
    epsilon: float,
    bridge_min_support_tqe: float,
) -> list[dict[str, Any]]:
    if len(branches) < 2:
        raise ValueError("At least two branches are required for causal effects")
    final_recalls = [float(row["final_support_recall"]) for row in branches]
    answer_f1s = [float(row["answer_f1"]) for row in branches]
    direct_gains = [float(row["immediate_support_gain"]) for row in branches]
    downstream_gains = [
        float(row["final_support_recall"]) - float(row["recall_after_intervention"])
        for row in branches
    ]
    composite_values = [
        final_recall
        + answer_weight * answer_f1
        - search_cost * float(row["suffix_search_count"])
        for final_recall, answer_f1, row in zip(
            final_recalls, answer_f1s, branches, strict=True
        )
    ]
    support_tqe = leave_one_out_effect(final_recalls)
    answer_tqe = leave_one_out_effect(answer_f1s)
    direct_effect = leave_one_out_effect(direct_gains)
    downstream_effect = leave_one_out_effect(downstream_gains)
    composite_tqe = leave_one_out_effect(composite_values)

    output: list[dict[str, Any]] = []
    for index, source in enumerate(branches):
        row = dict(source)
        row.update(
            {
                "downstream_support_gain": downstream_gains[index],
                "composite_utility": composite_values[index],
                "support_tqe": support_tqe[index],
                "answer_tqe": answer_tqe[index],
                "direct_effect": direct_effect[index],
                "downstream_effect": downstream_effect[index],
                "composite_tqe": composite_tqe[index],
            }
        )
        row["mediated_bridge"] = int(
            float(row["immediate_support_gain"]) <= epsilon
            and bool(str(row.get("next_query", "")).strip())
            and float(row.get("next_query_evidence_gain", 0.0)) > epsilon
            and int(row.get("transferred_bridge_token_count", 0)) > 0
        )
        row["positive_causal_bridge"] = int(
            row["mediated_bridge"] == 1
            and float(row["support_tqe"]) >= bridge_min_support_tqe
        )
        row["redundant_direct"] = int(
            float(row["immediate_support_gain"]) > epsilon
            and float(row["support_tqe"]) <= 0.0
        )
        if abs(
            float(row["support_tqe"])
            - float(row["direct_effect"])
            - float(row["downstream_effect"])
        ) > 1e-8:
            raise RuntimeError("Direct/downstream query-effect decomposition drifted")
        output.append(row)
    return output


def transferred_bridge_tokens(
    *,
    next_query: str,
    intervention_titles: Sequence[str],
    prior_text: str,
) -> list[str]:
    next_tokens = token_set(next_query, content_only=True)
    title_tokens = token_set(" ".join(intervention_titles), content_only=True)
    prior_tokens = token_set(prior_text, content_only=True)
    return sorted((next_tokens & title_tokens) - prior_tokens)


def bootstrap_by_state(
    rows: Sequence[dict[str, Any]],
    *,
    state_key: str,
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[state_key]), []).append(dict(row))
    states = sorted(grouped)
    if not states:
        raise RuntimeError("No states are available for bootstrap")
    observed = float(statistic([row for state in states for row in grouped[state]]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(states, size=len(states), replace=True)
        sample_rows: list[dict[str, Any]] = []
        for draw_index, state in enumerate(sampled):
            for row in grouped[str(state)]:
                copy = dict(row)
                copy["_bootstrap_state"] = f"{draw_index}:{state}"
                sample_rows.append(copy)
        draws[index] = float(statistic(sample_rows))
    finite = draws[np.isfinite(draws)]
    if len(finite) != len(draws):
        raise RuntimeError("A bootstrap statistic produced non-finite values")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_states": float(len(states)),
        "n_rows": float(len(rows)),
    }

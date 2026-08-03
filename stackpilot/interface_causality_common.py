from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA = 1
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "why", "with",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported interface-causality config: {config_path}")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_hash(*parts: object, length: int = 24) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def word_tokens(value: str, *, content_only: bool = False) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value))]
    if content_only:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def token_set(value: str, *, content_only: bool = False) -> set[str]:
    return set(word_tokens(value, content_only=content_only))


def ngram_set(value: str, n: int = 3) -> set[str]:
    text = " ".join(word_tokens(value))
    if len(text) < n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def discover_paths(patterns: Sequence[str]) -> list[Path]:
    import glob

    output: dict[str, Path] = {}
    for pattern in patterns:
        expanded = os.path.expanduser(str(pattern))
        if Path(expanded).is_file():
            path = Path(expanded).resolve()
            output[str(path)] = path
            continue
        for raw in glob.glob(expanded, recursive=True):
            path = Path(raw).resolve()
            if path.is_file() and path.suffix == ".json":
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def source_patterns(cfg: dict[str, Any], provided: Sequence[str] | None = None) -> list[str]:
    if provided:
        return [str(value) for value in provided]
    environment = os.environ.get("INTERFACE_CAUSAL_INPUTS", "").strip()
    if environment:
        return [part for part in environment.replace("\n", os.pathsep).split(os.pathsep) if part]
    return [str(value) for value in cfg["source"]["input_globs"]]


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def candidate_observed_titles(candidate: dict[str, Any], *, final: bool) -> list[str]:
    if not final:
        return _strings(candidate.get("intervention_observed_titles"))
    values: list[str] = []
    for record in candidate.get("branch_turns", []) or []:
        if isinstance(record, dict):
            values.extend(_strings(record.get("observed_titles")))
    if not values:
        values = _strings(candidate.get("intervention_observed_titles"))
    return values


def gold_support_set(candidate: dict[str, Any], state: dict[str, Any], *, final: bool) -> tuple[str, ...]:
    gold = {normalize_title(value) for value in _strings(state.get("support_titles"))}
    observed = {
        normalize_title(value)
        for value in candidate_observed_titles(candidate, final=final)
    }
    return tuple(sorted(gold & observed))


def ranked_transition(candidate: dict[str, Any]) -> tuple[str, ...]:
    titles = candidate_observed_titles(candidate, final=False)
    return tuple(normalize_title(value) for value in titles)


def behavior_signature(
    candidate: dict[str, Any],
    state: dict[str, Any],
    *,
    mode: str,
) -> tuple[Any, ...]:
    if mode == "ranked-transition":
        return ranked_transition(candidate)
    if mode == "gold-transition":
        return (
            gold_support_set(candidate, state, final=False),
            gold_support_set(candidate, state, final=True),
            int(float(candidate.get("answer_em", 0.0)) > 0.5),
        )
    if mode == "final-outcome":
        return (
            gold_support_set(candidate, state, final=True),
            int(float(candidate.get("answer_em", 0.0)) > 0.5),
            int(candidate.get("total_search_count", 0)),
        )
    raise ValueError(f"Unknown behavior-signature mode: {mode}")


def validate_state_result(payload: dict[str, Any], path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = payload.get("state")
    candidates = payload.get("candidates")
    label = str(path) if path is not None else str(payload.get("state_signature", "state"))
    if not isinstance(state, dict) or not isinstance(candidates, list) or len(candidates) < 2:
        raise RuntimeError(f"Invalid causal-query result: {label}")
    required_state = {
        "state_id", "question_id", "question", "dataset", "backend", "topk",
        "source_turn", "support_titles",
    }
    missing = required_state - set(state)
    if missing:
        raise RuntimeError(f"{label} state misses {sorted(missing)}")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"{label} candidate {index} is not an object")
        for key in ("candidate_id", "query", "immediate_support_gain", "final_support_recall"):
            if key not in candidate:
                raise RuntimeError(f"{label} candidate {index} misses {key}")
        for key in (
            "immediate_support_gain", "final_support_recall", "answer_f1",
            "support_tqe", "composite_tqe",
        ):
            value = float(candidate.get(key, 0.0))
            if not math.isfinite(value):
                raise RuntimeError(f"{label} candidate {index} has non-finite {key}")
    return state, candidates


def load_state_results(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = discover_paths(patterns)
    if not paths:
        raise RuntimeError(f"No causal-query state JSON matched: {list(patterns)}")
    output = []
    run_signatures: set[str] = set()
    seen_state_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state, _candidates = validate_state_result(payload, path)
        state_id = str(state["state_id"])
        if state_id in seen_state_ids:
            continue
        seen_state_ids.add(state_id)
        run_signature = str(payload.get("run_signature", ""))
        if run_signature:
            run_signatures.add(run_signature)
        payload["_source_path"] = str(path)
        output.append(payload)
    if len(run_signatures) > 1:
        raise RuntimeError(f"Input state results mix run signatures: {sorted(run_signatures)}")
    return output


def balanced_state_subset(
    results: Sequence[dict[str, Any]],
    limit: int,
    *,
    keys: Sequence[str] = ("backend", "dataset"),
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(results):
        return sorted((dict(row) for row in results), key=lambda row: str(row["state"]["state_id"]))
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        state = result["state"]
        buckets[tuple(state.get(key) for key in keys)].append(dict(result))
    for key in buckets:
        buckets[key].sort(key=lambda row: stable_hash("balanced-state", row["state"]["state_id"]))
    active = sorted(buckets, key=repr)
    output: list[dict[str, Any]] = []
    cursor = 0
    while active and len(output) < limit:
        key = active[cursor % len(active)]
        if buckets[key]:
            output.append(buckets[key].pop(0))
            cursor += 1
        else:
            active.remove(key)
            if active:
                cursor %= len(active)
    return output


def group_candidates(
    state: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        grouped[behavior_signature(candidate, state, mode=mode)].append(dict(candidate))
    classes = list(grouped.values())
    classes.sort(
        key=lambda rows: (
            -max(float(row.get("final_support_recall", 0.0)) for row in rows),
            -max(float(row.get("answer_f1", 0.0)) for row in rows),
            -len(rows),
            min(str(row.get("candidate_id", "")) for row in rows),
        )
    )
    return classes


def candidate_reward(candidate: dict[str, Any], cfg: dict[str, Any]) -> float:
    weights = cfg["reward"]
    value = (
        float(weights["support"]) * float(candidate.get("final_support_recall", 0.0))
        + float(weights["answer_f1"]) * float(candidate.get("answer_f1", 0.0))
        + float(weights["immediate_gain"]) * float(candidate.get("immediate_support_gain", 0.0))
        - float(weights["search_cost"]) * float(candidate.get("total_search_count", 0.0))
        - float(weights["protocol_cost"]) * float(candidate.get("protocol_failure", 0.0))
    )
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite candidate reward: {value}")
    return value


def normalize_advantages(values: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot normalize empty reward vector")
    standard = float(array.std(ddof=0))
    if standard <= epsilon:
        return np.zeros_like(array)
    return (array - float(array.mean())) / standard


def effective_count(probabilities: Sequence[float]) -> float:
    array = np.asarray(probabilities, dtype=np.float64)
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    array = array / total
    return float(1.0 / np.square(array).sum())


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    cluster_key: str,
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(dict(row))
    keys = sorted(grouped)
    if not keys:
        raise RuntimeError("Cannot bootstrap empty rows")
    observed = float(statistic([item for key in keys for item in grouped[key]]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_keys = generator.choice(keys, size=len(keys), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        for draw_index, sampled_key in enumerate(sampled_keys):
            for row in grouped[str(sampled_key)]:
                copy = dict(row)
                copy["_bootstrap_cluster"] = f"{sampled_key}:{draw_index}"
                sampled_rows.append(copy)
        draws[index] = float(statistic(sampled_rows))
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise RuntimeError("All bootstrap draws are non-finite")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_clusters": float(len(keys)),
        "n_rows": float(len(rows)),
        "finite_draws": float(len(finite)),
    }


def markdown_table(frame: Any, digits: int = 4) -> str:
    import pandas as pd

    if frame is None or frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(map(float, scores), map(int, labels)), key=lambda row: row[0])
    positives = sum(label == 1 for _score, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        rank_sum += average_rank * sum(label == 1 for _score, label in pairs[index:end])
        index = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    ordered_labels = np.asarray(labels, dtype=np.int64)[order]
    positives = int(ordered_labels.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(ordered_labels)
    precision = cumulative / (np.arange(len(ordered_labels)) + 1)
    return float((precision * ordered_labels).sum() / positives)


def classification_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities_array >= threshold
    true_positive = int(((predictions == 1) & (labels_array == 1)).sum())
    false_positive = int(((predictions == 1) & (labels_array == 0)).sum())
    false_negative = int(((predictions == 0) & (labels_array == 1)).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "auc": auc_score(labels_array, probabilities_array),
        "average_precision": average_precision(labels_array, probabilities_array),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "brier": float(np.mean(np.square(probabilities_array - labels_array))),
        "positive_rate": float(labels_array.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "n": float(len(labels_array)),
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float,
    l2: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mean = features.mean(axis=0)
    standard = features.std(axis=0)
    standard[standard < 1e-8] = 1.0
    normalized = (features - mean) / standard
    augmented = np.concatenate([np.ones((len(normalized), 1)), normalized], axis=1)
    weights = np.zeros(augmented.shape[1], dtype=np.float64)
    for _step in range(steps):
        probabilities = sigmoid(augmented @ weights)
        gradient = augmented.T @ (probabilities - labels) / len(labels)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights, mean, standard


def predict_logistic(features: np.ndarray, weights: np.ndarray, mean: np.ndarray, standard: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(features, dtype=np.float64) - mean) / standard
    augmented = np.concatenate([np.ones((len(normalized), 1)), normalized], axis=1)
    return sigmoid(augmented @ weights)

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA = 1
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
FEATURE_NAMES = (
    "reciprocal_rank",
    "rank_fraction",
    "retriever_score_z",
    "query_title_jaccard",
    "query_text_jaccard",
    "exact_title_in_query",
    "exact_query_in_text",
    "numeric_overlap_title",
    "numeric_overlap_text",
    "title_tokens",
    "text_tokens_log1p",
    "query_tokens",
    "duplicate_title_count",
    "backend_e5",
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported credit-routing config: {config_path}")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: object, length: int = 24) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts, length=8), 16) % (2**31)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            output.append(row)
    return output


def discover_paths(patterns: Sequence[str], suffixes: tuple[str, ...] = (".json", ".jsonl")) -> list[Path]:
    output: dict[str, Path] = {}
    for pattern in patterns:
        expanded = os.path.expanduser(str(pattern))
        direct = Path(expanded)
        if direct.is_file() and (not suffixes or direct.suffix in suffixes):
            output[str(direct.resolve())] = direct.resolve()
            continue
        for raw in glob.glob(expanded, recursive=True):
            path = Path(raw).resolve()
            if path.is_file() and (not suffixes or path.suffix in suffixes):
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def env_patterns(name: str, defaults: Sequence[str], provided: Sequence[str] | None = None) -> list[str]:
    if provided:
        return [str(value) for value in provided]
    raw = os.environ.get(name, "").strip()
    if raw:
        return [value for value in raw.replace("\n", os.pathsep).split(os.pathsep) if value]
    return [str(value) for value in defaults]


def normalize_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def word_tokens(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value or ""))]


def token_set(value: Any) -> set[str]:
    return set(word_tokens(value))


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def normalize_document(item: dict[str, Any]) -> tuple[str, str, float]:
    if not isinstance(item, dict):
        raise TypeError("Retriever item must be an object")
    document = item.get("document") if isinstance(item.get("document"), dict) else item
    metadata = document.get("document_metadata") or document.get("metadata") or {}
    contents = str(document.get("contents") or "")
    title = str(document.get("title") or metadata.get("title") or "").strip()
    text = str(document.get("text") or document.get("content") or "").strip()
    if contents:
        first, separator, remainder = contents.partition("\n")
        if not title:
            title = first.strip().strip('"')
        if not text and separator:
            text = remainder.strip()
    score = float(item.get("score", document.get("score", 0.0)) or 0.0)
    if not title:
        raise RuntimeError("Retriever item has no document title")
    if not math.isfinite(score):
        raise RuntimeError("Retriever item has non-finite score")
    return title, text, score


def feature_rows(query: str, items: Sequence[dict[str, Any]], backend: str) -> list[dict[str, float]]:
    normalized: list[tuple[str, str, float]] = [normalize_document(dict(item)) for item in items]
    scores = np.asarray([row[2] for row in normalized], dtype=np.float64)
    score_mean = float(scores.mean()) if len(scores) else 0.0
    score_std = float(scores.std()) if len(scores) else 0.0
    query_tokens = token_set(query)
    query_text = normalize_title(query)
    title_counts = Counter(normalize_title(title) for title, _, _ in normalized)
    total = max(1, len(normalized))
    output: list[dict[str, float]] = []
    for rank, (title, text, score) in enumerate(normalized, start=1):
        title_tokens = token_set(title)
        text_tokens = token_set(text)
        title_normalized = normalize_title(title)
        text_normalized = normalize_title(text)
        query_numbers = {value for value in query_tokens if value.isdigit()}
        row = {
            "reciprocal_rank": 1.0 / rank,
            "rank_fraction": rank / total,
            "retriever_score_z": (score - score_mean) / (score_std + 1e-8),
            "query_title_jaccard": jaccard(query_tokens, title_tokens),
            "query_text_jaccard": jaccard(query_tokens, text_tokens),
            "exact_title_in_query": float(bool(title_normalized) and title_normalized in query_text),
            "exact_query_in_text": float(bool(query_text) and query_text in text_normalized),
            "numeric_overlap_title": float(bool(query_numbers & {v for v in title_tokens if v.isdigit()})),
            "numeric_overlap_text": float(bool(query_numbers & {v for v in text_tokens if v.isdigit()})),
            "title_tokens": float(len(word_tokens(title))),
            "text_tokens_log1p": float(math.log1p(len(word_tokens(text)))),
            "query_tokens": float(len(word_tokens(query))),
            "duplicate_title_count": float(title_counts[title_normalized]),
            "backend_e5": float(str(backend).lower() == "e5"),
        }
        output.append(row)
    return output


def matrix_from_feature_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row.get(name, 0.0)) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Feature matrix has invalid shape")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Feature matrix contains non-finite values")
    return matrix


def fit_standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    scale = np.asarray(matrix.std(axis=0), dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_standardizer(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(matrix, dtype=np.float64) - mean) / scale
    if not np.isfinite(standardized).all():
        raise RuntimeError("Standardized feature matrix contains non-finite values")
    return standardized


def fit_ridge(matrix: np.ndarray, targets: np.ndarray, *, l2: float) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    if not np.isfinite(weights).all():
        raise RuntimeError("Ridge fit produced non-finite weights")
    return weights


def predict_ridge(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [np.ones((len(matrix), 1), dtype=np.float64), np.asarray(matrix, dtype=np.float64)],
        axis=1,
    )
    values = design @ np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Ridge prediction produced non-finite values")
    return values


def score_artifact(query: str, items: Sequence[dict[str, Any]], backend: str, artifact: dict[str, Any]) -> np.ndarray:
    if list(artifact.get("feature_names", [])) != list(FEATURE_NAMES):
        raise RuntimeError("Utility artifact feature schema does not match runtime features")
    rows = feature_rows(query, items, backend)
    matrix = matrix_from_feature_rows(rows)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    standardized = apply_standardizer(matrix, mean, scale)
    draws = [predict_ridge(standardized, np.asarray(weights, dtype=np.float64)) for weights in artifact["weights"]]
    scores = np.mean(draws, axis=0)
    if len(scores) != len(items):
        raise RuntimeError("Utility artifact score count does not match retriever output")
    return scores


def selection_indices(scores: Sequence[float], k: int, *, mode: str) -> list[int]:
    if k < 1:
        raise ValueError("k must be positive")
    count = len(scores)
    if count < k:
        raise RuntimeError(f"Only {count} candidates are available for k={k}")
    if mode == "rank":
        return list(range(k))
    if mode == "utility":
        return sorted(range(count), key=lambda index: (-float(scores[index]), index))[:k]
    raise ValueError(f"Unknown selection mode: {mode}")


def aggregate_document_utility(scores: Sequence[float], *, k: int, mode: str) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if mode == "mean-topk":
        selected = np.sort(values)[::-1][: min(k, len(values))]
        return float(selected.mean())
    if mode == "max":
        return float(values.max())
    if mode == "mean":
        return float(values.mean())
    if mode == "positive-sum":
        return float(np.maximum(values, 0.0).sum())
    raise ValueError(f"Unknown utility aggregation: {mode}")


def fixed_budget_contexts(candidate_count: int, keep_k: int) -> list[tuple[int, ...]]:
    """Enumerate every equal-cardinality observation context.

    For the preregistered top-8 / keep-3 setting this produces 56 contexts.
    Enumerating contexts once lets every document be compared through the same
    matched-swap design rather than against a privileged rank-specific anchor.
    """
    if keep_k < 1 or candidate_count <= keep_k:
        raise ValueError("Fixed-budget utility needs candidate_count > keep_k >= 1")
    return [tuple(values) for values in combinations(range(candidate_count), keep_k)]


def matched_swap_utilities(
    context_values: dict[tuple[int, ...], float],
    *,
    candidate_count: int,
    keep_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate comparable document utility by exact matched swaps.

    For document ``i``, compare ``B ∪ {i}`` with ``B ∪ {j}`` for every other
    document ``j`` and every common ``keep_k - 1`` context ``B`` that excludes
    both.  This holds context cardinality and all companion documents fixed.
    The resulting score is the average advantage of selecting ``i`` instead of
    another candidate under the same fixed observation budget.
    """
    expected = set(fixed_budget_contexts(candidate_count, keep_k))
    provided = {tuple(sorted(key)) for key in context_values}
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise RuntimeError(
            f"Context-value grid is incomplete: missing={missing[:3]}, extra={extra[:3]}"
        )
    values = {tuple(sorted(key)): float(value) for key, value in context_values.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Context-value grid contains non-finite values")
    utilities = np.zeros(candidate_count, dtype=np.float64)
    counts = np.zeros(candidate_count, dtype=np.int64)
    for candidate in range(candidate_count):
        differences: list[float] = []
        for control in range(candidate_count):
            if control == candidate:
                continue
            remaining = [
                index
                for index in range(candidate_count)
                if index not in {candidate, control}
            ]
            for base in combinations(remaining, keep_k - 1):
                with_candidate = tuple(sorted((*base, candidate)))
                with_control = tuple(sorted((*base, control)))
                differences.append(values[with_candidate] - values[with_control])
        if not differences:
            raise RuntimeError(f"No matched swaps were available for document {candidate}")
        utilities[candidate] = float(np.mean(differences))
        counts[candidate] = len(differences)
    if abs(float(utilities.sum())) > 1e-8:
        raise RuntimeError(
            f"Exact matched-swap utilities must sum to zero; got {utilities.sum()}"
        )
    return utilities, counts


def paired_context_indices(candidate_index: int, candidate_count: int, keep_k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if keep_k < 1 or candidate_count <= keep_k:
        raise ValueError("Budgeted CTU needs candidate_count > keep_k >= 1")
    if candidate_index < 0 or candidate_index >= candidate_count:
        raise IndexError(candidate_index)
    baseline = tuple(range(keep_k))
    if candidate_index < keep_k:
        with_document = baseline
        without_document = tuple(
            index for index in range(keep_k + 1) if index != candidate_index
        )[:keep_k]
    else:
        with_document = tuple(range(keep_k - 1)) + (candidate_index,)
        without_document = baseline
    if len(with_document) != keep_k or len(without_document) != keep_k:
        raise RuntimeError("Budgeted CTU contexts do not have equal cardinality")
    if candidate_index not in with_document or candidate_index in without_document:
        raise RuntimeError("Budgeted CTU intervention failed inclusion contract")
    return with_document, without_document


def question_split(question_id: str, *, salt: str, train_fraction: float, validation_fraction: float) -> str:
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split fractions")
    value = int(stable_hash(salt, question_id, length=12), 16) / float(16**12)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Spearman inputs contain non-finite values")
    a_rank = _average_ranks(a)
    b_rank = _average_ranks(b)
    if a_rank.std() < 1e-12 or b_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def markdown_table(rows: Any) -> str:
    try:
        return rows.to_markdown(index=False, floatfmt=".4f")
    except AttributeError:
        values = list(rows)
        if not values:
            return "(empty)"
        columns = list(values[0])
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in values:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

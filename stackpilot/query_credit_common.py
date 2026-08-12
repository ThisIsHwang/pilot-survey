from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != 1:
        raise ValueError(f"Unsupported query-credit config: {path}")
    return payload


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
    )


def read_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def stable_hash(*parts: object, length: int = 24) -> str:
    digest = hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()
    return digest[:length]


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts, length=8), 16) % (2**31)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def word_tokens(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value or ""))]


def token_set(value: Any) -> set[str]:
    return set(word_tokens(value))


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a = {str(value) for value in left}
    b = {str(value) for value in right}
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def behavior_signature(titles: Sequence[str]) -> str:
    return stable_hash("ranked-titles-v1", *[normalize_text(value) for value in titles], length=32)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def composite_reward(
    *,
    support_recall: float,
    answer_f1: float,
    search_count: float,
    invalid_action_count: float,
    weights: Mapping[str, float],
) -> float:
    return (
        float(weights.get("support", 1.0)) * float(support_recall)
        + float(weights.get("answer_f1", 0.5)) * float(answer_f1)
        - float(weights.get("search_cost", 0.0)) * float(search_count)
        - float(weights.get("invalid_action", 0.0)) * float(invalid_action_count)
    )


def best_replacement_gap(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        raise ValueError("At least two query candidates are required")
    array = np.asarray(values, dtype=np.float64)
    output: list[float] = []
    for index, value in enumerate(array):
        best_other = float(np.max(np.delete(array, index)))
        output.append(float(value - best_other))
    return output


def centered_action_advantage(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        raise ValueError("At least two query candidates are required")
    array = np.asarray(values, dtype=np.float64)
    return [float(value - array.mean()) for value in array]


def aggregate_document_credit(values: Sequence[float], mode: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if mode == "positive-sum":
        return float(np.maximum(array, 0.0).sum())
    if mode == "signed-sum":
        return float(array.sum())
    if mode == "max":
        return float(array.max())
    if mode == "mean":
        return float(array.mean())
    if mode == "top2-sum":
        return float(np.sort(array)[-min(2, array.size) :].sum())
    raise ValueError(f"Unknown document-credit aggregation: {mode}")


def state_standardize(values: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    std = float(array.std())
    if std <= epsilon:
        return np.zeros_like(array)
    return (array - array.mean()) / (std + epsilon)


def average_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
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
    a = average_rank(left)
    b = average_rank(right)
    if float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = average_rank(s)
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / (np.arange(len(sorted_y)) + 1)
    return float((precision * sorted_y).sum() / positives)


def question_split(question_id: str, salt: str = "query-credit-v1") -> str:
    bucket = int(stable_hash(salt, question_id, length=8), 16) % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "validation"
    return "test"


def bootstrap_by_cluster(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    statistic,
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(dict(row))
    clusters = sorted(grouped)
    if not clusters:
        raise RuntimeError("No clusters available")
    observed = float(statistic([row for key in clusters for row in grouped[key]]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values: list[dict[str, Any]] = []
        for draw_index, key in enumerate(sampled):
            for row in grouped[str(key)]:
                copy = dict(row)
                copy["_bootstrap_cluster"] = f"{draw_index}:{key}"
                values.append(copy)
        draws[draw] = float(statistic(values))
    if not np.isfinite(draws).all():
        raise RuntimeError("Bootstrap produced non-finite values")
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_clusters": float(len(clusters)),
        "n_rows": float(len(rows)),
    }


FEATURE_NAMES = (
    "bias",
    "rank",
    "reciprocal_rank",
    "score",
    "score_z",
    "query_title_jaccard",
    "query_text_jaccard",
    "query_title_exact_overlap",
    "query_text_exact_overlap",
    "number_overlap",
    "title_token_count",
    "text_token_count_log",
    "query_token_count_log",
    "backend_is_e5",
)


def document_features(row: Mapping[str, Any]) -> np.ndarray:
    query_tokens = token_set(row.get("query", ""))
    title_tokens = token_set(row.get("document_title", ""))
    text_tokens = token_set(row.get("document_text", ""))
    query_numbers = {token for token in query_tokens if token.isdigit()}
    document_numbers = {token for token in title_tokens | text_tokens if token.isdigit()}
    rank = max(1.0, float(row.get("document_rank", 1.0)))
    score = float(row.get("retriever_score", 0.0) or 0.0)
    score_z = float(row.get("retriever_score_z", 0.0) or 0.0)
    values = np.asarray(
        [
            1.0,
            rank,
            1.0 / rank,
            score,
            score_z,
            jaccard(query_tokens, title_tokens),
            jaccard(query_tokens, text_tokens),
            float(len(query_tokens & title_tokens)),
            float(len(query_tokens & text_tokens)),
            float(len(query_numbers & document_numbers)),
            float(len(title_tokens)),
            math.log1p(len(text_tokens)),
            math.log1p(len(query_tokens)),
            float(str(row.get("backend", "")).lower() == "e5"),
        ],
        dtype=np.float64,
    )
    return values


@dataclass(frozen=True)
class LinearUtilityModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: tuple[np.ndarray, ...]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LinearUtilityModel":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coefficients=tuple(
                np.asarray(values, dtype=np.float64)
                for values in payload["coefficients"]
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": [values.tolist() for values in self.coefficients],
        }

    def predict_vector(self, vector: np.ndarray) -> float:
        if tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise RuntimeError("Document feature schema mismatch")
        standardized = (vector - self.mean) / self.scale
        return float(np.mean([standardized @ coef for coef in self.coefficients]))

    def predict_row(self, row: Mapping[str, Any]) -> float:
        return self.predict_vector(document_features(row))


def fit_ridge_ensemble(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alphas: Sequence[float],
) -> LinearUtilityModel:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Invalid ridge training geometry")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    coefficients = []
    identity = np.eye(standardized.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0
    for alpha in alphas:
        lhs = standardized.T @ standardized + float(alpha) * identity
        rhs = standardized.T @ y
        try:
            coefficients.append(np.linalg.solve(lhs, rhs))
        except np.linalg.LinAlgError:
            coefficients.append(np.linalg.pinv(lhs) @ rhs)
    return LinearUtilityModel(tuple(FEATURE_NAMES), mean, scale, tuple(coefficients))


def find_subsequence(values: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    if not needle:
        return -1
    stop = len(values) - len(needle) + 1
    for index in range(max(0, start), max(0, stop)):
        if list(values[index : index + len(needle)]) == list(needle):
            return index
    return -1


def build_search_span_ids(
    responses: Any,
    *,
    pad_token_id: int,
    open_ids: Sequence[int],
    close_ids: Sequence[int],
) -> Any:
    """Return a tensor-like array with 1-based search-turn IDs on query tokens.

    The function accepts NumPy arrays or Torch tensors and returns the same broad
    type. Tags are included in the span because Search-R1 learns the entire
    executable action. Observation tokens remain zero.
    """

    is_torch = hasattr(responses, "detach")
    array = responses.detach().cpu().numpy() if is_torch else np.asarray(responses)
    output = np.zeros_like(array, dtype=np.int64)
    for row_index, raw in enumerate(array):
        values = [int(value) for value in raw]
        cursor = 0
        turn = 1
        while True:
            opening = find_subsequence(values, open_ids, cursor)
            if opening < 0:
                break
            closing = find_subsequence(values, close_ids, opening + len(open_ids))
            if closing < 0:
                break
            end = closing + len(close_ids)
            for position in range(opening, end):
                if values[position] != int(pad_token_id):
                    output[row_index, position] = turn
            cursor = end
            turn += 1
    if is_torch:
        import torch

        return torch.as_tensor(output, device=responses.device, dtype=torch.long)
    return output


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

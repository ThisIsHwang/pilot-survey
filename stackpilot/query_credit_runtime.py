from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stackpilot.query_credit_common import (
    LinearUtilityModel,
    aggregate_document_credit,
    behavior_signature,
    build_search_span_ids,
    state_standardize,
    stable_seed,
)

_MODEL_CACHE: dict[str, LinearUtilityModel] = {}


def load_utility_model(path: str) -> LinearUtilityModel:
    resolved = str(Path(path).resolve())
    model = _MODEL_CACHE.get(resolved)
    if model is None:
        payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
        model = LinearUtilityModel.from_json(payload)
        _MODEL_CACHE[resolved] = model
    return model


def score_document_batch(
    model: LinearUtilityModel,
    *,
    query: str,
    documents: Sequence[dict[str, Any]],
    backend: str,
) -> list[float]:
    scores = np.asarray(
        [float(row.get("retriever_score", 0.0) or 0.0) for row in documents],
        dtype=np.float64,
    )
    std = float(scores.std()) if scores.size else 0.0
    z = np.zeros_like(scores) if std <= 1e-12 else (scores - scores.mean()) / std
    output = []
    for index, document in enumerate(documents):
        row = dict(document)
        row.update(
            {
                "query": query,
                "backend": backend,
                "document_rank": int(document.get("document_rank", index + 1)),
                "retriever_score_z": float(z[index]) if index < len(z) else 0.0,
            }
        )
        output.append(model.predict_row(row))
    return output


def _row_turn_scores(
    query_batches: Sequence[Sequence[str]],
    document_batches: Sequence[Sequence[Sequence[dict[str, Any]]]],
    *,
    model: LinearUtilityModel,
    backend: str,
    aggregation: str,
) -> list[list[float]]:
    output = []
    for queries, turns in zip(query_batches, document_batches, strict=True):
        row_scores = []
        for query, documents in zip(list(queries), list(turns), strict=False):
            utilities = score_document_batch(
                model,
                query=str(query),
                documents=list(documents),
                backend=backend,
            )
            row_scores.append(aggregate_document_credit(utilities, aggregation))
        output.append(row_scores)
    return output


def _normalize_turn_scores(
    raw_scores: list[list[float]],
    *,
    index: Sequence[Any],
    title_batches: Sequence[Sequence[Sequence[str]]],
    mode: str,
    seed: int,
) -> list[list[float]]:
    output = [[0.0 for _ in row] for row in raw_scores]
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row_index, prompt_id in enumerate(index):
        for turn_index in range(len(raw_scores[row_index])):
            groups[(str(prompt_id), turn_index)].append(row_index)
    rng = np.random.default_rng(seed)
    for (_prompt, turn_index), row_indices in groups.items():
        values = np.asarray([raw_scores[row_index][turn_index] for row_index in row_indices], dtype=np.float64)
        if mode == "shuffled-doc":
            values = values[rng.permutation(len(values))]
            normalized = state_standardize(values)
            for local, row_index in enumerate(row_indices):
                output[row_index][turn_index] = float(normalized[local])
            continue
        if mode == "alias-normalized":
            classes: dict[str, list[int]] = defaultdict(list)
            for local, row_index in enumerate(row_indices):
                titles = []
                if row_index < len(title_batches) and turn_index < len(title_batches[row_index]):
                    titles = list(title_batches[row_index][turn_index])
                classes[behavior_signature(titles)].append(local)
            class_keys = sorted(classes)
            class_values = np.asarray(
                [float(np.mean([values[local] for local in classes[key]])) for key in class_keys],
                dtype=np.float64,
            )
            class_normalized = state_standardize(class_values)
            for class_index, key in enumerate(class_keys):
                members = classes[key]
                share = float(class_normalized[class_index]) / max(1, len(members))
                for local in members:
                    output[row_indices[local]][turn_index] = share
            continue
        normalized = state_standardize(values)
        for local, row_index in enumerate(row_indices):
            output[row_index][turn_index] = float(normalized[local])
    return output


def apply_query_credit_bonus(
    *,
    advantages: Any,
    search_span_ids: Any,
    index: Sequence[Any],
    query_batches: Sequence[Sequence[str]],
    title_batches: Sequence[Sequence[Sequence[str]]],
    document_batches: Sequence[Sequence[Sequence[dict[str, Any]]]],
    mode: str,
    model_path: str,
    backend: str,
    aggregation: str,
    alpha: float,
    seed: int,
) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
    if mode == "outcome":
        return advantages, {"query_credit/bonus_abs_mean": 0.0}, []
    if mode not in {"doc-to-action", "alias-normalized", "shuffled-doc"}:
        raise ValueError(f"Unknown online query-credit mode: {mode}")
    model = load_utility_model(model_path)
    raw = _row_turn_scores(
        query_batches,
        document_batches,
        model=model,
        backend=backend,
        aggregation=aggregation,
    )
    normalized = _normalize_turn_scores(
        raw,
        index=index,
        title_batches=title_batches,
        mode=mode,
        seed=seed,
    )
    updated = advantages.clone()
    rows = []
    values = []
    for row_index, turn_scores in enumerate(normalized):
        for turn_index, bonus in enumerate(turn_scores, start=1):
            mask = search_span_ids[row_index] == turn_index
            if bool(mask.any()):
                updated[row_index, mask] = updated[row_index, mask] + float(alpha) * float(bonus)
            values.append(abs(float(bonus)))
            rows.append(
                {
                    "row_index": row_index,
                    "turn": turn_index,
                    "raw_document_credit": float(raw[row_index][turn_index - 1]),
                    "normalized_query_bonus": float(bonus),
                    "mode": mode,
                }
            )
    metrics = {
        "query_credit/bonus_abs_mean": float(np.mean(values)) if values else 0.0,
        "query_credit/bonus_nonzero_rate": float(np.mean([value > 1e-12 for value in values])) if values else 0.0,
        "query_credit/search_spans": float(len(values)),
    }
    return updated, metrics, rows


__all__ = [
    "apply_query_credit_bonus",
    "build_search_span_ids",
    "load_utility_model",
    "score_document_batch",
]

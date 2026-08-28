from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from stackpilot.query_credit_common import normalize_text, spearman, stable_hash


def apply_model_override(cfg: dict[str, Any]) -> dict[str, Any]:
    """Use a resolved local model path when the launcher exports one."""
    value = os.environ.get("BASE_MODEL", "").strip()
    if value:
        cfg.setdefault("model", {})["base_model"] = value
        if os.path.isdir(value):
            cfg["model"]["revision"] = None
    return cfg


def stable_balanced_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    backends: Sequence[str],
    per_cell: int,
    salt: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Choose paired, result-blind questions across dataset/backend cells."""
    wanted_datasets = sorted({str(value).lower() for value in datasets})
    wanted_backends = sorted({str(value).lower() for value in backends})
    indexed: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw in rows:
        row = dict(raw)
        state = row.get("state", row)
        dataset = str(state.get("dataset", "")).lower()
        backend = str(state.get("backend", "")).lower()
        if dataset not in wanted_datasets or backend not in wanted_backends:
            continue
        question_id = str(state.get("question_id", ""))
        if question_id:
            indexed[(dataset, backend)][question_id].append(row)

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for dataset in wanted_datasets:
        question_sets = [
            set(indexed[(dataset, backend)]) for backend in wanted_backends
        ]
        common_questions = set.intersection(*question_sets) if question_sets else set()
        ordered_questions = sorted(
            common_questions,
            key=lambda question_id: stable_hash(
                salt, dataset, question_id, length=32
            ),
        )
        chosen_questions = ordered_questions[: int(per_cell)]
        for backend in wanted_backends:
            for question_id in chosen_questions:
                candidates = indexed[(dataset, backend)][question_id]
                chosen = min(
                    candidates,
                    key=lambda item: stable_hash(
                        salt,
                        dataset,
                        backend,
                        question_id,
                        str(item.get("state", item).get("state_id", "")),
                        length=32,
                    ),
                )
                selected.append(chosen)
            counts[f"{dataset}/{backend}"] = len(chosen_questions)
    return selected, counts


def _document_parts(item: Mapping[str, Any]) -> tuple[str, str]:
    document = item.get("document")
    if isinstance(document, Mapping):
        contents = str(document.get("contents") or "")
        title, separator, text = contents.partition("\n")
        return title.strip(), text.strip() if separator else ""
    return (
        str(item.get("title") or item.get("document_title") or "").strip(),
        str(item.get("text") or item.get("content") or item.get("document_text") or "").strip(),
    )


def document_token_length(item: Mapping[str, Any], tokenizer: Any) -> int:
    title, text = _document_parts(item)
    encoded = tokenizer(f"{title}\n{text}", add_special_tokens=False)
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return int(len(values))


def choose_length_matched_replacements(
    results: Sequence[Mapping[str, Any]],
    *,
    visible_documents: int,
    pool_start_rank: int,
    pool_end_rank: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Precommit one fixed-cardinality replacement per visible slot."""
    if visible_documents < 1:
        raise ValueError("visible_documents must be positive")
    if len(results) < visible_documents:
        raise ValueError("retrieval result is smaller than the visible set")
    first_pool_index = max(visible_documents, int(pool_start_rank) - 1)
    last_pool_index = min(len(results), int(pool_end_rank))
    pool = list(range(first_pool_index, last_pool_index))
    if len(pool) < visible_documents:
        raise ValueError(
            f"Need at least {visible_documents} replacement documents, found {len(pool)}"
        )
    visible_titles = {
        normalize_text(_document_parts(results[index])[0])
        for index in range(visible_documents)
    }
    available = [
        index
        for index in pool
        if normalize_text(_document_parts(results[index])[0]) not in visible_titles
    ]
    if len(available) < visible_documents:
        raise ValueError("Replacement pool has too few title-distinct documents")
    lengths = [document_token_length(item, tokenizer) for item in results]
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for slot in range(visible_documents):
        candidates = [index for index in available if index not in used]
        replacement_index = min(
            candidates,
            key=lambda index: (abs(lengths[index] - lengths[slot]), index),
        )
        used.add(replacement_index)
        replacement_title, _ = _document_parts(results[replacement_index])
        original_title, _ = _document_parts(results[slot])
        chosen.append(
            {
                "slot": slot,
                "original_rank": slot + 1,
                "original_title": original_title,
                "original_token_length": lengths[slot],
                "replacement_index": replacement_index,
                "replacement_rank": replacement_index + 1,
                "replacement_title": replacement_title,
                "replacement_token_length": lengths[replacement_index],
                "absolute_length_difference": abs(lengths[replacement_index] - lengths[slot]),
            }
        )
    return chosen


def apply_fixed_cardinality_swap(
    results: Sequence[Mapping[str, Any]],
    *,
    visible_documents: int,
    slot: int,
    replacement_index: int,
) -> list[dict[str, Any]]:
    visible = [dict(value) for value in results[:visible_documents]]
    if slot < 0 or slot >= len(visible):
        raise IndexError(slot)
    if replacement_index < visible_documents or replacement_index >= len(results):
        raise IndexError(replacement_index)
    visible[slot] = dict(results[replacement_index])
    return visible


def aggregate_swap_credit(seed_by_document: Sequence[Sequence[float]]) -> dict[str, Any]:
    array = np.asarray(seed_by_document, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("swap credit must have seed x document geometry")
    document_means = array.mean(axis=0)
    per_seed_signed_mean = array.mean(axis=1)
    per_seed_positive_sum = np.maximum(array, 0.0).sum(axis=1)
    return {
        "document_means": document_means.tolist(),
        "signed_mean": float(document_means.mean()),
        "positive_sum": float(np.maximum(document_means, 0.0).sum()),
        "signed_sum": float(document_means.sum()),
        "per_seed_signed_mean": per_seed_signed_mean.tolist(),
        "per_seed_positive_sum": per_seed_positive_sum.tolist(),
    }


def _preference(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def pairwise_preference_accuracy(
    truth: Sequence[float],
    score: Sequence[float],
    *,
    epsilon: float = 1e-9,
) -> float:
    if len(truth) != len(score) or len(truth) < 2:
        return float("nan")
    points: list[float] = []
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            target = _preference(float(truth[left]) - float(truth[right]), epsilon)
            prediction = _preference(float(score[left]) - float(score[right]), epsilon)
            if target == 0 or prediction == 0:
                points.append(0.5)
            else:
                points.append(float(target == prediction))
    return float(np.mean(points)) if points else float("nan")


def top1_regret(truth: Sequence[float], score: Sequence[float]) -> dict[str, float]:
    if len(truth) != len(score) or not truth:
        return {
            "regret": float("nan"),
            "normalized_regret": float("nan"),
            "agreement": float("nan"),
        }
    truth_array = np.asarray(truth, dtype=np.float64)
    score_array = np.asarray(score, dtype=np.float64)
    score_top = float(score_array.max())
    selected_set = set(np.flatnonzero(np.isclose(score_array, score_top)).tolist())
    optimum = float(truth_array.max())
    selected_truth = float(np.mean([truth_array[index] for index in selected_set]))
    regret = optimum - selected_truth
    value_range = float(truth_array.max() - truth_array.min())
    truth_top_set = set(np.flatnonzero(np.isclose(truth_array, optimum)).tolist())
    agreement = len(selected_set & truth_top_set) / max(1, len(selected_set))
    return {
        "regret": float(regret),
        "normalized_regret": 0.0 if value_range <= 1e-12 else float(regret / value_range),
        "agreement": float(agreement),
    }


def split_half_means(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 4:
        raise ValueError("At least four continuation seeds are required")
    return float(array[::2].mean()), float(array[1::2].mean())


def state_audit_metrics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    reward_view: str,
    document_signal: str,
    epsilon: float,
) -> dict[str, float]:
    truth: list[float] = []
    action_half_a: list[float] = []
    action_half_b: list[float] = []
    document: list[float] = []
    document_half_a: list[float] = []
    document_half_b: list[float] = []
    per_seed_key = {
        "signed_mean": "per_seed_signed_mean",
        "positive_sum": "per_seed_positive_sum",
    }.get(document_signal)
    if per_seed_key is None:
        raise ValueError(f"Unsupported document signal for cross-fitting: {document_signal}")
    for row in candidates:
        seed_rewards = [float(value) for value in row["full_seed_rewards"][reward_view]]
        first, second = split_half_means(seed_rewards)
        per_seed_document = [
            float(value)
            for value in row["swap_credit"][reward_view][per_seed_key]
        ]
        doc_first, doc_second = split_half_means(per_seed_document)
        truth.append(float(np.mean(seed_rewards)))
        action_half_a.append(first)
        action_half_b.append(second)
        document.append(float(row["swap_credit"][reward_view][document_signal]))
        document_half_a.append(doc_first)
        document_half_b.append(doc_second)
    action_self = pairwise_preference_accuracy(
        action_half_a, action_half_b, epsilon=epsilon
    )
    cross_a = pairwise_preference_accuracy(
        action_half_a, document_half_b, epsilon=epsilon
    )
    cross_b = pairwise_preference_accuracy(
        action_half_b, document_half_a, epsilon=epsilon
    )
    document_action = float(np.mean([cross_a, cross_b]))
    cross_regret_a = top1_regret(action_half_a, document_half_b)
    cross_regret_b = top1_regret(action_half_b, document_half_a)
    full_regret = top1_regret(truth, document)
    return {
        "candidate_count": float(len(candidates)),
        "action_self_pairwise": action_self,
        "document_action_pairwise": document_action,
        "reliability_gap": action_self - document_action,
        "within_state_spearman": float(
            np.mean(
                [
                    spearman(action_half_a, document_half_b),
                    spearman(action_half_b, document_half_a),
                ]
            )
        ),
        "regret": float(np.mean([cross_regret_a["regret"], cross_regret_b["regret"]])),
        "normalized_regret": float(
            np.mean(
                [
                    cross_regret_a["normalized_regret"],
                    cross_regret_b["normalized_regret"],
                ]
            )
        ),
        "agreement": float(
            np.mean([cross_regret_a["agreement"], cross_regret_b["agreement"]])
        ),
        "full_sample_spearman": spearman(truth, document),
        "full_sample_normalized_regret": float(full_regret["normalized_regret"]),
    }


def state_standardize(values: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    std = float(array.std())
    if std <= epsilon:
        return np.zeros_like(array)
    return (array - array.mean()) / (std + epsilon)


def match_rms(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    source_rms = float(np.sqrt(np.mean(array**2)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    if source_rms <= 1e-12 or target_rms <= 1e-12:
        return np.zeros_like(array) if target_rms <= 1e-12 else array
    return array * (target_rms / source_rms)


def shaped_signal(
    outcome: Sequence[float],
    document: Sequence[float],
    *,
    alpha: float,
) -> np.ndarray:
    base = state_standardize(outcome)
    bonus = state_standardize(document)
    combined = base + float(alpha) * bonus
    centered = combined - combined.mean()
    return match_rms(centered, base)


def cluster_mean_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if math.isfinite(value):
            grouped[str(row[cluster_key])].append(value)
    cluster_values = np.asarray(
        [float(np.mean(values)) for _, values in sorted(grouped.items())],
        dtype=np.float64,
    )
    if cluster_values.size == 0:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "clusters": 0.0,
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sample = rng.choice(cluster_values, size=len(cluster_values), replace=True)
        draws[index] = float(sample.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(cluster_values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "clusters": float(len(cluster_values)),
    }


def two_way_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed_key: str,
    item_key: str,
    value_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_seed_item: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if math.isfinite(value):
            by_seed_item[(str(row[seed_key]), str(row[item_key]))].append(value)
    seeds = sorted({key[0] for key in by_seed_item})
    items = sorted({key[1] for key in by_seed_item})
    if not seeds or not items:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "seeds": float(len(seeds)),
            "items": float(len(items)),
        }
    matrix = np.full((len(seeds), len(items)), np.nan, dtype=np.float64)
    seed_index = {value: index for index, value in enumerate(seeds)}
    item_index = {value: index for index, value in enumerate(items)}
    for (run_seed, item), values in by_seed_item.items():
        matrix[seed_index[run_seed], item_index[item]] = float(np.mean(values))
    observed = float(np.nanmean(matrix))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled_seed = rng.integers(0, len(seeds), size=len(seeds))
        sampled_item = rng.integers(0, len(items), size=len(items))
        draws[index] = float(np.nanmean(matrix[np.ix_(sampled_seed, sampled_item)]))
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size == 0:
        return {
            "estimate": observed,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "seeds": float(len(seeds)),
            "items": float(len(items)),
        }
    low, high = np.quantile(finite_draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "seeds": float(len(seeds)),
        "items": float(len(items)),
    }


def flatten(values: Iterable[Iterable[Any]]) -> list[Any]:
    return [item for group in values for item in group]

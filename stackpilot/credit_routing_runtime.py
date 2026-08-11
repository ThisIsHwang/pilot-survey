from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def aggregate_search_utilities(values: Any, mode: str) -> float:
    if values is None:
        return 0.0
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise RuntimeError(f"Search utility metadata must be a list, got {type(values)!r}")
    numbers = np.asarray([float(value) for value in values], dtype=np.float64)
    if numbers.size == 0:
        return 0.0
    if not np.isfinite(numbers).all():
        raise RuntimeError("Search utility metadata contains non-finite values")
    if mode == "mean":
        return float(numbers.mean())
    if mode == "max":
        return float(numbers.max())
    if mode == "sum":
        return float(numbers.sum())
    if mode == "last":
        return float(numbers[-1])
    raise ValueError(f"Unknown trajectory utility aggregation: {mode}")


def normalized_group_shaping(
    raw_values: Sequence[float],
    groups: Sequence[Any],
    *,
    coefficient: float,
    clip: float,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    values = np.asarray(raw_values, dtype=np.float64)
    if len(values) != len(groups):
        raise RuntimeError("Utility values and GRPO group IDs have different lengths")
    if not np.isfinite(values).all():
        raise RuntimeError("Action utility contains non-finite values")
    shaped = np.zeros(len(values), dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    positions: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        positions[str(group)].append(index)
    for uid, indices in positions.items():
        local = values[indices]
        mean = float(local.mean())
        std = float(local.std())
        if std <= epsilon:
            normalized = np.zeros(len(local), dtype=np.float64)
        else:
            normalized = (local - mean) / std
        normalized = np.clip(normalized, -float(clip), float(clip))
        local_shaping = float(coefficient) * normalized
        shaped[indices] = local_shaping
        diagnostics.append(
            {
                "uid": uid,
                "credit_routing_raw_mean": mean,
                "credit_routing_raw_std": std,
                "credit_routing_shaping_abs_mean": float(np.abs(local_shaping).mean()),
                "credit_routing_shaping_max": float(local_shaping.max(initial=0.0)),
                "credit_routing_shaping_min": float(local_shaping.min(initial=0.0)),
                "credit_routing_nonzero_fraction": float(np.mean(np.abs(local_shaping) > epsilon)),
            }
        )
    return shaped, diagnostics


def apply_action_utility_shaping(
    *,
    token_level_rewards: Any,
    eos_mask: Any,
    index: Sequence[Any],
    utility_batches: Sequence[Any] | None,
    route_mode: str,
    trajectory_aggregation: str,
    coefficient: float,
    clip: float,
) -> tuple[Any, dict[str, float], dict[str, dict[str, float]]]:
    if route_mode not in {"off", "document-utility"}:
        raise ValueError(f"Unknown action route mode: {route_mode}")
    if utility_batches is None:
        if route_mode == "document-utility":
            raise RuntimeError("Action routing requires search utility metadata")
        utility_batches = [[] for _ in range(len(index))]
    if len(utility_batches) != len(index):
        raise RuntimeError("Search utility metadata does not match reward batch")
    raw = [
        aggregate_search_utilities(values, trajectory_aggregation)
        for values in utility_batches
    ]
    if route_mode == "off":
        shaping = np.zeros(len(raw), dtype=np.float64)
        diagnostics = []
        by_group: dict[str, list[float]] = defaultdict(list)
        for uid, value in zip(index, raw, strict=True):
            by_group[str(uid)].append(float(value))
        for uid, values in by_group.items():
            local = np.asarray(values, dtype=np.float64)
            diagnostics.append(
                {
                    "uid": uid,
                    "credit_routing_raw_mean": float(local.mean()),
                    "credit_routing_raw_std": float(local.std()),
                    "credit_routing_shaping_abs_mean": 0.0,
                    "credit_routing_shaping_max": 0.0,
                    "credit_routing_shaping_min": 0.0,
                    "credit_routing_nonzero_fraction": 0.0,
                }
            )
    else:
        shaping, diagnostics = normalized_group_shaping(
            raw,
            index,
            coefficient=float(coefficient),
            clip=float(clip),
        )

    output = token_level_rewards.clone()
    if output.ndim != 2 or eos_mask.shape != output.shape:
        raise RuntimeError("Token rewards and EOS mask must have the same 2D shape")
    for row_index, value in enumerate(shaping):
        valid = eos_mask[row_index].nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            raise RuntimeError("Cannot route action utility to an empty response")
        output[row_index, int(valid[-1].item())] += float(value)
    if not bool(output.isfinite().all()):
        raise RuntimeError("Action routing produced non-finite token rewards")

    group_rows = {str(row["uid"]): row for row in diagnostics}
    metrics_accumulator: dict[str, list[float]] = defaultdict(list)
    for row in diagnostics:
        for key, value in row.items():
            if key == "uid":
                continue
            if math.isfinite(float(value)):
                metrics_accumulator[key].append(float(value))
    metrics = {
        f"credit_routing/{key}": float(np.mean(values))
        for key, values in metrics_accumulator.items()
        if values
    }
    metrics["credit_routing/action_route_enabled"] = float(route_mode == "document-utility")
    metrics["credit_routing/action_coefficient"] = float(coefficient)
    return output, metrics, group_rows

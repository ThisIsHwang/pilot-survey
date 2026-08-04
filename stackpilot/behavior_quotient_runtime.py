from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _normalize_title(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def behavior_signature(
    queries: Sequence[Any],
    title_batches: Sequence[Any],
    *,
    mode: str = "trajectory-ranked",
) -> str:
    normalized_queries = [str(query).strip() for query in (queries or [])]
    normalized_batches: list[list[str]] = []
    for batch in title_batches or []:
        if isinstance(batch, (list, tuple, np.ndarray)):
            normalized_batches.append([_normalize_title(value) for value in list(batch)])
        else:
            normalized_batches.append([])
    if mode == "trajectory-ranked":
        payload: Any = normalized_batches if normalized_batches else [["<no-search>"]]
    elif mode == "trajectory-unordered":
        payload = [sorted(set(batch)) for batch in normalized_batches] or [["<no-search>"]]
    elif mode == "last-ranked":
        payload = normalized_batches[-1] if normalized_batches else ["<no-search>"]
    elif mode == "query-text":
        payload = normalized_queries if normalized_queries else ["<no-search>"]
    else:
        raise ValueError(f"Unknown behavior signature mode: {mode}")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def behavior_signatures(
    query_batches: Sequence[Any],
    title_batches: Sequence[Any],
    *,
    mode: str,
) -> list[str]:
    if len(query_batches) != len(title_batches):
        raise RuntimeError(
            "search-query and title-batch metadata have different batch lengths: "
            f"{len(query_batches)} != {len(title_batches)}"
        )
    return [
        behavior_signature(queries, titles, mode=mode)
        for queries, titles in zip(query_batches, title_batches)
    ]


def _stable_order(seed: int, *parts: Any) -> str:
    text = "\n".join([str(seed), *map(str, parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample_indices(
    group_indices: list[int],
    signatures: Sequence[str],
    *,
    selection_mode: str,
    update_per_prompt: int,
    seed: int,
    uid: str,
) -> list[int]:
    if update_per_prompt <= 0 or update_per_prompt >= len(group_indices):
        return list(group_indices)
    budget = int(update_per_prompt)
    if selection_mode in {"all", "surface-all"}:
        return list(group_indices)
    if selection_mode in {"surface-random", "random"}:
        return sorted(
            group_indices,
            key=lambda index: _stable_order(seed, uid, index, signatures[index]),
        )[:budget]
    if selection_mode not in {"behavior-balanced", "balanced"}:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    classes: dict[str, list[int]] = defaultdict(list)
    for index in group_indices:
        classes[str(signatures[index])].append(index)
    ordered_classes = sorted(classes, key=lambda value: _stable_order(seed, uid, value))
    for class_signature in ordered_classes:
        classes[class_signature].sort(
            key=lambda index: _stable_order(seed, uid, class_signature, index)
        )
    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for class_signature in ordered_classes:
            members = classes[class_signature]
            if depth < len(members):
                selected.append(members[depth])
                added = True
                if len(selected) == budget:
                    break
        if not added:
            break
        depth += 1
    return selected


def _surface_advantages(scores: torch.Tensor, selected: list[int], epsilon: float) -> dict[int, float]:
    if len(selected) <= 1:
        return {index: 0.0 for index in selected}
    selected_scores = scores[selected].detach().float()
    standard = torch.std(selected_scores, unbiased=True)
    if not torch.isfinite(standard) or float(standard.item()) <= epsilon:
        return {index: 0.0 for index in selected}
    normalized = (selected_scores - selected_scores.mean()) / (standard + epsilon)
    return {index: float(value.item()) for index, value in zip(selected, normalized)}


def _quotient_advantages(
    scores: torch.Tensor,
    group_indices: list[int],
    selected: list[int],
    signatures: Sequence[str],
    epsilon: float,
) -> dict[int, float]:
    all_classes: dict[str, list[int]] = defaultdict(list)
    selected_classes: dict[str, list[int]] = defaultdict(list)
    for index in group_indices:
        all_classes[str(signatures[index])].append(index)
    for index in selected:
        selected_classes[str(signatures[index])].append(index)
    class_names = sorted(selected_classes)
    if len(class_names) <= 1:
        return {index: 0.0 for index in selected}
    class_rewards = torch.stack(
        [scores[all_classes[class_name]].detach().float().mean() for class_name in class_names]
    )
    standard = torch.std(class_rewards, unbiased=True)
    if not torch.isfinite(standard) or float(standard.item()) <= epsilon:
        return {index: 0.0 for index in selected}
    class_values = (class_rewards - class_rewards.mean()) / (standard + epsilon)
    group_size = len(group_indices)
    number_of_classes = len(class_names)
    output: dict[int, float] = {}
    for class_name, class_value in zip(class_names, class_values):
        members = selected_classes[class_name]
        member_scale = group_size / (number_of_classes * len(members))
        for index in members:
            output[index] = float(class_value.item()) * member_scale
    return output


def _class_capped_advantages(
    scores: torch.Tensor,
    group_indices: list[int],
    selected: list[int],
    signatures: Sequence[str],
    epsilon: float,
) -> dict[int, float]:
    surface = _surface_advantages(scores, selected, epsilon)
    classes: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        classes[str(signatures[index])].append(index)
    if not classes:
        return {}
    group_size = len(group_indices)
    number_of_classes = len(classes)
    output: dict[int, float] = {}
    for members in classes.values():
        scale = group_size / (number_of_classes * len(members))
        for index in members:
            output[index] = surface[index] * scale
    return output


def _effective_behavior_count(signatures: Sequence[str], indices: list[int]) -> float:
    counts: dict[str, int] = defaultdict(int)
    for index in indices:
        counts[str(signatures[index])] += 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
    return float(1.0 / np.square(probabilities).sum())


def compute_behavior_advantages(
    *,
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: Sequence[Any],
    query_batches: Sequence[Any],
    title_batches: Sequence[Any],
    advantage_mode: str = "surface",
    selection_mode: str = "all",
    update_per_prompt: int = 0,
    signature_mode: str = "trajectory-ranked",
    seed: int = 0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], list[dict[str, Any]], list[str], torch.Tensor]:
    if token_level_rewards.ndim != 2 or eos_mask.shape != token_level_rewards.shape:
        raise RuntimeError("token rewards and EOS mask must have identical 2D shapes")
    scores = token_level_rewards.sum(dim=-1).detach().float()
    if len(index) != scores.shape[0]:
        raise RuntimeError("GRPO uid metadata does not match reward batch")
    signatures = behavior_signatures(query_batches, title_batches, mode=signature_mode)
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, uid in enumerate(index):
        groups[str(uid)].append(row_index)

    row_advantages = torch.zeros_like(scores, dtype=torch.float32)
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)
    telemetry: list[dict[str, Any]] = []
    summary_accumulator: dict[str, list[float]] = defaultdict(list)
    for uid, group_indices in groups.items():
        selected = _sample_indices(
            group_indices,
            signatures,
            selection_mode=selection_mode,
            update_per_prompt=int(update_per_prompt),
            seed=int(seed),
            uid=uid,
        )
        if advantage_mode == "surface":
            values = _surface_advantages(scores, selected, epsilon)
            if selected and len(selected) < len(group_indices):
                scale = len(group_indices) / len(selected)
                values = {key: value * scale for key, value in values.items()}
        elif advantage_mode == "quotient":
            values = _quotient_advantages(
                scores, group_indices, selected, signatures, epsilon
            )
        elif advantage_mode == "class-capped":
            values = _class_capped_advantages(
                scores, group_indices, selected, signatures, epsilon
            )
        else:
            raise ValueError(f"Unknown advantage mode: {advantage_mode}")
        for row_index in selected:
            selected_mask[row_index] = True
        for row_index, value in values.items():
            row_advantages[row_index] = float(value)

        class_members: dict[str, list[int]] = defaultdict(list)
        for row_index in group_indices:
            class_members[signatures[row_index]].append(row_index)
        class_sizes = [len(members) for members in class_members.values()]
        class_rewards = [
            float(scores[members].mean().item()) for members in class_members.values()
        ]
        surface_best = max(
            class_members,
            key=lambda class_name: float(scores[class_members[class_name]].mean().item()),
        )
        selected_classes = {signatures[row_index] for row_index in selected}
        behavior_count = len(class_members)
        alias_fraction = 1.0 - behavior_count / max(1, len(group_indices))
        selected_behavior_coverage = len(selected_classes) / max(1, behavior_count)
        telemetry_row = {
            "uid": uid,
            "surface_count": len(group_indices),
            "behavior_count": behavior_count,
            "alias_fraction": alias_fraction,
            "effective_behavior_count": _effective_behavior_count(signatures, group_indices),
            "largest_class_share": max(class_sizes) / max(1, len(group_indices)),
            "selected_count": len(selected),
            "selected_behavior_count": len(selected_classes),
            "selected_behavior_coverage": selected_behavior_coverage,
            "selected_duplicate_rate": 1.0 - len(selected_classes) / max(1, len(selected)),
            "surface_reward_variance": float(scores[group_indices].var(unbiased=False).item()),
            "class_reward_variance": float(np.var(class_rewards)) if class_rewards else 0.0,
            "nonzero_advantage_fraction": float(
                np.mean([abs(values.get(row_index, 0.0)) > epsilon for row_index in group_indices])
            ),
            "surface_best_class": surface_best,
            "class_sizes": sorted(class_sizes, reverse=True),
            "class_rewards": class_rewards,
            "selection_mode": selection_mode,
            "advantage_mode": advantage_mode,
            "signature_mode": signature_mode,
        }
        telemetry.append(telemetry_row)
        for key, value in telemetry_row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summary_accumulator[key].append(float(value))

    response_length = token_level_rewards.shape[-1]
    advantages = row_advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
    summary = {
        f"behavior_quotient/{key}": float(np.mean(values))
        for key, values in summary_accumulator.items()
        if values
    }
    summary["behavior_quotient/groups"] = float(len(groups))
    summary["behavior_quotient/selected_rows"] = float(
        sum(int(row["selected_count"]) for row in telemetry)
    )
    return (
        advantages,
        advantages.clone(),
        summary,
        telemetry,
        signatures,
        selected_mask,
    )


def select_behavior_rows(data: Any) -> Any:
    """Return the actor-update DataProto containing only preregistered rows.

    Rollout and reward computation remain unchanged. Physical row selection is
    applied only after advantages are computed, avoiding all-zero loss-mask
    microbatches and ensuring that PPO, entropy, and actor KL use the same K rows.
    """
    selected_mask = data.batch.get("stackpilot_bq_selected_mask")
    if selected_mask is None:
        return data
    selected_mask = selected_mask.detach().bool().cpu()
    if selected_mask.ndim != 1 or selected_mask.numel() != len(data):
        raise RuntimeError("behavior-quotient selected-row mask does not match batch")
    indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("behavior-quotient selection produced zero actor rows")
    if indices.numel() == len(data):
        return data
    from verl import DataProto

    numpy_indices = indices.numpy()
    selected = DataProto(
        batch=data.batch[indices],
        non_tensor_batch={
            key: value[numpy_indices] for key, value in data.non_tensor_batch.items()
        },
        meta_info=dict(data.meta_info),
    )
    selected.meta_info["stackpilot_bq_original_rows"] = int(len(data))
    selected.meta_info["stackpilot_bq_actor_rows"] = int(len(selected))
    return selected


def append_behavior_telemetry(
    path: str | Path,
    *,
    global_step: int,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    with destination.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for row in rows:
            payload = {"global_step": int(global_step), **metadata, **dict(row)}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

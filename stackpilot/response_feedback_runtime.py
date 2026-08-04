from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

FEEDBACK_OPEN = "<rollout_feedback>"
FEEDBACK_CLOSE = "</rollout_feedback>"


def normalize_title(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def stable_prompt_key(token_ids: Sequence[int], pad_token_id: int) -> str:
    values = [int(value) for value in token_ids if int(value) != int(pad_token_id)]
    digest = hashlib.sha256(",".join(map(str, values)).encode("utf-8")).hexdigest()
    return digest[:24]


def phase_indices(group_keys: Sequence[str], first_count: int) -> tuple[list[int], list[int]]:
    if first_count <= 0:
        raise ValueError("first_count must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        groups[str(key)].append(index)
    first: list[int] = []
    second: list[int] = []
    for key in sorted(groups, key=lambda value: min(groups[value])):
        members = groups[key]
        first.extend(members[:first_count])
        second.extend(members[first_count:])
    return first, second


def flatten_title_batches(value: Any) -> list[str]:
    output: list[str] = []
    if not isinstance(value, (list, tuple, np.ndarray)):
        return output
    for batch in value:
        if not isinstance(batch, (list, tuple, np.ndarray)):
            continue
        for title in batch:
            text = " ".join(str(title).split())
            if text:
                output.append(text)
    return output


def feedback_title_map(
    *,
    group_keys: Sequence[str],
    first_indices: Sequence[int],
    first_title_batches: Sequence[Any],
    maximum_titles: int,
) -> dict[str, list[str]]:
    if len(first_indices) != len(first_title_batches):
        raise RuntimeError("first-phase indices and title metadata have different lengths")
    output: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for original_index, title_batches in zip(first_indices, first_title_batches):
        key = str(group_keys[int(original_index)])
        for title in flatten_title_batches(title_batches):
            normalized = normalize_title(title)
            if not normalized or normalized in seen[key]:
                continue
            seen[key].add(normalized)
            output[key].append(title)
            if len(output[key]) >= int(maximum_titles):
                break
    return dict(output)


def feedback_text(
    titles: Sequence[str],
    *,
    maximum_chars: int,
    mode: str = "response-feedback",
) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for title in titles:
        text = " ".join(str(title).split())
        normalized = normalize_title(text)
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(text)
    if mode == "text-feedback":
        attempted = "; ".join(cleaned) if cleaned else "(no valid query wording was produced)"
        instruction = (
            "Sibling rollouts for this same question already attempted these query "
            f"wordings: {attempted}. Generate a substantially different next search "
            "query. Preserve the information need, but avoid merely paraphrasing an "
            "already attempted query."
        )
    else:
        visible = "; ".join(cleaned) if cleaned else "(no document titles were returned)"
        instruction = (
            "Sibling rollouts for this same question already retrieved these visible "
            f"document titles: {visible}. Generate a next search query that seeks a "
            "different retrieval outcome when useful. Do not repeat a query solely to "
            "retrieve the same titles. These titles are observations, not gold labels."
        )
    body = f"\n\n{FEEDBACK_OPEN}\n{instruction}\n{FEEDBACK_CLOSE}\n\n"
    if len(body) > int(maximum_chars):
        body = body[: max(0, int(maximum_chars) - len(FEEDBACK_CLOSE) - 2)].rstrip()
        body += f"\n{FEEDBACK_CLOSE}\n\n"
    return body


def _select_proto(data: Any, indices: Sequence[int]) -> Any:
    import torch
    from verl import DataProto

    tensor_indices = torch.as_tensor(list(indices), dtype=torch.long)
    numpy_indices = np.asarray(list(indices), dtype=np.int64)
    tensors = {key: value[tensor_indices] for key, value in data.batch.items()}
    non_tensors = {
        key: value[numpy_indices] for key, value in data.non_tensor_batch.items()
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=dict(data.meta_info),
    )


def _left_pad_rows(rows: Sequence[Any], pad_value: int) -> Any:
    import torch

    width = max((int(row.numel()) for row in rows), default=0)
    if not rows:
        return torch.empty((0, 0), dtype=torch.long)
    output = torch.full(
        (len(rows), width),
        int(pad_value),
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    for index, row in enumerate(rows):
        output[index, width - row.numel() :] = row
    return output


def _right_pad_rows(rows: Sequence[Any], pad_value: int) -> Any:
    import torch

    width = max((int(row.numel()) for row in rows), default=0)
    if not rows:
        return torch.empty((0, 0), dtype=torch.long)
    output = torch.full(
        (len(rows), width),
        int(pad_value),
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
    return output


def augment_prompt_batch(
    generation_manager: Any,
    gen_batch: Any,
    initial_input_ids: Any,
    feedback_texts: Sequence[str],
    *,
    prompt_token_budget: int,
) -> tuple[Any, Any]:
    import torch
    from verl import DataProto

    if len(feedback_texts) != int(initial_input_ids.shape[0]):
        raise RuntimeError("feedback text count does not match second-phase prompt batch")
    tokenizer = generation_manager.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    combined_rows = []
    for row, text in zip(initial_input_ids, feedback_texts):
        prompt = row[row != pad_id]
        feedback_ids = tokenizer(
            str(text), add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(device=prompt.device, dtype=prompt.dtype)
        combined = torch.cat([prompt, feedback_ids], dim=0)
        combined = combined[-int(prompt_token_budget) :]
        combined_rows.append(combined)
    input_ids = _left_pad_rows(combined_rows, pad_id)
    attention_mask = generation_manager.tensor_fn.create_attention_mask(input_ids)
    position_ids = generation_manager.tensor_fn.create_position_ids(attention_mask)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    augmented = DataProto.from_dict(tensors=tensors, meta_info=dict(gen_batch.meta_info))
    return augmented, input_ids


def _merge_meta_info(
    first: dict[str, Any],
    second: dict[str, Any],
    first_indices: Sequence[int],
    second_indices: Sequence[int],
    total: int,
) -> dict[str, Any]:
    output = dict(first)
    for key, value in second.items():
        if key not in output:
            output[key] = value
    for key in set(first) | set(second):
        left = first.get(key)
        right = second.get(key)
        if not isinstance(left, list) or not isinstance(right, list):
            continue
        if len(left) != len(first_indices) or len(right) != len(second_indices):
            continue
        merged: list[Any] = [None] * total
        for original, item in zip(first_indices, left):
            merged[int(original)] = item
        for original, item in zip(second_indices, right):
            merged[int(original)] = item
        output[key] = merged
    return output


def _merge_outputs(
    generation_manager: Any,
    first_output: Any,
    second_output: Any,
    first_indices: Sequence[int],
    second_indices: Sequence[int],
    feedback_by_group: dict[str, list[str]],
    group_keys: Sequence[str],
) -> Any:
    import torch
    from verl import DataProto

    total = len(first_indices) + len(second_indices)
    if total != len(group_keys):
        raise RuntimeError("phase outputs do not cover the full rollout batch")
    pad_id = int(generation_manager.tokenizer.pad_token_id)

    def ordered_rows(key: str, *, strip_padding: bool) -> list[Any]:
        rows: list[Any] = [None] * total
        for output, indices in ((first_output, first_indices), (second_output, second_indices)):
            tensor = output.batch.get(key)
            if tensor is None:
                raise RuntimeError(f"generation output is missing tensor {key!r}")
            for original, row in zip(indices, tensor):
                value = row
                if strip_padding:
                    value = value[value != pad_id]
                rows[int(original)] = value
        if any(row is None for row in rows):
            raise RuntimeError(f"generation output did not populate all rows for {key}")
        return rows

    prompt_rows = ordered_rows("prompts", strip_padding=True)
    response_rows = ordered_rows("responses", strip_padding=True)
    info_response_rows = ordered_rows("responses_with_info_mask", strip_padding=False)
    # Observation tokens are deliberately replaced by pad IDs in the info mask.
    # Their positions still belong to the response and must not be trimmed by a
    # non-pad search. Use the unmasked response length as the authoritative width.
    trimmed_info_rows = [
        info_row[: int(response_row.numel())]
        for info_row, response_row in zip(info_response_rows, response_rows)
    ]

    prompts = _left_pad_rows(prompt_rows, pad_id)
    responses = _right_pad_rows(response_rows, pad_id)
    responses_with_info_mask = _right_pad_rows(trimmed_info_rows, pad_id)
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat(
        [
            generation_manager.tensor_fn.create_attention_mask(prompts),
            generation_manager.tensor_fn.create_attention_mask(responses),
        ],
        dim=1,
    )
    info_mask = torch.cat(
        [
            generation_manager.tensor_fn.create_attention_mask(prompts),
            generation_manager.tensor_fn.create_attention_mask(responses_with_info_mask),
        ],
        dim=1,
    )
    position_ids = generation_manager.tensor_fn.create_position_ids(attention_mask)
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "responses_with_info_mask": responses_with_info_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "info_mask": info_mask,
        "position_ids": position_ids,
    }

    all_non_tensor_keys = set(first_output.non_tensor_batch) | set(second_output.non_tensor_batch)
    non_tensors: dict[str, np.ndarray] = {}
    first_lookup = {int(original): offset for offset, original in enumerate(first_indices)}
    second_lookup = {int(original): offset for offset, original in enumerate(second_indices)}
    for key in all_non_tensor_keys:
        merged = np.empty(total, dtype=object)
        for original in range(total):
            if original in first_lookup and key in first_output.non_tensor_batch:
                merged[original] = first_output.non_tensor_batch[key][first_lookup[original]]
            elif original in second_lookup and key in second_output.non_tensor_batch:
                merged[original] = second_output.non_tensor_batch[key][second_lookup[original]]
            else:
                merged[original] = None
        non_tensors[key] = merged

    phases = np.empty(total, dtype=object)
    feedback_titles = np.empty(total, dtype=object)
    for index in range(total):
        is_second = index in second_lookup
        phases[index] = "feedback" if is_second else "initial"
        feedback_titles[index] = list(feedback_by_group.get(str(group_keys[index]), [])) if is_second else []
    non_tensors["stackpilot_feedback_phase"] = phases
    non_tensors["stackpilot_feedback_titles"] = feedback_titles

    meta_info = _merge_meta_info(
        dict(first_output.meta_info),
        dict(second_output.meta_info),
        first_indices,
        second_indices,
        total,
    )
    meta_info["stackpilot_response_feedback"] = True
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)


def run_grouped_feedback_rollouts(
    *,
    generation_manager: Any,
    gen_batch: Any,
    initial_input_ids: Any,
    mode: str = "iid",
    first_count: int = 4,
    maximum_titles: int = 24,
    maximum_chars: int = 1800,
    prompt_token_budget: int = 2048,
) -> Any:
    """Generate IID or two-phase response-feedback rollouts.

    The feedback phase receives only document titles observed by sibling
    first-phase rollouts for the exact same original prompt. It never receives
    support annotations, answers, rewards, or hidden retriever metadata.
    """

    if mode in {"iid", "none", "off", ""}:
        return generation_manager.run_llm_loop(
            gen_batch=gen_batch, initial_input_ids=initial_input_ids
        )
    if mode != "response-feedback":
        raise ValueError(f"Unknown rollout feedback mode: {mode}")

    pad_id = int(generation_manager.tokenizer.pad_token_id)
    group_keys = [
        stable_prompt_key(row.tolist(), pad_id) for row in initial_input_ids
    ]
    first_indices, second_indices = phase_indices(group_keys, int(first_count))
    if not second_indices:
        return generation_manager.run_llm_loop(
            gen_batch=gen_batch, initial_input_ids=initial_input_ids
        )

    first_batch = _select_proto(gen_batch, first_indices)
    first_prompts = initial_input_ids[first_indices]
    first_output = generation_manager.run_llm_loop(
        gen_batch=first_batch,
        initial_input_ids=first_prompts,
    )
    title_metadata = first_output.non_tensor_batch.get(
        "stackpilot_search_title_batches"
    )
    if title_metadata is None:
        raise RuntimeError(
            "response-feedback rollout requires structured ranked-title metadata"
        )
    feedback_by_group = feedback_title_map(
        group_keys=group_keys,
        first_indices=first_indices,
        first_title_batches=list(title_metadata),
        maximum_titles=int(maximum_titles),
    )

    second_batch = _select_proto(gen_batch, second_indices)
    second_prompts = initial_input_ids[second_indices]
    feedback_texts = [
        feedback_text(
            feedback_by_group.get(str(group_keys[index]), []),
            maximum_chars=int(maximum_chars),
        )
        for index in second_indices
    ]
    augmented_batch, augmented_prompts = augment_prompt_batch(
        generation_manager,
        second_batch,
        second_prompts,
        feedback_texts,
        prompt_token_budget=int(prompt_token_budget),
    )
    second_output = generation_manager.run_llm_loop(
        gen_batch=augmented_batch,
        initial_input_ids=augmented_prompts,
    )
    return _merge_outputs(
        generation_manager,
        first_output,
        second_output,
        first_indices,
        second_indices,
        feedback_by_group,
        group_keys,
    )

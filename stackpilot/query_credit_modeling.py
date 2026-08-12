from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from stackpilot.query_credit_common import state_standardize, stable_seed


def signal_values(rows: Sequence[Mapping[str, Any]], method: str, *, shuffle_seed: int = 0) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["state_id"])].append(index)
    output = np.zeros(len(rows), dtype=np.float64)
    rng = np.random.default_rng(shuffle_seed)
    for state_id, indices in grouped.items():
        state_rows = [rows[index] for index in indices]
        if method == "query-oracle":
            raw = [float(row["query_action_advantage"]) for row in state_rows]
        elif method == "doc-positive-sum":
            raw = [float(row["document_credit"]["positive-sum"]) for row in state_rows]
        elif method == "doc-alias-normalized":
            classes: dict[str, list[int]] = defaultdict(list)
            for local_index, row in enumerate(state_rows):
                classes[str(row.get("behavior_signature", local_index))].append(local_index)
            class_keys = sorted(classes)
            class_raw = [
                float(np.mean([
                    float(state_rows[local]["document_credit"]["positive-sum"])
                    for local in classes[key]
                ]))
                for key in class_keys
            ]
            class_standardized = state_standardize(class_raw)
            raw = [0.0] * len(state_rows)
            for class_index, key in enumerate(class_keys):
                members = classes[key]
                share = float(class_standardized[class_index]) / max(1, len(members))
                for local in members:
                    raw[local] = share
            for local_index, global_index in enumerate(indices):
                output[global_index] = raw[local_index]
            continue
        elif method == "shuffled-doc":
            raw = [float(row["document_credit"]["positive-sum"]) for row in state_rows]
            raw = list(np.asarray(raw)[rng.permutation(len(raw))])
        elif method == "outcome":
            raw = [float(row["full_reward"]) for row in state_rows]
        else:
            raise ValueError(f"Unknown query-credit signal: {method}")
        standardized = state_standardize(raw)
        for local_index, global_index in enumerate(indices):
            output[global_index] = standardized[local_index]
    return output.tolist()


def build_tokenized_example(tokenizer: Any, prefix_messages: list[dict[str, str]], query: str, maximum_length: int) -> dict[str, Any]:
    action = f"<search>{query}</search>"
    full_messages = copy.deepcopy(prefix_messages) + [{"role": "assistant", "content": action}]
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prefix_ids = [int(value) for value in prefix_ids]
    full_ids = [int(value) for value in full_ids]
    common = 0
    for left, right in zip(prefix_ids, full_ids):
        if left != right:
            break
        common += 1
    if common == 0:
        raise RuntimeError("Chat-template prefix and completion have no common prefix")
    dropped = max(0, len(full_ids) - maximum_length)
    full_ids = full_ids[dropped:]
    start = max(0, common - dropped)
    if start >= len(full_ids):
        raise RuntimeError("Maximum length removed the complete search action")
    labels = [-100] * len(full_ids)
    labels[start:] = full_ids[start:]
    return {"input_ids": full_ids, "labels": labels, "query": query}


def collate_examples(tokenizer: Any, examples: Sequence[Mapping[str, Any]], weights: Sequence[float], device: Any) -> dict[str, Any]:
    import torch

    maximum = max(len(example["input_ids"]) for example in examples)
    pad = int(tokenizer.pad_token_id)
    input_ids = []
    labels = []
    attention = []
    for example in examples:
        difference = maximum - len(example["input_ids"])
        input_ids.append([pad] * difference + list(example["input_ids"]))
        labels.append([-100] * difference + list(example["labels"]))
        attention.append([0] * difference + [1] * len(example["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def weighted_query_loss(model: Any, batch: Mapping[str, Any]) -> Any:
    import torch

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid
    denominators = valid.sum(dim=1).clamp_min(1)
    sequence_log_probs = token_log_probs.sum(dim=1) / denominators
    return -(batch["weights"] * sequence_log_probs).mean(), sequence_log_probs


def trainable_gradient_vector(model: Any) -> Any:
    import torch

    vectors = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            vectors.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            vectors.append(gradient.detach().float().cpu().reshape(-1))
    if not vectors:
        raise RuntimeError("No trainable gradients found")
    return torch.cat(vectors)


def cosine(left: Any, right: Any) -> float:
    import torch

    denominator = float(left.norm() * right.norm())
    return 0.0 if denominator <= 1e-12 else float(torch.dot(left, right) / denominator)


def load_lora_model(cfg: Mapping[str, Any], *, device: str = "cuda") -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg["model"]
    grad_cfg = cfg["gradient"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(grad_cfg["lora_rank"]),
        lora_alpha=int(grad_cfg["lora_alpha"]),
        lora_dropout=float(grad_cfg["lora_dropout"]),
        target_modules=[str(value) for value in grad_cfg["target_modules"]],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    return tokenizer, model


def batch_indices(length: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(length).tolist()
    return [order[offset : offset + batch_size] for offset in range(0, length, batch_size)]

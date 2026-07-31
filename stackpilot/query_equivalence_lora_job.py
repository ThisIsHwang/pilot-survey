from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.query_equivalence_common import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
)

SYSTEM_MESSAGE = (
    "You generate the next search query after observing prior retrieval results. "
    "Return only the query, with no explanation or XML tags."
)


@dataclass
class EncodedTarget:
    target_id: str
    input_ids: list[int]
    labels: list[int]
    target_tokens: int
    weight: float
    metadata: dict[str, Any]


@dataclass
class EncodedGroup:
    group_id: str
    targets: list[EncodedTarget]
    metadata: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {SYSTEM_MESSAGE}\nUser: {prompt}\nAssistant:"


def encode_target(
    tokenizer: Any,
    prompt: str,
    target: dict[str, Any],
    *,
    max_length: int,
) -> EncodedTarget:
    target_text = str(target["text"]).strip()
    if not target_text:
        raise RuntimeError(f"Empty target text for {target.get('target_id')}")
    prompt_ids = tokenizer.encode(_prompt_text(tokenizer, prompt), add_special_tokens=False)
    target_ids = tokenizer.encode(
        target_text + (tokenizer.eos_token or ""), add_special_tokens=False
    )
    if not target_ids:
        raise RuntimeError(f"No target tokens for {target.get('target_id')}")
    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1] + [tokenizer.eos_token_id]
    prompt_ids = prompt_ids[-max(1, max_length - len(target_ids)) :]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    weight = float(target.get("weight", 1.0))
    if not math.isfinite(weight) or weight <= 0.0:
        raise RuntimeError(f"Training weights must be finite and positive; got {weight}")
    metadata = {key: value for key, value in target.items() if key not in {"text", "weight"}}
    return EncodedTarget(
        target_id=str(target["target_id"]),
        input_ids=input_ids,
        labels=labels,
        target_tokens=len(target_ids),
        weight=weight,
        metadata=metadata,
    )


def encode_group(tokenizer: Any, row: dict[str, Any], *, max_length: int) -> EncodedGroup:
    raw_targets = row.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RuntimeError(f"Group {row.get('group_id')} has no targets")
    targets = [
        encode_target(tokenizer, str(row["prompt"]), target, max_length=max_length)
        for target in raw_targets
    ]
    weight_sum = sum(target.weight for target in targets)
    if abs(weight_sum - 1.0) > 1e-6:
        if not all(abs(target.weight - 1.0) <= 1e-8 for target in targets):
            raise RuntimeError(
                f"Group {row.get('group_id')} target weights must sum to 1 or all equal 1; got {weight_sum}"
            )
    metadata = {key: value for key, value in row.items() if key not in {"targets", "prompt"}}
    return EncodedGroup(group_id=str(row["group_id"]), targets=targets, metadata=metadata)


class GroupDataset:
    def __init__(self, groups: list[EncodedGroup]) -> None:
        if not groups:
            raise ValueError("GroupDataset is empty")
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> EncodedGroup:
        return self.groups[index]


def collate_groups(groups: list[EncodedGroup], pad_token_id: int) -> dict[str, Any]:
    import torch

    targets = [target for group in groups for target in group.targets]
    max_length = max(len(target.input_ids) for target in targets)
    input_ids = []
    labels = []
    attention_mask = []
    group_indices = []
    weights = []
    target_ids = []
    target_tokens = []
    target_metadata = []
    group_ids = []
    group_metadata = []
    for group_index, group in enumerate(groups):
        group_ids.append(group.group_id)
        group_metadata.append(group.metadata)
        for target in group.targets:
            padding = max_length - len(target.input_ids)
            input_ids.append(target.input_ids + [pad_token_id] * padding)
            labels.append(target.labels + [-100] * padding)
            attention_mask.append([1] * len(target.input_ids) + [0] * padding)
            group_indices.append(group_index)
            weights.append(target.weight)
            target_ids.append(target.target_id)
            target_tokens.append(target.target_tokens)
            target_metadata.append(target.metadata)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "group_indices": torch.tensor(group_indices, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "group_ids": group_ids,
        "group_metadata": group_metadata,
        "target_ids": target_ids,
        "target_tokens": target_tokens,
        "target_metadata": target_metadata,
    }


def per_target_nll(model: Any, batch: dict[str, Any], *, grad: bool) -> Any:
    import torch
    import torch.nn.functional as functional

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    attention_mask = batch["attention_mask"].cuda(non_blocking=True)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        active = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~active, 0)
        token_losses = functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape_as(safe_labels)
        counts = active.sum(dim=1).clamp_min(1)
        losses = (token_losses * active).sum(dim=1) / counts
    return losses, counts


def grouped_loss(losses: Any, batch: dict[str, Any]) -> Any:
    import torch

    indices = batch["group_indices"].cuda(non_blocking=True)
    weights = batch["weights"].cuda(non_blocking=True)
    group_losses = []
    for group_index in range(len(batch["group_ids"])):
        mask = indices.eq(group_index)
        group_weights = weights[mask]
        group_weights = group_weights / group_weights.sum().clamp_min(1e-8)
        group_losses.append((group_weights * losses[mask]).sum())
    return torch.stack(group_losses).mean()


def evaluate(
    model: Any,
    dataset: GroupDataset,
    *,
    pad_token_id: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_groups(rows, pad_token_id),
    )
    model.eval()
    output = []
    for batch in loader:
        losses, counts = per_target_nll(model, batch, grad=False)
        for index, (target_id, loss, count, metadata) in enumerate(
            zip(
                batch["target_ids"],
                losses.detach().cpu().numpy(),
                counts.detach().cpu().numpy(),
                batch["target_metadata"],
                strict=True,
            )
        ):
            group_index = int(batch["group_indices"][index])
            row = {
                "group_id": str(batch["group_ids"][group_index]),
                "target_id": str(target_id),
                "nll": float(loss),
                "target_tokens": int(count),
                **batch["group_metadata"][group_index],
                **metadata,
            }
            output.append(row)
    return output


def train(model: Any, dataset: GroupDataset, tokenizer: Any, job: dict[str, Any]) -> dict[str, float]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    lora_cfg = job["lora"]
    batch_size = int(lora_cfg["batch_size"])
    accumulation = int(lora_cfg["gradient_accumulation"])
    max_steps = int(job["max_steps"])
    generator = torch.Generator().manual_seed(int(job["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        collate_fn=lambda rows: collate_groups(rows, tokenizer.pad_token_id),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs = {
        "lr": float(lora_cfg["learning_rate"]),
        "weight_decay": float(lora_cfg["weight_decay"]),
    }
    try:
        optimizer = AdamW(parameters, fused=True, **optimizer_kwargs)
    except TypeError:
        optimizer = AdamW(parameters, **optimizer_kwargs)
    warmup_steps = int(round(max_steps * float(lora_cfg["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cycle = itertools.cycle(loader)
    optimizer_step = 0
    micro_step = 0
    total_loss = 0.0
    processed_targets = 0
    started = time.time()
    while optimizer_step < max_steps:
        batch = next(cycle)
        losses, _counts = per_target_nll(model, batch, grad=True)
        state_loss = grouped_loss(losses, batch)
        if not torch.isfinite(state_loss):
            raise RuntimeError(f"Non-finite training loss at micro step {micro_step}: {state_loss}")
        (state_loss / accumulation).backward()
        total_loss += float(state_loss.detach().cpu())
        processed_targets += len(batch["target_ids"])
        micro_step += 1
        if micro_step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(parameters, float(lora_cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
    torch.cuda.synchronize()
    return {
        "optimizer_steps": float(optimizer_step),
        "micro_steps": float(micro_step),
        "mean_group_loss": total_loss / max(1, micro_step),
        "processed_targets": float(processed_targets),
        "elapsed_seconds": time.time() - started,
    }


def _validate_finite(payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite metric {key}: {value}")


def run_job(job_path: Path, *, force: bool) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]:
            print(f"Reusing completed query-equivalence job: {job['job_id']}")
            return
        stale = output_dir.with_name(f"{output_dir.name}.stale.{int(time.time())}")
        output_dir.rename(stale)
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_path = output_dir / "invalid.json"
    if invalid_path.exists():
        invalid_path.unlink()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Each query-equivalence job requires exactly one CUDA_VISIBLE_DEVICES entry")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA GPU; found {torch.cuda.device_count()}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"CUDA 12.9 wheel required; found {torch.version.cuda}")
    set_seed(int(job["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    local_only = Path(str(job["base_model"])).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        job["base_model"],
        trust_remote_code=bool(job.get("trust_remote_code", False)),
        local_files_only=local_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        job["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=bool(job.get("trust_remote_code", False)),
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    if bool(job["lora"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(job["lora"]["rank"]),
            lora_alpha=int(job["lora"]["alpha"]),
            lora_dropout=float(job["lora"]["dropout"]),
            target_modules=list(job["lora"]["target_modules"]),
            bias="none",
        ),
    )
    max_length = int(job["max_length"])
    train_groups = [
        encode_group(tokenizer, row, max_length=max_length) for row in read_jsonl(job["train_file"])
    ]
    eval_groups = [
        encode_group(tokenizer, row, max_length=max_length) for row in read_jsonl(job["eval_file"])
    ]
    train_dataset = GroupDataset(train_groups)
    eval_dataset = GroupDataset(eval_groups)
    try:
        baseline = evaluate(
            model,
            eval_dataset,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=int(job["lora"]["eval_batch_size"]),
        )
        train_metrics = train(model, train_dataset, tokenizer, job)
        adapted = evaluate(
            model,
            eval_dataset,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=int(job["lora"]["eval_batch_size"]),
        )
        adapted_by_id = {
            (row["group_id"], row["target_id"]): row for row in adapted
        }
        losses = []
        for base_row in baseline:
            key = (base_row["group_id"], base_row["target_id"])
            after = adapted_by_id[key]
            row = {
                **{key: value for key, value in base_row.items() if key != "nll"},
                "job_id": job["job_id"],
                "experiment_id": job["experiment_id"],
                "variant": job["variant"],
                "seed": int(job["seed"]),
                "baseline_nll": float(base_row["nll"]),
                "adapted_nll": float(after["nll"]),
                "heldout_gain": float(base_row["nll"] - after["nll"]),
            }
            _validate_finite(row)
            losses.append(row)
        baseline_mean = float(np.mean([row["baseline_nll"] for row in losses]))
        adapted_mean = float(np.mean([row["adapted_nll"] for row in losses]))
        metrics = {
            "schema": SCHEMA,
            "job_id": job["job_id"],
            "job_signature": job["job_signature"],
            "experiment_id": job["experiment_id"],
            "direction": job["direction"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "profile": job["profile"],
            "baseline_nll": baseline_mean,
            "adapted_nll": adapted_mean,
            "heldout_gain": baseline_mean - adapted_mean,
            "eval_targets": len(losses),
            "train_groups": len(train_groups),
            "train_targets": sum(len(group.targets) for group in train_groups),
            "train_file_sha256": file_sha256(job["train_file"]),
            "eval_file_sha256": file_sha256(job["eval_file"]),
            "metadata": job["metadata"],
            **train_metrics,
        }
        _validate_finite(metrics)
        atomic_write_jsonl(output_dir / "eval_losses.jsonl", losses)
        atomic_write_json(metrics_path, metrics)
    except Exception as exc:
        if metrics_path.exists():
            metrics_path.unlink()
        atomic_write_json(
            invalid_path,
            {
                "job_id": job.get("job_id"),
                "job_signature": job.get("job_signature"),
                "error": f"{type(exc).__name__}: {exc}",
                "time": time.time(),
            },
        )
        raise
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EXP-015 grouped LoRA micro-update.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_job(Path(args.job).resolve(), force=args.force)


if __name__ == "__main__":
    main()

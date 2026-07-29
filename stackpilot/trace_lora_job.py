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

from stackpilot.trace_common import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    file_sha256,
    read_jsonl,
)


SYSTEM_MESSAGE = (
    "You generate the next search query after observing prior retrieval results. "
    "Return only the query, with no explanation or XML tags."
)


@dataclass
class EncodedExample:
    example_id: str
    input_ids: list[int]
    labels: list[int]
    target_tokens: int
    weight: float


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


def encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
) -> EncodedExample:
    prompt_text = _prompt_text(tokenizer, str(row["prompt"]))
    target_text = str(row["target"]).strip()
    if not target_text:
        raise RuntimeError(f"TRACE example {row.get('example_id')} has an empty target")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(
        target_text + (tokenizer.eos_token or ""), add_special_tokens=False
    )
    if not target_ids:
        raise RuntimeError(f"TRACE example {row.get('example_id')} produced no target tokens")
    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1] + [tokenizer.eos_token_id]
    available_prompt = max(1, max_length - len(target_ids))
    prompt_ids = prompt_ids[-available_prompt:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return EncodedExample(
        example_id=str(row["example_id"]),
        input_ids=input_ids,
        labels=labels,
        target_tokens=len(target_ids),
        weight=float(row.get("weight", 1.0)),
    )


class TraceDataset:
    def __init__(self, examples: list[EncodedExample]) -> None:
        if not examples:
            raise ValueError("TRACE dataset is empty")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def collate(examples: list[EncodedExample], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_length = max(len(example.input_ids) for example in examples)
    input_ids = []
    labels = []
    attention_mask = []
    example_ids = []
    target_tokens = []
    weights = []
    for example in examples:
        padding = max_length - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_token_id] * padding)
        labels.append(example.labels + [-100] * padding)
        attention_mask.append([1] * len(example.input_ids) + [0] * padding)
        example_ids.append(example.example_id)
        target_tokens.append(example.target_tokens)
        weights.append(example.weight)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "example_ids": example_ids,
        "target_tokens": target_tokens,
        "weights": torch.tensor(weights, dtype=torch.float32),
    }


def per_example_nll(model: Any, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    attention_mask = batch["attention_mask"].cuda(non_blocking=True)
    with torch.no_grad():
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
    token_losses = token_losses * active
    counts = active.sum(dim=1).clamp_min(1)
    losses = token_losses.sum(dim=1) / counts
    return losses.cpu().numpy(), counts.cpu().numpy()


def evaluate(
    model: Any,
    dataset: TraceDataset,
    *,
    pad_token_id: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate(rows, pad_token_id),
    )
    model.eval()
    output: list[dict[str, Any]] = []
    for batch in loader:
        losses, counts = per_example_nll(model, batch)
        for example_id, loss, count in zip(
            batch["example_ids"], losses, counts, strict=True
        ):
            output.append(
                {
                    "example_id": str(example_id),
                    "nll": float(loss),
                    "target_tokens": int(count),
                }
            )
    return output


def train(model: Any, dataset: TraceDataset, tokenizer: Any, job: dict[str, Any]) -> dict[str, float]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    lora_cfg = job["lora"]
    batch_size = int(lora_cfg["batch_size"])
    accumulation = int(lora_cfg["gradient_accumulation"])
    max_steps = int(job["max_steps"])
    generator = torch.Generator()
    generator.manual_seed(int(job["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
    )
    optimizer_kwargs = {
        "lr": float(lora_cfg["learning_rate"]),
        "weight_decay": float(lora_cfg["weight_decay"]),
    }
    try:
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            fused=True,
            **optimizer_kwargs,
        )
    except TypeError:
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            **optimizer_kwargs,
        )
    warmup_steps = int(round(max_steps * float(lora_cfg["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    total_loss = 0.0
    total_target_tokens = 0
    cycle = itertools.cycle(loader)
    started = time.time()
    while step < max_steps:
        batch = next(cycle)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        active = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~active, 0)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape_as(safe_labels)
        counts = active.sum(dim=1).clamp_min(1)
        example_nll = (token_losses * active).sum(dim=1) / counts
        weights = batch["weights"].cuda(non_blocking=True)
        signed_loss = (weights * example_nll).sum() / weights.abs().sum().clamp_min(1e-6)
        loss = signed_loss / accumulation
        loss.backward()
        total_loss += float(signed_loss.detach().cpu())
        total_target_tokens += int(sum(batch["target_tokens"]))
        micro_step += 1
        if micro_step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(lora_cfg["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    torch.cuda.synchronize()
    return {
        "optimizer_steps": float(step),
        "micro_steps": float(micro_step),
        "mean_micro_loss": total_loss / max(1, micro_step),
        "processed_target_tokens": float(total_target_tokens),
        "elapsed_seconds": time.time() - started,
    }


def load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "job_id",
        "job_signature",
        "base_model",
        "train_file",
        "eval_file",
        "output_dir",
        "lora",
        "max_steps",
        "seed",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"TRACE job is missing fields: {sorted(missing)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TRACE LoRA micro-update job.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("TRACE LoRA jobs require one CUDA_VISIBLE_DEVICES entry")
    if "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Each TRACE LoRA job must receive exactly one visible GPU")

    job_path = Path(args.job).resolve()
    job = load_job(job_path)
    output_dir = Path(job["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]:
            print(f"Reusing completed TRACE job: {job['job_id']}")
            return
        stale = output_dir.with_name(f"{output_dir.name}.stale.{int(time.time())}")
        output_dir.rename(stale)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"TRACE job expected one visible CUDA GPU; found {torch.cuda.device_count()}"
        )
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"TRACE requires the CUDA 12.9 wheel; found {torch.version.cuda}")
    set_seed(int(job["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        job["base_model"], trust_remote_code=bool(job.get("trust_remote_code", False))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        job["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=bool(job.get("trust_remote_code", False)),
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    if bool(job["lora"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(job["lora"]["rank"]),
        lora_alpha=int(job["lora"]["alpha"]),
        lora_dropout=float(job["lora"]["dropout"]),
        target_modules=list(job["lora"]["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    train_rows = read_jsonl(job["train_file"])
    eval_rows = read_jsonl(job["eval_file"])
    max_length = int(job["max_length"])
    train_dataset = TraceDataset(
        [encode_example(tokenizer, row, max_length=max_length) for row in train_rows]
    )
    eval_dataset = TraceDataset(
        [encode_example(tokenizer, row, max_length=max_length) for row in eval_rows]
    )

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
    adapted_by_id = {row["example_id"]: row for row in adapted}
    losses = []
    for base_row in baseline:
        after = adapted_by_id[base_row["example_id"]]
        losses.append(
            {
                "job_id": job["job_id"],
                "experiment_id": job["experiment_id"],
                "variant": job["variant"],
                "seed": int(job["seed"]),
                "example_id": base_row["example_id"],
                "baseline_nll": float(base_row["nll"]),
                "adapted_nll": float(after["nll"]),
                "heldout_gain": float(base_row["nll"] - after["nll"]),
                "target_tokens": int(base_row["target_tokens"]),
            }
        )
    baseline_mean = float(np.mean([row["baseline_nll"] for row in losses]))
    adapted_mean = float(np.mean([row["adapted_nll"] for row in losses]))
    metrics = {
        "schema": 1,
        "job_id": job["job_id"],
        "job_signature": job["job_signature"],
        "experiment_id": job["experiment_id"],
        "variant": job["variant"],
        "seed": int(job["seed"]),
        "profile": job["profile"],
        "baseline_nll": baseline_mean,
        "adapted_nll": adapted_mean,
        "heldout_gain": baseline_mean - adapted_mean,
        "eval_examples": len(losses),
        "train_examples": len(train_rows),
        "train_file_sha256": file_sha256(job["train_file"]),
        "eval_file_sha256": file_sha256(job["eval_file"]),
        "metadata": job["metadata"],
        "training": train_metrics,
    }
    metrics["result_signature"] = canonical_signature(metrics)

    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    atomic_write_jsonl(output_dir / "eval_losses.jsonl", losses)
    atomic_write_json(output_dir / "job.json", job)
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

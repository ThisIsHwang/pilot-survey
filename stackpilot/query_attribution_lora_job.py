from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.query_attribution_common import SCHEMA, atomic_write_json, atomic_write_jsonl, file_sha256, read_jsonl

SYSTEM_MESSAGE = "You generate the next search query after observing prior retrieval results. Return only the query, with no explanation or XML tags."


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


def prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_MESSAGE}\nUser: {prompt}\nAssistant:"


def encode_target(tokenizer: Any, prompt: str, row: dict[str, Any], max_length: int) -> EncodedTarget:
    target_text = str(row["text"]).strip()
    if not target_text:
        raise RuntimeError(f"Empty target {row.get('target_id')}")
    prompt_ids = tokenizer.encode(prompt_text(tokenizer, prompt), add_special_tokens=False)
    target_ids = tokenizer.encode(target_text + (tokenizer.eos_token or ""), add_special_tokens=False)
    if not target_ids:
        raise RuntimeError(f"No tokens for target {row.get('target_id')}")
    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1] + [tokenizer.eos_token_id]
    prompt_ids = prompt_ids[-max(1, max_length - len(target_ids)) :]
    weight = float(row.get("weight", 1.0))
    if not math.isfinite(weight) or weight <= 0.0:
        raise RuntimeError(f"Invalid positive weight {weight}")
    return EncodedTarget(str(row["target_id"]), prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids, len(target_ids), weight, {key: value for key, value in row.items() if key not in {"text", "weight"}})


def encode_group(tokenizer: Any, row: dict[str, Any], max_length: int) -> EncodedGroup:
    targets = [encode_target(tokenizer, str(row["prompt"]), target, max_length) for target in row["targets"]]
    weights = [target.weight for target in targets]
    if abs(sum(weights) - 1.0) > 1e-6 and not all(abs(value - 1.0) <= 1e-8 for value in weights):
        raise RuntimeError(f"Group {row['group_id']} weights must sum to 1 or all be 1")
    return EncodedGroup(str(row["group_id"]), targets, {key: value for key, value in row.items() if key not in {"targets", "prompt"}})


def collate(groups: list[EncodedGroup], pad_token_id: int) -> dict[str, Any]:
    import torch
    flat = [target for group in groups for target in group.targets]
    max_length = max(len(target.input_ids) for target in flat)
    input_ids, labels, masks, group_indices, weights = [], [], [], [], []
    target_ids, target_tokens, target_metadata, group_ids, group_metadata = [], [], [], [], []
    for group_index, group in enumerate(groups):
        group_ids.append(group.group_id)
        group_metadata.append(group.metadata)
        for target in group.targets:
            padding = max_length - len(target.input_ids)
            input_ids.append(target.input_ids + [pad_token_id] * padding)
            labels.append(target.labels + [-100] * padding)
            masks.append([1] * len(target.input_ids) + [0] * padding)
            group_indices.append(group_index)
            weights.append(target.weight)
            target_ids.append(target.target_id)
            target_tokens.append(target.target_tokens)
            target_metadata.append(target.metadata)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long), "attention_mask": torch.tensor(masks, dtype=torch.long), "group_indices": torch.tensor(group_indices, dtype=torch.long), "weights": torch.tensor(weights, dtype=torch.float32), "group_ids": group_ids, "group_metadata": group_metadata, "target_ids": target_ids, "target_tokens": target_tokens, "target_metadata": target_metadata}


def per_target_nll(model: Any, batch: dict[str, Any], *, grad: bool) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=batch["input_ids"].cuda(non_blocking=True), attention_mask=batch["attention_mask"].cuda(non_blocking=True)).logits
        labels = batch["labels"].cuda(non_blocking=True)
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        active = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~active, 0)
        token_losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), safe_labels.reshape(-1), reduction="none").reshape_as(safe_labels)
        counts = active.sum(dim=1).clamp_min(1)
        losses = (token_losses * active).sum(dim=1) / counts
    return losses, counts


def objective_loss(losses: Any, batch: dict[str, Any], objective: dict[str, Any]) -> Any:
    import torch
    indices = batch["group_indices"].cuda(non_blocking=True)
    weights = batch["weights"].cuda(non_blocking=True)
    kind = str(objective["kind"])
    group_losses = []
    for group_index in range(len(batch["group_ids"])):
        mask = indices.eq(group_index)
        values = losses[mask]
        local_weights = weights[mask]
        local_weights = local_weights / local_weights.sum().clamp_min(1e-8)
        mean = (local_weights * values).sum()
        if kind == "weighted_mean":
            value = mean
        elif kind == "softmax":
            temperature = float(objective.get("temperature", 0.2))
            value = temperature * torch.logsumexp(values / temperature + torch.log(local_weights.clamp_min(1e-8)), dim=0)
        elif kind == "mean_plus_variance":
            variance = (local_weights * (values - mean).pow(2)).sum()
            value = mean + float(objective.get("coefficient", 0.5)) * variance
        else:
            raise ValueError(f"Unknown objective kind {kind}")
        group_losses.append(value)
    return torch.stack(group_losses).mean()


def evaluate(model: Any, groups: list[EncodedGroup], tokenizer: Any, batch_size: int) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader
    loader = DataLoader(groups, batch_size=batch_size, shuffle=False, collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id))
    model.eval()
    output = []
    for batch in loader:
        losses, counts = per_target_nll(model, batch, grad=False)
        losses = losses.detach().cpu().numpy()
        counts = counts.detach().cpu().numpy()
        for index, (target_id, loss, count, metadata) in enumerate(zip(batch["target_ids"], losses, counts, batch["target_metadata"], strict=True)):
            group_index = int(batch["group_indices"][index])
            output.append({"group_id": str(batch["group_ids"][group_index]), "target_id": str(target_id), "nll": float(loss), "target_tokens": int(count), **batch["group_metadata"][group_index], **metadata})
    return output


def train(model: Any, groups: list[EncodedGroup], tokenizer: Any, job: dict[str, Any]) -> dict[str, float]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    lora = job["lora"]
    generator = torch.Generator().manual_seed(int(job["seed"]))
    loader = DataLoader(groups, batch_size=int(lora["batch_size"]), shuffle=True, generator=generator, collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        optimizer = AdamW(parameters, lr=float(lora["learning_rate"]), weight_decay=float(lora["weight_decay"]), fused=True)
    except TypeError:
        optimizer = AdamW(parameters, lr=float(lora["learning_rate"]), weight_decay=float(lora["weight_decay"]))
    steps = int(job["max_steps"])
    accumulation = int(lora["gradient_accumulation"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(round(steps * float(lora["warmup_ratio"]))), num_training_steps=steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stream = itertools.cycle(loader)
    optimizer_step = micro_step = target_count = 0
    total_loss = 0.0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    while optimizer_step < steps:
        batch = next(stream)
        losses, _counts = per_target_nll(model, batch, grad=True)
        value = objective_loss(losses, batch, job["objective"])
        if not torch.isfinite(value):
            raise RuntimeError(f"Non-finite training loss at micro-step {micro_step}")
        (value / accumulation).backward()
        total_loss += float(value.detach().cpu())
        target_count += len(batch["target_ids"])
        micro_step += 1
        if micro_step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(parameters, float(lora["max_grad_norm"]))
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_step += 1
    torch.cuda.synchronize()
    return {"optimizer_steps": float(optimizer_step), "micro_steps": float(micro_step), "mean_group_loss": total_loss / max(1, micro_step), "processed_targets": float(target_count), "elapsed_seconds": time.time() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}


def validate_finite(payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite metric {key}: {value}")


def run(job_path: Path, *, force: bool) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]:
            print(f"Reusing {job['job_id']}"); return
        output_dir.rename(output_dir.with_name(f"{output_dir.name}.stale.{int(time.time())}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Each attribution job requires exactly one visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one CUDA GPU, found {torch.cuda.device_count()}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"CUDA 12.9 wheel required; found {torch.version.cuda}")
    set_seed(int(job["seed"])); torch.backends.cuda.matmul.allow_tf32 = True
    local_only = Path(str(job["base_model"])).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(job["base_model"], trust_remote_code=bool(job.get("trust_remote_code", False)), local_files_only=local_only)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(job["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=bool(job.get("trust_remote_code", False)), local_files_only=local_only, low_cpu_mem_usage=True).cuda()
    model.config.use_cache = False
    if bool(job["lora"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(job["lora"]["rank"]), lora_alpha=int(job["lora"]["alpha"]), lora_dropout=float(job["lora"]["dropout"]), target_modules=list(job["lora"]["target_modules"]), bias="none"))
    max_length = int(job["max_length"])
    train_groups = [encode_group(tokenizer, row, max_length) for row in read_jsonl(job["train_file"])]
    eval_groups = [encode_group(tokenizer, row, max_length) for row in read_jsonl(job["eval_file"])]
    try:
        baseline = evaluate(model, eval_groups, tokenizer, int(job["lora"]["eval_batch_size"]))
        train_metrics = train(model, train_groups, tokenizer, job)
        adapted = evaluate(model, eval_groups, tokenizer, int(job["lora"]["eval_batch_size"]))
        adapted_map = {(row["group_id"], row["target_id"]): row for row in adapted}
        losses = []
        for base in baseline:
            after = adapted_map[(base["group_id"], base["target_id"])]
            row = {**{key: value for key, value in base.items() if key != "nll"}, "job_id": job["job_id"], "experiment_id": job["experiment_id"], "family": job["family"], "direction": job["direction"], "variant": job["variant"], "seed": int(job["seed"]), "baseline_nll": float(base["nll"]), "adapted_nll": float(after["nll"]), "heldout_gain": float(base["nll"] - after["nll"])}
            validate_finite(row); losses.append(row)
        baseline_mean = float(np.mean([row["baseline_nll"] for row in losses])); adapted_mean = float(np.mean([row["adapted_nll"] for row in losses]))
        metrics = {"schema": SCHEMA, "job_id": job["job_id"], "job_signature": job["job_signature"], "experiment_id": job["experiment_id"], "family": job["family"], "direction": job["direction"], "variant": job["variant"], "seed": int(job["seed"]), "profile": job["profile"], "baseline_nll": baseline_mean, "adapted_nll": adapted_mean, "heldout_gain": baseline_mean - adapted_mean, "train_groups": len(train_groups), "train_targets": sum(len(group.targets) for group in train_groups), "eval_targets": len(losses), "train_file_sha256": file_sha256(job["train_file"]), "eval_file_sha256": file_sha256(job["eval_file"]), "metadata": job["metadata"], **train_metrics}
        validate_finite(metrics)
        atomic_write_jsonl(output_dir / "eval_losses.jsonl", losses)
        adapter_dir = output_dir / "adapter"; model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
        atomic_write_json(metrics_path, metrics)
    except Exception as exc:
        if metrics_path.exists(): metrics_path.unlink()
        atomic_write_json(output_dir / "invalid.json", {"job_id": job.get("job_id"), "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        del model; torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one attribution-hypothesis LoRA job.")
    parser.add_argument("--job", required=True); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); run(Path(args.job).resolve(), force=args.force)


if __name__ == "__main__":
    main()

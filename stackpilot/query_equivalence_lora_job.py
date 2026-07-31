from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.trace_common import atomic_write_json, atomic_write_jsonl, file_sha256, read_jsonl
from stackpilot.trace_lora_job import TraceDataset, collate, encode_example, evaluate, set_seed


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite {label}: {value!r}")
    return number


def load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "job_id", "job_signature", "experiment_id", "variant", "direction",
        "seed", "profile", "base_model", "train_file", "eval_file",
        "output_dir", "lora", "max_steps", "max_length", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Equivalence job misses fields: {sorted(missing)}")
    return payload


def grouped_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        state_id = str(row.get("state_id", ""))
        if not state_id:
            raise RuntimeError(f"Training example {row.get('example_id')} has no state_id")
        weight = float(row.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(f"EXP-014 uses positive group credit only; invalid weight {weight}")
        by_state[state_id].append(row)
    if not by_state:
        raise RuntimeError("Equivalence training data is empty")
    return [by_state[key] for key in sorted(by_state)]


def train_grouped(model: Any, tokenizer: Any, train_rows: list[dict[str, Any]], job: dict[str, Any]) -> dict[str, float]:
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    max_length = int(job["max_length"])
    groups = grouped_rows(train_rows)
    encoded_groups = [
        [encode_example(tokenizer, row, max_length=max_length) for row in group]
        for group in groups
    ]
    lora_cfg = job["lora"]
    state_batch_size = int(lora_cfg["batch_size"])
    accumulation = int(lora_cfg["gradient_accumulation"])
    max_steps = int(job["max_steps"])
    rng = random.Random(int(job["seed"]))
    order = list(range(len(encoded_groups)))

    def batch_stream():
        while True:
            rng.shuffle(order)
            for start in range(0, len(order), state_batch_size):
                yield order[start : start + state_batch_size]

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs = {
        "lr": float(lora_cfg["learning_rate"]),
        "weight_decay": float(lora_cfg["weight_decay"]),
    }
    try:
        optimizer = AdamW(parameters, fused=True, **optimizer_kwargs)
    except TypeError:
        optimizer = AdamW(parameters, **optimizer_kwargs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(max_steps * float(lora_cfg["warmup_ratio"]))),
        num_training_steps=max_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = total_state_groups = total_targets = 0
    total_loss = 0.0
    started = time.time()
    batches = batch_stream()
    while step < max_steps:
        batch_indices = next(batches)
        flat_examples, group_assignments = [], []
        for local_group, group_index in enumerate(batch_indices):
            for example in encoded_groups[group_index]:
                flat_examples.append(example)
                group_assignments.append(local_group)
        batch = collate(flat_examples, tokenizer.pad_token_id)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        active = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~active, 0)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            safe_labels.reshape(-1), reduction="none",
        ).reshape_as(safe_labels)
        counts = active.sum(dim=1).clamp_min(1)
        example_nll = (token_losses * active).sum(dim=1) / counts
        group_tensor = torch.tensor(group_assignments, device=example_nll.device)
        state_losses = [
            example_nll[group_tensor == group_id].mean()
            for group_id in range(len(batch_indices))
        ]
        grouped_loss = torch.stack(state_losses).mean()
        _finite(grouped_loss.detach().float().cpu(), "training loss")
        (grouped_loss / accumulation).backward()
        total_loss += float(grouped_loss.detach().float().cpu())
        total_state_groups += len(state_losses)
        total_targets += int(sum(batch["target_tokens"]))
        micro_step += 1
        if micro_step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(parameters, float(lora_cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    torch.cuda.synchronize()
    return {
        "optimizer_steps": float(step), "micro_steps": float(micro_step),
        "mean_micro_loss": total_loss / max(1, micro_step),
        "processed_state_groups": float(total_state_groups),
        "processed_target_tokens": float(total_targets),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EXP-014 group-credit LoRA job.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Each EXP-014 job requires exactly one CUDA_VISIBLE_DEVICES entry")

    job = load_job(Path(args.job).resolve())
    output_dir = Path(job["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]:
            print(f"Reusing completed EXP-014 job: {job['job_id']}")
            return
        output_dir.rename(output_dir.with_name(f"{output_dir.name}.stale.{int(time.time())}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"EXP-014 expected one visible CUDA GPU; found {torch.cuda.device_count()}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"EXP-014 requires PyTorch CUDA 12.9; found {torch.version.cuda}")
    set_seed(int(job["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(job["base_model"], trust_remote_code=bool(job.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        job["base_model"], torch_dtype=torch.bfloat16,
        trust_remote_code=bool(job.get("trust_remote_code", False)), low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    if bool(job["lora"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(job["lora"]["rank"]),
        lora_alpha=int(job["lora"]["alpha"]), lora_dropout=float(job["lora"]["dropout"]),
        target_modules=list(job["lora"]["target_modules"]), bias="none",
    ))

    train_rows = read_jsonl(job["train_file"])
    eval_rows = read_jsonl(job["eval_file"])
    eval_dataset = TraceDataset([
        encode_example(tokenizer, row, max_length=int(job["max_length"])) for row in eval_rows
    ])
    baseline = evaluate(model, eval_dataset, pad_token_id=tokenizer.pad_token_id, batch_size=int(job["lora"]["eval_batch_size"]))
    train_metrics = train_grouped(model, tokenizer, train_rows, job)
    adapted = evaluate(model, eval_dataset, pad_token_id=tokenizer.pad_token_id, batch_size=int(job["lora"]["eval_batch_size"]))
    adapted_by_id = {row["example_id"]: row for row in adapted}
    eval_by_id = {str(row["example_id"]): row for row in eval_rows}
    losses = []
    for base_row in baseline:
        example_id = str(base_row["example_id"])
        after, metadata = adapted_by_id[example_id], eval_by_id[example_id]
        baseline_nll = _finite(base_row["nll"], f"{example_id} baseline_nll")
        adapted_nll = _finite(after["nll"], f"{example_id} adapted_nll")
        losses.append({
            "job_id": job["job_id"], "experiment_id": job["experiment_id"],
            "variant": job["variant"], "seed": int(job["seed"]),
            "direction": job["direction"], "example_id": example_id,
            "state_id": str(metadata["state_id"]), "class_id": str(metadata["class_id"]),
            "style": str(metadata["style"]), "origin": str(metadata["origin"]),
            "baseline_nll": baseline_nll, "adapted_nll": adapted_nll,
            "heldout_gain": baseline_nll - adapted_nll,
            "target_tokens": int(base_row["target_tokens"]),
        })
    for row in losses:
        _finite(row["heldout_gain"], f"{row['example_id']} heldout_gain")
    baseline_mean = float(np.mean([row["baseline_nll"] for row in losses]))
    adapted_mean = float(np.mean([row["adapted_nll"] for row in losses]))
    metrics = {
        "schema": 1, "job_id": job["job_id"], "job_signature": job["job_signature"],
        "experiment_id": job["experiment_id"], "variant": job["variant"],
        "direction": job["direction"], "seed": int(job["seed"]),
        "profile": job["profile"], "baseline_nll": baseline_mean,
        "adapted_nll": adapted_mean, "heldout_gain": baseline_mean - adapted_mean,
        "eval_examples": len(losses), "train_states": len(grouped_rows(train_rows)),
        "train_examples": len(train_rows), "train_file_sha256": file_sha256(job["train_file"]),
        "eval_file_sha256": file_sha256(job["eval_file"]),
        "metadata": job["metadata"], "train_metrics": train_metrics,
    }
    for name in ("baseline_nll", "adapted_nll", "heldout_gain"):
        _finite(metrics[name], name)
    atomic_write_jsonl(output_dir / "eval_losses.jsonl", losses)
    atomic_write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

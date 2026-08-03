from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.interface_causality_common import (
    atomic_write_json,
    balanced_state_subset,
    candidate_reward,
    group_candidates,
    load_config,
    load_state_results,
    normalize_advantages,
    source_patterns,
)

SYSTEM_MESSAGE = (
    "You generate the next search query after observing prior retrieval results. "
    "Return only the query, without explanation or XML tags."
)


def prompt_text(result: dict[str, Any]) -> str:
    state = result["state"]
    lines = [f"Question: {state['question']}", "Search history:"]
    records = result.get("prefix", {}).get("records", []) or []
    if not records:
        records = state.get("prior_turns", []) or []
    for record in records:
        lines.append(f"- Query {record.get('turn')}: {record.get('query', '')}")
        titles = record.get("observed_titles", [])
        if isinstance(titles, list) and titles:
            lines.append("  Observed titles: " + " | ".join(map(str, titles[:10])))
    lines.extend(
        [
            "Generate the next search query that best advances the unresolved information need.",
            "Return only the query.",
        ]
    )
    return "\n".join(lines)


def chat_prompt(tokenizer: Any, prompt: str) -> str:
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


def encode(tokenizer: Any, prompt: str, target: str, max_length: int) -> tuple[Any, Any, Any]:
    import torch

    prompt_ids = tokenizer.encode(chat_prompt(tokenizer, prompt), add_special_tokens=False)
    target_ids = tokenizer.encode(target.strip() + (tokenizer.eos_token or ""), add_special_tokens=False)
    if not target_ids:
        raise RuntimeError("Empty query target")
    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1] + [tokenizer.eos_token_id]
    prompt_ids = prompt_ids[-max(1, max_length - len(target_ids)) :]
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device="cuda")
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    return input_ids, labels, attention


def gradient_vector(model: Any, tokenizer: Any, prompt: str, target: str, max_length: int) -> np.ndarray:
    import torch

    model.zero_grad(set_to_none=True)
    input_ids, labels, attention = encode(tokenizer, prompt, target, max_length)
    logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
    shifted = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()
    parts = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            parts.append(parameter.grad.detach().flatten().to(dtype=torch.float16, device="cpu"))
    if not parts:
        raise RuntimeError("No LoRA gradients were produced")
    vector = torch.cat(parts).float().numpy()
    if not np.isfinite(vector).all():
        raise RuntimeError("Non-finite gradient vector")
    return vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 1.0 if np.linalg.norm(left - right) <= 1e-12 else 0.0
    return float(np.dot(left, right) / denominator)


def aggregate_surface(
    gradients: list[np.ndarray],
    rewards: list[float],
    class_ids: list[int],
    injected_class: int,
    multiplicity: int,
) -> np.ndarray:
    expanded_gradients = []
    expanded_rewards = []
    for gradient, reward, class_id in zip(gradients, rewards, class_ids, strict=True):
        repeat = multiplicity if class_id == injected_class else 1
        for _copy in range(repeat):
            expanded_gradients.append(gradient)
            expanded_rewards.append(reward)
    advantages = normalize_advantages(expanded_rewards)
    output = np.zeros_like(gradients[0], dtype=np.float64)
    for advantage, gradient in zip(advantages, expanded_gradients, strict=True):
        output += float(advantage) * gradient
    return output / max(1, len(expanded_gradients))


def aggregate_quotient(
    gradients: list[np.ndarray],
    rewards: list[float],
    class_ids: list[int],
) -> np.ndarray:
    class_count = max(class_ids) + 1
    class_rewards = []
    class_gradients = []
    for class_id in range(class_count):
        indices = [index for index, value in enumerate(class_ids) if value == class_id]
        class_rewards.append(float(np.mean([rewards[index] for index in indices])))
        class_gradients.append(np.mean([gradients[index] for index in indices], axis=0))
    advantages = normalize_advantages(class_rewards)
    output = np.zeros_like(gradients[0], dtype=np.float64)
    for advantage, gradient in zip(advantages, class_gradients, strict=True):
        output += float(advantage) * gradient
    return output / max(1, class_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe actual 7B LoRA update rotation under surface alias injection.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Alias-gradient probe requires exactly one CUDA_VISIBLE_DEVICES entry")
    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    base_model = args.base_model or os.environ.get("INTERFACE_BASE_MODEL") or cfg["model"]["base_model"]

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA GPU; found {torch.cuda.device_count()}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    probe_cfg = cfg["gradient_probe"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(probe_cfg["rank"]),
            lora_alpha=int(probe_cfg["alpha"]),
            lora_dropout=0.0,
            target_modules=list(probe_cfg["target_modules"]),
            bias="none",
        ),
    )
    model.eval()
    results = load_state_results(source_patterns(cfg, args.inputs))
    results = balanced_state_subset(results, int(profile["alias_gradient_states"]))
    multiplicities = [int(value) for value in cfg["alias_audit"]["multiplicities"]]
    rows = []
    for result in results:
        state = result["state"]
        candidates = [
            row for row in result["candidates"] if int(row.get("protocol_failure", 0)) == 0
        ]
        classes = group_candidates(
            state,
            candidates,
            mode=str(cfg["alias_audit"]["behavior_signature"]),
        )
        if len(classes) < 2:
            continue
        candidate_to_class = {
            str(candidate["candidate_id"]): class_id
            for class_id, members in enumerate(classes)
            for candidate in members
        }
        gradients = [
            gradient_vector(
                model,
                tokenizer,
                prompt_text(result),
                str(candidate["query"]),
                int(probe_cfg["max_length"]),
            )
            for candidate in candidates
        ]
        rewards = [candidate_reward(candidate, cfg) for candidate in candidates]
        class_ids = [candidate_to_class[str(candidate["candidate_id"])] for candidate in candidates]
        class_rewards = [
            np.mean([reward for reward, cid in zip(rewards, class_ids, strict=True) if cid == class_id])
            for class_id in range(len(classes))
        ]
        best_class = int(np.argmax(class_rewards))
        injected_class = max(
            (index for index in range(len(classes)) if index != best_class),
            key=lambda index: (len(classes[index]), -index),
        )
        quotient = aggregate_quotient(gradients, rewards, class_ids)
        baseline = aggregate_surface(gradients, rewards, class_ids, injected_class, 1)
        for multiplicity in multiplicities:
            surface = aggregate_surface(
                gradients,
                rewards,
                class_ids,
                injected_class,
                multiplicity,
            )
            rows.append(
                {
                    "state_id": str(state["state_id"]),
                    "backend": str(state["backend"]),
                    "dataset": str(state["dataset"]),
                    "multiplicity": multiplicity,
                    "surface_cosine_to_m1": cosine(surface, baseline),
                    "quotient_cosine_to_m1": cosine(quotient, quotient),
                    "surface_relative_norm_change": abs(
                        np.linalg.norm(surface) - np.linalg.norm(baseline)
                    ) / max(1e-12, np.linalg.norm(baseline)),
                    "surface_vs_quotient_cosine": cosine(surface, quotient),
                    "gradient_dimensions": len(surface),
                    "behavior_classes": len(classes),
                }
            )
    if not rows:
        raise RuntimeError("No eligible states for alias-gradient probing")
    frame = pd.DataFrame(rows)
    output_dir = Path(
        args.output_dir
        or Path(cfg["work_dir"]) / "reports" / args.profile / "EXP-020-gradient"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "gradient_rotation.csv", index=False)
    maximum = max(multiplicities)
    max_rows = frame[frame["multiplicity"] == maximum]
    decision = {
        "profile": args.profile,
        "states": int(frame["state_id"].nunique()),
        "max_multiplicity": maximum,
        "mean_surface_cosine_to_m1": float(max_rows["surface_cosine_to_m1"].mean()),
        "mean_surface_relative_norm_change": float(max_rows["surface_relative_norm_change"].mean()),
        "mean_surface_vs_quotient_cosine": float(max_rows["surface_vs_quotient_cosine"].mean()),
        "go": bool(
            float(max_rows["surface_cosine_to_m1"].mean())
            <= float(probe_cfg["maximum_surface_cosine_at_max_alias"])
        ),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    (output_dir / "EXP020_GRADIENT_REPORT.md").write_text(
        "# EXP-020 Alias-gradient probe\n\n```text\n"
        + json.dumps(decision, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

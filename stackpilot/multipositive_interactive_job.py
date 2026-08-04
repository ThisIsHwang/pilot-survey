from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from stackpilot.multipositive_common import atomic_write_json, atomic_write_jsonl, read_jsonl, stable_hash
from stackpilot.query_attribution_common import query_jaccard

SEARCH_TAG = re.compile(r"<search>(.*?)</search>", re.I | re.S)


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def recall(gold: list[str], observed: list[str]) -> float:
    gold_set = {normalize_title(value) for value in gold if str(value).strip()}
    observed_set = {normalize_title(value) for value in observed if str(value).strip()}
    return len(gold_set & observed_set) / max(1, len(gold_set))


def clean_query(text: str) -> str:
    match = SEARCH_TAG.search(text)
    if match:
        text = match.group(1)
    return " ".join(text.strip().strip('"').splitlines()[0].split())


def title_of(item: dict[str, Any]) -> str:
    document = item.get("document", item)
    if not isinstance(document, dict):
        return ""
    title = str(document.get("title") or "").strip()
    if title:
        return title
    contents = str(document.get("contents") or "")
    return contents.splitlines()[0].strip('" ') if contents else ""


def retrieve(url: str, query: str, topk: int) -> list[str]:
    response = requests.post(url, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180)
    response.raise_for_status()
    rows = response.json().get("result", [[]])[0]
    return [title for title in (title_of(item) for item in rows) if title]


def transition_signature(titles: list[str]) -> tuple[str, ...]:
    return tuple(normalize_title(value) for value in titles)


def mean_pairwise_query_distance(queries: list[str]) -> float:
    if len(queries) < 2:
        return 0.0
    values = [1.0 - query_jaccard(left, right) for left, right in itertools.combinations(queries, 2)]
    return float(sum(values) / len(values))


def run(job_path: Path, *, force: bool) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"])
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]:
            print(f"Reusing {job['job_id']}")
            return
    adapter_dir = Path(job["adapter_dir"])
    if not adapter_dir.is_dir():
        raise RuntimeError(f"Missing adapter: {adapter_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Interactive job requires one visible GPU")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    local_base = Path(str(job["base_model"])).is_dir()
    base = AutoModelForCausalLM.from_pretrained(job["base_model"], torch_dtype=torch.bfloat16, local_files_only=local_base, low_cpu_mem_usage=True).cuda()
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    max_budget = max(int(value) for value in job["sample_budgets"])
    results = []
    started = time.time()

    for group in read_jsonl(job["eval_file"]):
        messages = [{"role": "system", "content": "Generate one useful next search query. Return only the query."}, {"role": "user", "content": str(group["prompt"])}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.manual_seed(int(stable_hash(job["seed"], group["state_id"], length=8), 16))
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=int(job["max_new_tokens"]), do_sample=max_budget > 1, temperature=max(float(job["temperature"]), 1e-6), num_return_sequences=max_budget, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        queries = [clean_query(tokenizer.decode(row[encoded["input_ids"].shape[1]:], skip_special_tokens=True)) for row in generated]
        query_records = []
        before = recall(list(group["support_titles"]), list(group["prefix_observed_titles"]))
        for query in queries:
            invalid = int(len(query.split()) < 2 or len(query.split()) > 64)
            titles = [] if invalid else retrieve(str(job["retrieval_url"]), query, int(group["topk"]))
            after = recall(list(group["support_titles"]), list(group["prefix_observed_titles"]) + titles)
            query_records.append({"query": query, "invalid": invalid, "titles": titles, "evidence_gain": after - before})
        for budget in sorted(set(int(value) for value in job["sample_budgets"])):
            selected = query_records[:budget]
            valid = [row for row in selected if not row["invalid"]]
            union_titles = list(group["prefix_observed_titles"]) + [title for row in valid for title in row["titles"]]
            union_gain = recall(list(group["support_titles"]), union_titles) - before
            signatures = {transition_signature(row["titles"]) for row in valid}
            valid_queries = [str(row["query"]) for row in valid]
            result = {
                "job_id": job["job_id"],
                "job_signature": job["job_signature"],
                "direction": job["direction"],
                "variant": job["variant"],
                "seed": int(job["seed"]),
                "state_id": str(group["state_id"]),
                "question_id": str(group["question_id"]),
                "dataset": str(group["dataset"]),
                "backend": str(group["backend"]),
                "sample_budget": budget,
                "queries": [row["query"] for row in selected],
                "retrieved_titles": [row["titles"] for row in selected],
                "mean_evidence_gain": float(sum(row["evidence_gain"] for row in selected) / budget),
                "best_evidence_gain": float(max((row["evidence_gain"] for row in selected), default=0.0)),
                "union_evidence_gain": float(union_gain),
                "unique_behavior_count": float(len(signatures)),
                "duplicate_behavior_rate": float(1.0 - len(signatures) / max(1, len(valid))),
                "query_diversity": mean_pairwise_query_distance(valid_queries),
                "invalid_rate": float(sum(row["invalid"] for row in selected) / budget),
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
                raise RuntimeError(f"Non-finite result for {group['state_id']}")
            results.append(result)
    atomic_write_jsonl(output_dir / "results.jsonl", results)
    atomic_write_json(metrics_path, {"schema": 1, "job_id": job["job_id"], "job_signature": job["job_signature"], "experiment_id": "EXP-027", "direction": job["direction"], "variant": job["variant"], "seed": int(job["seed"]), "rows": len(results), "elapsed_seconds": time.time() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one interactive multi-positive evaluation.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.job).resolve(), force=args.force)


if __name__ == "__main__":
    main()

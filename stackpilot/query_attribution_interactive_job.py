from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from stackpilot.query_attribution_common import SCHEMA, atomic_write_json, atomic_write_jsonl, read_jsonl

SEARCH_TAG = re.compile(r"<search>(.*?)</search>", re.I | re.S)


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def recall(gold: list[str], observed: list[str]) -> float:
    gold_set = {normalize_title(value) for value in gold if str(value).strip()}; observed_set = {normalize_title(value) for value in observed if str(value).strip()}; return len(gold_set & observed_set) / max(1, len(gold_set))


def clean_query(text: str) -> str:
    match = SEARCH_TAG.search(text)
    if match: text = match.group(1)
    return " ".join(text.strip().splitlines()[0].strip().strip('"').split())


def title_of(item: dict[str, Any]) -> str:
    document = item.get("document", item)
    if not isinstance(document, dict): return ""
    title = str(document.get("title") or "").strip()
    if title: return title
    contents = str(document.get("contents") or ""); return contents.splitlines()[0].strip('" ') if contents else ""


def retrieve(url: str, query: str, topk: int) -> list[str]:
    response = requests.post(url, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180); response.raise_for_status(); results = response.json().get("result", [[]])[0]; return [title for title in (title_of(item) for item in results) if title]


def run(job_path: Path, *, force: bool) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8")); output_dir = Path(job["output_dir"]); metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("job_signature") == job["job_signature"]: print(f"Reusing {job['job_id']}"); return
    adapter_dir = Path(job["adapter_dir"])
    if not adapter_dir.is_dir(): raise RuntimeError(f"Missing adapter directory: {adapter_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]: raise RuntimeError("Each interactive job requires one visible GPU")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    local_base = Path(str(job["base_model"])).is_dir(); base = AutoModelForCausalLM.from_pretrained(job["base_model"], torch_dtype=torch.bfloat16, local_files_only=local_base, low_cpu_mem_usage=True).cuda(); model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    results = []; started = time.time()
    for group in read_jsonl(job["eval_file"]):
        messages = [{"role": "system", "content": "Generate the next search query. Return only the query without explanation."}, {"role": "user", "content": str(group["prompt"])}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=int(job["max_new_tokens"]), do_sample=float(job["temperature"]) > 0, temperature=max(float(job["temperature"]), 1e-6), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        query = clean_query(tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)); invalid = int(len(query.split()) < 2 or len(query.split()) > 64); titles = [] if invalid else retrieve(str(job["retrieval_url"]), query, int(group["topk"])); before = recall(list(group["support_titles"]), list(group["prefix_observed_titles"])); after = recall(list(group["support_titles"]), list(group["prefix_observed_titles"]) + titles)
        result = {"job_id": job["job_id"], "job_signature": job["job_signature"], "direction": job["direction"], "variant": job["variant"], "seed": int(job["seed"]), "state_id": str(group["state_id"]), "question_id": str(group["question_id"]), "dataset": str(group["dataset"]), "backend": str(group["backend"]), "query": query, "invalid_query": invalid, "retrieved_titles": titles, "prefix_support_recall": before, "final_support_recall": after, "evidence_gain": after - before}
        if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()): raise RuntimeError(f"Non-finite interactive result for {group['state_id']}")
        results.append(result)
    atomic_write_jsonl(output_dir / "results.jsonl", results); atomic_write_json(metrics_path, {"schema": SCHEMA, "job_id": job["job_id"], "job_signature": job["job_signature"], "experiment_id": "EXP-019", "direction": job["direction"], "variant": job["variant"], "seed": int(job["seed"]), "states": len(results), "mean_evidence_gain": sum(row["evidence_gain"] for row in results) / len(results), "mean_final_support_recall": sum(row["final_support_recall"] for row in results) / len(results), "invalid_rate": sum(row["invalid_query"] for row in results) / len(results), "elapsed_seconds": time.time() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EXP-019 interactive adapter evaluation."); parser.add_argument("--job", required=True); parser.add_argument("--force", action="store_true"); args = parser.parse_args(); run(Path(args.job).resolve(), force=args.force)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import (
    answer_em,
    answer_f1,
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.react_agent_eval import SYSTEM_PROMPT, format_results, parse_action
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import check_vllm, effective_model_name

RESULT_SCHEMA = 2
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("_", " ").strip()
    text = text.strip('"\'')
    return " ".join(text.lower().split())


def token_set(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value)}


def query_features(question: str, query: str, previous_query: str | None) -> dict[str, float]:
    tokens = TOKEN_RE.findall(query)
    question_tokens = token_set(question)
    query_tokens = {token.lower() for token in tokens}
    previous_tokens = token_set(previous_query or "")
    overlap = len(question_tokens & query_tokens) / max(1, len(question_tokens))
    change = 1.0
    if previous_query is not None:
        change = 1.0 - len(query_tokens & previous_tokens) / max(1, len(query_tokens | previous_tokens))
    return {
        "query_token_count": float(len(tokens)),
        "query_question_overlap": float(overlap),
        "query_has_quotes": float('"' in query or "'" in query),
        "query_capitalized_ratio": float(
            sum(token[:1].isupper() for token in tokens) / max(1, len(tokens))
        ),
        "query_numeric_ratio": float(
            sum(any(character.isdigit() for character in token) for token in tokens)
            / max(1, len(tokens))
        ),
        "query_lexical_change": float(change),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    model_path = os.environ.get("MODEL_PATH") or os.environ.get("MODEL") or cfg["llm"]["model"]
    identity: dict[str, Any] = {
        "model_path": str(model_path),
        "served_model_name": effective_model_name(cfg),
    }
    path = Path(str(model_path)).expanduser()
    if path.is_dir():
        identity["resolved_model_path"] = str(path.resolve())
        config_path = path / "config.json"
        if config_path.is_file():
            identity["config_sha256"] = file_sha256(config_path)
    return identity


def run_signature(cfg: dict[str, Any], data_file: Path, tag: str, seed: int) -> str:
    payload = {
        "schema": RESULT_SCHEMA,
        "config": cfg,
        "data_sha256": file_sha256(data_file),
        "tag": tag,
        "seed": seed,
        "model": model_identity(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_retriever(name: str, port: int) -> None:
    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok" or payload.get("backend") != name:
        raise RuntimeError(f"Unexpected {name} health response: {payload}")


def complete(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    cfg: dict[str, Any],
    request_seed: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        seed=request_seed,
    )
    return response.choices[0].message.content or ""


def best_answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    return (
        max(answer_em(prediction, answer) for answer in answers),
        max(answer_f1(prediction, answer) for answer in answers),
    )


def value_at(values: list[float], index: int) -> float:
    return values[index] if len(values) > index else 0.0


def run_episode(
    client: OpenAI,
    model: str,
    retriever: RetrievalClient,
    item: dict[str, Any],
    cfg: dict[str, Any],
    topk: int,
    eval_seed: int,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"Question: {item['question']}"})

    gold_titles = {normalize_title(title) for title in item["support_titles"]}
    cumulative_titles: set[str] = set()
    previously_seen_titles: set[str] = set()
    turn_records: list[dict[str, Any]] = []
    searches: list[str] = []
    prediction = ""
    search_budget = int(cfg["agent"]["max_search_turns"])
    max_attempts = search_budget * 2 + 2
    attempts = 0
    previous_recall = 0.0

    while len(searches) < search_budget and attempts < max_attempts:
        attempts += 1
        seed_text = f"{eval_seed}:{item['id']}:{retriever.name}:{topk}:{attempts}"
        request_seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = complete(client, model, messages, cfg, request_seed)
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search" or not value:
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid format. Output <search>...</search> or <answer>...</answer>.",
                }
            )
            continue

        results = retriever.search(value, topk)
        previous_query = searches[-1] if searches else None
        searches.append(value)
        retrieved_titles = [str(result["title"]) for result in results]
        normalized_retrieved = {normalize_title(title) for title in retrieved_titles}
        cumulative_titles.update(normalized_retrieved)
        matched = gold_titles & cumulative_titles
        recall = len(matched) / max(1, len(gold_titles))
        gain = recall - previous_recall
        new_support = sorted((gold_titles & normalized_retrieved) - previously_seen_titles)
        turn_records.append(
            {
                "turn": len(searches),
                "query": value,
                "retrieved_titles": retrieved_titles,
                "support_recall": recall,
                "evidence_gain": gain,
                "new_support_titles": new_support,
                **query_features(str(item["question"]), value, previous_query),
            }
        )
        previously_seen_titles.update(normalized_retrieved)
        previous_recall = recall
        observation = format_results(results, int(cfg["agent"]["result_snippet_chars"]))
        messages.append(
            {"role": "user", "content": f"<information>\n{observation}\n</information>"}
        )

    if not prediction:
        messages.append(
            {
                "role": "user",
                "content": "The search budget is exhausted. Give your best final answer now as <answer>short answer</answer>.",
            }
        )
        seed_text = f"{eval_seed}:{item['id']}:{retriever.name}:{topk}:final"
        request_seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = complete(client, model, messages, cfg, request_seed)
        action, value = parse_action(content)
        prediction = value if action == "answer" else content.strip()

    answers = [str(answer) for answer in item.get("answers") or [item["answer"]]]
    em, f1 = best_answer_scores(prediction, answers)
    recalls = [float(record["support_recall"]) for record in turn_records]
    gains = [float(record["evidence_gain"]) for record in turn_records]
    turn1_recall = value_at(recalls, 0)
    turn2_recall = value_at(recalls, 1)
    turn3_recall = value_at(recalls, 2)
    turn2_gain = value_at(gains, 1)
    turn3_gain = value_at(gains, 2)
    final_recall = recalls[-1] if recalls else 0.0
    first_miss = turn1_recall < 1.0
    return {
        "question_id": str(item["id"]),
        "question": str(item["question"]),
        "dataset": str(item["dataset"]),
        "backend": retriever.name,
        "topk": topk,
        "prediction": prediction,
        "answers": answers,
        "em": em,
        "f1": f1,
        "support_recall": final_recall,
        "turn1_support_recall": turn1_recall,
        "turn2_support_recall": turn2_recall,
        "turn3_support_recall": turn3_recall,
        "turn2_evidence_gain": turn2_gain,
        "turn3_evidence_gain": turn3_gain,
        "search_count": len(searches),
        "recovery_at_2": float(first_miss and turn2_recall > turn1_recall),
        "recovery_at_3": float(first_miss and turn3_recall > turn1_recall),
        "full_recovery_at_2": float(first_miss and turn2_recall >= 1.0),
        "full_recovery_at_3": float(first_miss and turn3_recall >= 1.0),
        "queries": searches,
        "turns": turn_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backends", nargs="+", choices=("bm25", "e5"), default=("bm25", "e5"))
    parser.add_argument("--topks", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    check_vllm(cfg)
    retrieval_cfg = cfg["retrieval"]
    ports = {"bm25": int(retrieval_cfg["bm25_port"]), "e5": int(retrieval_cfg["e5_port"])}
    for backend in args.backends:
        check_retriever(backend, ports[backend])

    data_file = Path(args.data_file).resolve()
    rows = read_jsonl(data_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    topks = args.topks or [int(value) for value in retrieval_cfg["topks"]]
    work_dir = ensure_dir(Path(cfg["work_dir"]).resolve() / "results" / "policies")
    output_path = work_dir / f"{args.tag}-seed{args.seed}.jsonl"
    summary_path = work_dir / f"{args.tag}-seed{args.seed}_summary.csv"
    signature = run_signature(cfg, data_file, args.tag, args.seed)

    clients = {
        backend: RetrievalClient(backend, f"http://127.0.0.1:{ports[backend]}/retrieve")
        for backend in args.backends
    }
    client = OpenAI(
        base_url=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"],
        timeout=180.0,
        max_retries=5,
    )
    model = effective_model_name(cfg)

    selected_ids = {str(row["id"]) for row in rows}
    existing = read_jsonl_tolerant(output_path)
    row_by_key = {
        (str(row.get("question_id")), str(row.get("backend")), int(row.get("topk", -1))): row
        for row in existing
        if row.get("schema") == RESULT_SCHEMA
        and row.get("run_signature") == signature
        and str(row.get("question_id")) in selected_ids
    }

    total = len(rows) * len(args.backends) * len(topks)
    progress = tqdm(total=total, desc=f"hard RQ0 eval: {args.tag}/seed{args.seed}")
    for item in rows:
        for backend in args.backends:
            for topk in topks:
                key = (str(item["id"]), backend, int(topk))
                if key not in row_by_key:
                    result = run_episode(
                        client,
                        model,
                        clients[backend],
                        item,
                        cfg,
                        int(topk),
                        args.seed,
                    )
                    result.update(
                        {
                            "schema": RESULT_SCHEMA,
                            "run_signature": signature,
                            "policy_tag": args.tag,
                            "seed": args.seed,
                            "served_model": model,
                        }
                    )
                    append_jsonl(output_path, [result])
                    row_by_key[key] = result
                progress.update(1)
    progress.close()

    frame = pd.DataFrame(list(row_by_key.values()))
    metrics = [
        "em",
        "f1",
        "support_recall",
        "turn1_support_recall",
        "turn2_support_recall",
        "turn3_support_recall",
        "turn2_evidence_gain",
        "turn3_evidence_gain",
        "recovery_at_2",
        "recovery_at_3",
        "full_recovery_at_2",
        "full_recovery_at_3",
        "search_count",
    ]
    summary = frame.groupby(["dataset", "backend", "topk"])[metrics].mean().reset_index()
    summary.insert(0, "policy_tag", args.tag)
    summary.insert(1, "seed", args.seed)
    summary.insert(2, "run_signature", signature)
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Raw results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

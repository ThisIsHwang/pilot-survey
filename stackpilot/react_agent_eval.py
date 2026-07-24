from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.action_protocol import parse_action
from stackpilot.common import (
    answer_em,
    answer_f1,
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import (
    check_retrievers,
    check_vllm,
    effective_model_name,
)

RESULT_SCHEMA = 3

SYSTEM_PROMPT = """You are a search agent. Solve the question using the search tool.
At each turn output exactly one action:
<search>your query</search>
or, when ready,
<answer>short answer</answer>
Use returned evidence and do not include explanations inside <answer>.
"""

GUIDANCE = {
    "bm25": "The search system is lexical. Prefer concise entity and relation keywords, rare terms, and exact phrases.",
    "e5": "The search system is semantic. Prefer clear natural-language descriptions and paraphrased relations.",
    "colbert": "The search system uses token-level late interaction. Preserve exact entities and important relation terms in a fluent query.",
}


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalized_title_set(values: object) -> set[str] | None:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        return None
    return {" ".join(value.lower().split()) for value in values}


def episode_validation_error(row: dict, max_search_turns: int) -> str | None:
    if max_search_turns < 1:
        return "max_search_turns must be positive"
    prediction = row.get("prediction")
    raw_prediction = row.get("raw_text_prediction")
    gold = row.get("gold")
    if (
        not isinstance(prediction, str)
        or not isinstance(raw_prediction, str)
        or not isinstance(gold, str)
        or not gold.strip()
    ):
        return "prediction/gold fields are invalid"
    protocol_failure = row.get("protocol_failure")
    if isinstance(protocol_failure, bool) or protocol_failure not in (0, 1, 0.0, 1.0):
        return "protocol_failure must be 0 or 1"
    failed = bool(int(float(protocol_failure)))
    if failed and prediction:
        return "protocol failures must have an empty primary prediction"
    if not failed and not prediction.strip():
        return "successful episodes must have a primary prediction"
    queries = row.get("queries")
    if not isinstance(queries, list) or any(
        not isinstance(query, str) or not query.strip() for query in queries
    ):
        return "queries must be nonempty strings"
    search_count = row.get("search_count")
    invalid_count = row.get("invalid_action_count")
    if (
        isinstance(search_count, bool)
        or isinstance(invalid_count, bool)
        or not finite_number(search_count)
        or not float(search_count).is_integer()
        or int(float(search_count)) != len(queries)
        or not 0 <= int(float(search_count)) <= max_search_turns
    ):
        return "search_count is inconsistent with queries or the budget"
    if (
        not finite_number(invalid_count)
        or not float(invalid_count).is_integer()
        or int(float(invalid_count)) < 0
        or int(float(invalid_count)) > max_search_turns + 1
        or int(float(invalid_count)) + len(queries)
        > max_search_turns + int(failed)
    ):
        return "invalid/search action counts exceed the rollout budget"
    expected_metrics = {
        "em": 0.0 if failed else answer_em(prediction, gold),
        "f1": 0.0 if failed else answer_f1(prediction, gold),
        "raw_text_em": answer_em(raw_prediction, gold),
        "raw_text_f1": answer_f1(raw_prediction, gold),
    }
    for name, expected in expected_metrics.items():
        actual = row.get(name)
        if (
            not finite_number(actual)
            or not 0.0 <= float(actual) <= 1.0
            or abs(float(actual) - expected) > 1e-9
        ):
            return f"{name} is inconsistent with its prediction"
    support_titles = normalized_title_set(row.get("support_titles"))
    retrieved_titles = normalized_title_set(row.get("retrieved_titles"))
    if support_titles is None or not support_titles:
        return "support_titles must be nonempty strings"
    if retrieved_titles is None:
        return "retrieved_titles must be strings"
    expected_recall = len(support_titles & retrieved_titles) / len(support_titles)
    support_recall = row.get("support_recall")
    if (
        not finite_number(support_recall)
        or abs(float(support_recall) - expected_recall) > 1e-9
    ):
        return "support_recall is inconsistent with retrieved titles"
    return None


def episode_matches_source(row: dict, item: dict) -> bool:
    expected_support = [
        str(value).strip()
        for value in item.get("support_titles", [])
        if value is not None and str(value).strip()
    ]
    return (
        str(row.get("question_id", "")) == str(item.get("id", ""))
        and str(row.get("gold", "")) == str(item.get("answer", "")).strip()
        and row.get("support_titles") == expected_support
    )


def file_digest(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required run-identity file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(cfg: dict) -> dict:
    configured_model = str(cfg["llm"]["model"])
    model_path = (
        os.environ.get("MODEL_PATH") or os.environ.get("MODEL") or configured_model
    )
    identity = {
        "configured_model": configured_model,
        "model_path": model_path,
        "served_model_name": os.environ.get("SERVED_MODEL_NAME") or configured_model,
    }
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        identity["resolved_model_path"] = str(local_path.resolve())
        model_files = {
            path
            for pattern in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template*",
                "*.index.json",
                "*.safetensors",
                "*.bin",
            )
            for path in local_path.glob(pattern)
            if path.is_file()
        }
        local_files = []
        for path in sorted(model_files):
            state = {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if path.suffix not in {".safetensors", ".bin"}:
                state["sha256"] = file_digest(path)
            local_files.append(state)
        identity["local_files"] = local_files
    return identity


def run_identity(
    cfg: dict,
    work_dir: Path,
    backends: tuple[str, ...] | list[str] | None = None,
    evaluation_context: dict | None = None,
) -> tuple[str, dict]:
    selected = tuple(
        sorted(backends) if backends is not None else ("bm25", "e5", "colbert")
    )
    unknown = set(selected) - {"bm25", "e5", "colbert"}
    if unknown:
        raise ValueError(f"Unknown retriever backends: {sorted(unknown)}")
    artifacts = {
        "corpus": work_dir / "data" / "corpus.jsonl",
        "queries_eval": work_dir / "data" / "queries_eval.jsonl",
        "data_manifest": work_dir / "data" / ".pilot-manifest.json",
    }
    for backend in selected:
        artifacts[f"{backend}_manifest"] = (
            work_dir / "indexes" / backend / ".pilot-manifest.json"
        )
    payload = {
        "schema": RESULT_SCHEMA,
        "config": cfg,
        "artifacts": {name: file_digest(path) for name, path in artifacts.items()},
        "evaluator_files": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "action_protocol.py",
                "common.py",
                "react_agent_eval.py",
                "retrieval_clients.py",
            )
        },
        "model": model_identity(cfg),
    }
    if backends is not None:
        payload["retriever_backends"] = list(selected)
    if evaluation_context is not None:
        payload["evaluation_context"] = evaluation_context
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def format_results(results: list[dict], max_chars: int) -> str:
    parts = []
    for result in results:
        text = result["text"].replace("\n", " ")[:max_chars]
        parts.append(f"[{result['rank']}] {result['title']}: {text}")
    return "\n".join(parts)


def completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    cfg: dict,
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


def run_episode(
    client: OpenAI,
    model: str,
    retriever: RetrievalClient,
    item: dict,
    cfg: dict,
    oracle: bool,
) -> dict:
    gold_answer = str(item.get("answer", "")).strip()
    if not gold_answer:
        raise RuntimeError(
            f"Agent-evaluation row {item.get('id')!r} has no usable gold answer"
        )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = f"Question: {item['question']}"
    if oracle:
        user += "\n" + GUIDANCE[retriever.name]
    messages.append({"role": "user", "content": user})
    all_titles: list[str] = []
    searches: list[str] = []
    prediction = ""
    raw_text_prediction = ""
    invalid_action_count = 0
    protocol_failure = 0
    search_budget = int(cfg["agent"]["max_search_turns"])

    attempts = 0
    # Match Search-R1 training: malformed actions consume one of max_turns,
    # followed by one forced final action when no answer was produced.
    while attempts < search_budget:
        attempts += 1
        seed_payload = (
            f"{cfg['seed']}:{item['id']}:{retriever.name}:{int(oracle)}:{attempts}"
        )
        request_seed = int.from_bytes(
            hashlib.sha256(seed_payload.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = completion(client, model, messages, cfg, request_seed)
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            raw_text_prediction = value
            break
        if action != "search" or not value:
            invalid_action_count += 1
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid format. Output <search>...</search> or <answer>...</answer>.",
                }
            )
            continue
        results = retriever.search(value, int(cfg["retrieval"]["topk"]))
        searches.append(value)
        all_titles.extend(result["title"] for result in results)
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
        seed_payload = (
            f"{cfg['seed']}:{item['id']}:{retriever.name}:{int(oracle)}:final"
        )
        request_seed = int.from_bytes(
            hashlib.sha256(seed_payload.encode("utf-8")).digest()[:4], "big"
        ) % (2**31)
        content = completion(client, model, messages, cfg, request_seed)
        action, value = parse_action(content)
        raw_text_prediction = content.strip()
        if action == "answer":
            prediction = value
            raw_text_prediction = value
        else:
            prediction = ""
            protocol_failure = 1
            # Match Search-R1: a forced-final search is valid syntax but cannot
            # execute, so it fails the terminal-answer contract without being
            # counted as a malformed action.
            if action == "invalid":
                invalid_action_count += 1

    em = 0.0 if protocol_failure else answer_em(prediction, gold_answer)
    f1 = 0.0 if protocol_failure else answer_f1(prediction, gold_answer)
    raw_text_em = answer_em(raw_text_prediction, gold_answer)
    raw_text_f1 = answer_f1(raw_text_prediction, gold_answer)

    gold_titles = normalized_title_set(
        [
            str(value).strip()
            for value in item["support_titles"]
            if value is not None and str(value).strip()
        ]
    ) or set()
    got_titles = normalized_title_set([str(value) for value in all_titles]) or set()
    support_recall = len(gold_titles & got_titles) / max(1, len(gold_titles))
    return {
        "question_id": str(item["id"]),
        "backend": retriever.name,
        "variant": "oracle_guidance" if oracle else "blind",
        "prediction": prediction,
        "raw_text_prediction": raw_text_prediction,
        "gold": gold_answer,
        "support_titles": [
            str(value).strip()
            for value in item["support_titles"]
            if value is not None and str(value).strip()
        ],
        "retrieved_titles": [str(value) for value in all_titles],
        "em": em,
        "f1": f1,
        "raw_text_em": raw_text_em,
        "raw_text_f1": raw_text_f1,
        "protocol_failure": protocol_failure,
        "invalid_action_count": invalid_action_count,
        "support_recall": support_recall,
        "search_count": len(searches),
        "queries": searches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    check_retrievers(cfg)
    check_vllm(cfg)

    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results")
    output_path = out_dir / "agent_eval.jsonl"
    run_signature, _ = run_identity(cfg, work_dir)
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    limit = int(cfg["agent"]["eval_examples"]) if args.limit is None else args.limit
    queries = queries[:limit]

    r_cfg = cfg["retrieval"]
    retrievers = [
        RetrievalClient("bm25", f"http://127.0.0.1:{r_cfg['bm25_port']}/retrieve"),
        RetrievalClient("e5", f"http://127.0.0.1:{r_cfg['e5_port']}/retrieve"),
        RetrievalClient(
            "colbert", f"http://127.0.0.1:{r_cfg['colbert_port']}/retrieve"
        ),
    ]
    llm = OpenAI(
        base_url=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"],
        timeout=180.0,
        max_retries=5,
    )
    effective_model = effective_model_name(cfg)
    selected_ids = {str(item["id"]) for item in queries}
    item_by_id = {str(item["id"]): item for item in queries}
    expected_backends = {"bm25", "e5", "colbert"}
    expected_variants = {"blind", "oracle_guidance"}
    existing = read_jsonl_tolerant(output_path)
    row_by_key = {
        (
            str(row.get("question_id")),
            str(row.get("backend")),
            str(row.get("variant")),
        ): row
        for row in existing
        if row.get("schema") == RESULT_SCHEMA
        and row.get("run_signature") == run_signature
        and str(row.get("question_id")) in selected_ids
        and str(row.get("backend")) in expected_backends
        and str(row.get("variant")) in expected_variants
        and episode_matches_source(
            row, item_by_id[str(row.get("question_id"))]
        )
        and episode_validation_error(
            row, int(cfg["agent"]["max_search_turns"])
        )
        is None
    }

    total = len(queries) * len(retrievers) * 2
    progress = tqdm(total=total, desc="agent eval")
    for item in queries:
        for retriever in retrievers:
            for oracle in (False, True):
                variant = "oracle_guidance" if oracle else "blind"
                key = (str(item["id"]), retriever.name, variant)
                if key not in row_by_key:
                    row = run_episode(
                        llm, effective_model, retriever, item, cfg, oracle
                    )
                    row["schema"] = RESULT_SCHEMA
                    row["run_signature"] = run_signature
                    problem = episode_validation_error(
                        row, int(cfg["agent"]["max_search_turns"])
                    )
                    if problem is not None or not episode_matches_source(row, item):
                        raise RuntimeError(
                            f"New agent-evaluation row {key} is invalid: "
                            f"{problem or 'source row mismatch'}"
                        )
                    append_jsonl(output_path, [row])
                    row_by_key[key] = row
                progress.update(1)
    progress.close()

    rows = list(row_by_key.values())
    if len(rows) != total:
        raise RuntimeError(
            f"Expected exactly {total} agent-evaluation rows, found {len(rows)}"
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["backend", "variant"])[
            [
                "em",
                "f1",
                "raw_text_em",
                "raw_text_f1",
                "protocol_failure",
                "invalid_action_count",
                "support_recall",
                "search_count",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.insert(0, "run_signature", run_signature)
    summary.insert(1, "n_questions", len(queries))
    summary.to_csv(out_dir / "agent_eval_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Resumable episode results: {output_path}")


if __name__ == "__main__":
    main()

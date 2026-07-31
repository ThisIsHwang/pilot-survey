from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from stackpilot.behavior_alias_common import (
    SCHEMA,
    atomic_write_json,
    behavior_key,
    canonical_signature,
    choose_injection_class,
    file_sha256,
    load_config,
    natural_alias_metrics,
    normalize_query,
    normalize_title,
    read_jsonl,
    stable_hash,
    stable_seed,
    support_recall,
    valid_query,
)
from stackpilot.causal_query_replay import reconstruct_prefix
from stackpilot.observation_geometry import render_retrieval_observation
from stackpilot.retrieval_clients import RetrievalClient

_THREAD_LOCAL = threading.local()


def _client(cfg: dict[str, Any]) -> OpenAI:
    current = getattr(_THREAD_LOCAL, "client", None)
    if current is None:
        current = OpenAI(
            base_url=str(cfg["model"]["api_base"]),
            api_key=str(cfg["model"]["api_key"]),
            timeout=float(cfg["generation"]["request_timeout"]),
            max_retries=int(cfg["generation"]["request_retries"]),
        )
        _THREAD_LOCAL.client = current
    return current


def service_check(cfg: dict[str, Any]) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for backend in ("bm25", "e5"):
        port = int(cfg["retrieval"][f"{backend}_port"])
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or payload.get("backend") != backend:
            raise RuntimeError(f"Unexpected {backend} health response: {payload}")
        identities[backend] = payload
    response = requests.get(
        str(cfg["model"]["api_base"]).rstrip("/") + "/models", timeout=30
    )
    response.raise_for_status()
    model_ids = {str(row.get("id")) for row in response.json().get("data", [])}
    expected = str(cfg["model"]["served_model_name"])
    if expected not in model_ids:
        raise RuntimeError(f"vLLM does not serve {expected}; available={sorted(model_ids)}")
    identities["served_models"] = sorted(model_ids)
    return identities


def retriever_for(cfg: dict[str, Any], backend: str) -> RetrievalClient:
    port = int(cfg["retrieval"][f"{backend}_port"])
    return RetrievalClient(
        backend,
        f"http://127.0.0.1:{port}/retrieve",
        timeout=int(cfg["generation"]["request_timeout"]),
        retries=int(cfg["generation"]["request_retries"]),
    )


def generation_prompt(state: dict[str, Any], count: int) -> str:
    history: list[str] = []
    for turn in state["prior_turns"]:
        history.append(f"Previous query {turn['turn']}: {turn['query']}")
        titles = [
            str(value).strip()
            for value in turn.get("observed_titles", [])
            if str(value).strip()
        ]
        if titles:
            history.append("Observed titles: " + " | ".join(titles[:12]))
    return "\n".join(
        [
            "Generate one possible NEXT search query for this unresolved question.",
            "Return only the query, with no XML tags, numbering, answer, or explanation.",
            "Keep the query grounded in the question and already observed entities.",
            "Different samples should explore different lexical forms and information paths.",
            f"The experiment will sample {count} independent candidates from this prompt.",
            "",
            f"Question: {state['question']}",
            *history,
            "Next search query:",
        ]
    )


def generate_queries(
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
    count: int,
) -> list[str]:
    include_factual = bool(cfg["generation"].get("include_factual", True))
    queries: list[str] = []
    seen: set[str] = set()
    if include_factual:
        factual = normalize_query(str(state["factual_query"]))
        if factual:
            queries.append(factual)
            seen.add(factual.lower())
    observed_titles = [
        str(title)
        for turn in state["prior_turns"]
        for title in turn.get("observed_titles", [])
    ]
    prompt = generation_prompt(state, count)
    attempts = int(cfg["generation"]["attempts"])
    per_call = int(cfg["generation"]["samples_per_call"])
    for attempt in range(attempts):
        remaining = count - len(queries)
        if remaining <= 0:
            break
        response = _client(cfg).chat.completions.create(
            model=str(cfg["model"]["served_model_name"]),
            messages=[
                {
                    "role": "system",
                    "content": "You generate grounded search queries and output only the query text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(cfg["generation"]["temperature"]),
            max_tokens=int(cfg["generation"]["max_tokens"]),
            n=min(per_call, remaining),
            seed=stable_seed("behavior-alias", state["state_id"], attempt),
        )
        for choice in response.choices:
            query = normalize_query(choice.message.content or "")
            key = query.lower()
            if key in seen:
                continue
            if not valid_query(
                query,
                question=str(state["question"]),
                observed_titles=observed_titles,
                minimum_tokens=int(cfg["generation"]["minimum_tokens"]),
                maximum_tokens=int(cfg["generation"]["maximum_tokens"]),
            ):
                continue
            queries.append(query)
            seen.add(key)
            if len(queries) >= count:
                break
    if len(queries) < count:
        raise RuntimeError(
            f"State {state['state_id']} produced only {len(queries)}/{count} valid unique queries"
        )
    return queries[:count]


def evaluate_query(
    state: dict[str, Any],
    query: str,
    *,
    cfg: dict[str, Any],
    retriever: RetrievalClient,
    tokenizer: Any,
    prefix_titles: set[str],
    prefix_recall: float,
) -> dict[str, Any]:
    results = retriever.search(query, int(state["topk"]))
    rendered = render_retrieval_observation(
        results,
        tokenizer,
        int(cfg["agent"]["observation_token_budget"]),
    )
    observed_titles = list(rendered.observed_titles)
    visible_ids = [
        str(results[index].get("id") or normalize_title(observed_titles[index]))
        for index in range(min(len(observed_titles), len(results)))
    ]
    cumulative = set(prefix_titles)
    cumulative.update(normalize_title(value) for value in observed_titles)
    after = support_recall(state["support_titles"], cumulative)
    gain = after - prefix_recall
    return {
        "candidate_id": stable_hash(state["state_id"], query),
        "query": query,
        "observed_titles": observed_titles,
        "retrieved_titles": list(rendered.retrieved_titles),
        "observed_document_ids": visible_ids,
        "behavior_class_id": behavior_key(visible_ids),
        "support_recall": after,
        "support_gain": gain,
        "reward": gain,
        "observation_token_count": int(rendered.token_count),
        "observation_truncated": int(rendered.truncated),
    }


def summarize_classes(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["behavior_class_id"]), []).append(row)
    classes: list[dict[str, Any]] = []
    for class_id, rows in grouped.items():
        queries = sorted(
            {str(row["query"]) for row in rows},
            key=lambda value: (len(value), value),
        )
        observed = list(rows[0]["observed_titles"])
        document_ids = list(rows[0]["observed_document_ids"])
        if any(list(row["observed_document_ids"]) != document_ids for row in rows):
            raise RuntimeError(
                f"Exact behavior class {class_id} contains different document sequences"
            )
        classes.append(
            {
                "class_id": class_id,
                "observed_titles": observed,
                "observed_document_ids": document_ids,
                "queries": queries,
                "canonical_query": queries[0],
                "natural_alias_count": len(queries),
                "surface_sample_count": len(rows),
                "support_gain": float(
                    sum(float(row["support_gain"]) for row in rows) / len(rows)
                ),
                "reward": float(
                    sum(float(row["reward"]) for row in rows) / len(rows)
                ),
            }
        )
    return sorted(classes, key=lambda row: str(row["class_id"]))


def process_state(
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
    tokenizer: Any,
    candidate_count: int,
    run_signature: str,
    output_root: Path,
) -> dict[str, Any]:
    destination = output_root / str(state["backend"]) / f"{state['state_id']}.json"
    state_signature = canonical_signature(
        {
            "run_signature": run_signature,
            "state": state,
            "candidate_count": candidate_count,
        }
    )
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("state_signature") == state_signature:
            return existing
        destination.rename(destination.with_suffix(f".stale.{int(time.time())}.json"))

    retriever = retriever_for(cfg, str(state["backend"]))
    causal_cfg = {
        "agent": cfg["agent"],
        "validation": cfg["validation"],
    }
    prefix = reconstruct_prefix(
        state,
        cfg=causal_cfg,
        retriever=retriever,
        tokenizer=tokenizer,
    )
    queries = generate_queries(state, cfg=cfg, count=candidate_count)
    candidates = [
        evaluate_query(
            state,
            query,
            cfg=cfg,
            retriever=retriever,
            tokenizer=tokenizer,
            prefix_titles=set(prefix["observed_titles"]),
            prefix_recall=float(prefix["prefix_recall"]),
        )
        for query in queries
    ]

    if bool(cfg["generation"].get("include_factual", True)):
        factual = candidates[0]
        raw_titles = [
            normalize_title(value)
            for value in state.get("raw_factual_observed_titles", [])
        ]
        replay_titles = [normalize_title(value) for value in factual["observed_titles"]]
        intersection = len(set(raw_titles) & set(replay_titles))
        union = len(set(raw_titles) | set(replay_titles))
        title_jaccard = intersection / max(1, union)
        if title_jaccard < float(cfg["validation"]["minimum_factual_title_jaccard"]):
            raise RuntimeError(
                f"State {state['state_id']} factual title replay mismatch: {title_jaccard:.4f}"
            )
        if abs(
            float(factual["support_gain"])
            - float(state["raw_factual_evidence_gain"])
        ) > float(cfg["validation"]["recall_tolerance"]):
            raise RuntimeError(
                f"State {state['state_id']} factual evidence-gain mismatch"
            )
    else:
        title_jaccard = 1.0

    classes = summarize_classes(candidates)
    injection = choose_injection_class(
        classes,
        minimum_reward_gap=float(cfg["injection"]["minimum_reward_gap"]),
    )
    result = {
        "schema": SCHEMA,
        "state_signature": state_signature,
        "run_signature": run_signature,
        "state_id": state["state_id"],
        "question_id": state["question_id"],
        "question": state["question"],
        "dataset": state["dataset"],
        "backend": state["backend"],
        "topk": int(state["topk"]),
        "source_turn": int(state["source_turn"]),
        "policy_tag": state["policy_tag"],
        "policy_seed": int(state["policy_seed"]),
        "support_titles": list(state["support_titles"]),
        "prior_turns": list(state["prior_turns"]),
        "prefix_support_recall": float(prefix["prefix_recall"]),
        "factual_title_jaccard": title_jaccard,
        "candidates": candidates,
        "classes": classes,
        "injection_class_id": injection,
        "eligible_for_injection": int(injection is not None),
        "natural_alias_metrics": natural_alias_metrics(candidates),
    }
    atomic_write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and execute EXP-015 query pools."
    )
    parser.add_argument("--config", default="configs/behavior_alias_pilot.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--states-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    if args.model:
        cfg["model"]["base_model"] = str(Path(args.model).resolve())
    states_root = Path(
        args.states_root or Path(cfg["work_dir"]) / "states" / args.profile
    ).resolve()
    manifest_path = states_root / "manifest.json"
    states_path = states_root / "states.jsonl"
    if not manifest_path.is_file() or not states_path.is_file():
        raise RuntimeError(f"Missing prepared state bank under {states_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != file_sha256(states_path):
        raise RuntimeError("Prepared state bank does not match its manifest")
    states = read_jsonl(states_path)

    identities = service_check(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    candidate_count = int(profile["candidate_pool_size"])
    run_payload = {
        "schema": SCHEMA,
        "experiment_id": "EXP-015",
        "profile": args.profile,
        "config_sha256": file_sha256(args.config),
        "states_manifest_signature": manifest["signature"],
        "model": cfg["model"],
        "services": identities,
        "candidate_pool_size": candidate_count,
    }
    run_signature = canonical_signature(run_payload)
    output_root = Path(
        args.output_root
        or Path(cfg["work_dir"]) / "results" / args.profile / "states"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    workers = int(args.workers or profile["workers"])
    failures: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_state,
                state,
                cfg=cfg,
                tokenizer=tokenizer,
                candidate_count=candidate_count,
                run_signature=run_signature,
                output_root=output_root,
            ): state
            for state in states
        }
        for future in as_completed(futures):
            state = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"state_id": str(state["state_id"]), "error": repr(exc)}
                )

    summary = {
        **run_payload,
        "run_signature": run_signature,
        "requested_states": len(states),
        "completed_states": len(completed),
        "eligible_injection_states": sum(
            int(row["eligible_for_injection"]) for row in completed
        ),
        "failures": failures,
        "success": not failures and len(completed) == len(states),
    }
    atomic_write_json(output_root.parent / "run_summary.json", summary)
    if failures:
        raise SystemExit(
            f"{len(failures)} behavior-alias states failed; see run_summary.json"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

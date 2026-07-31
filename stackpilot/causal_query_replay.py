from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from stackpilot.action_protocol import parse_action
from stackpilot.causal_query_common import (
    attach_query_effects,
    canonical_signature,
    jaccard,
    load_causal_query_config,
    normalize_title,
    parse_alternative_queries,
    stable_hash,
    stable_seed,
    support_recall,
    token_set,
    transferred_bridge_tokens,
    validate_alternatives,
    word_tokens,
)
from stackpilot.common import answer_em, answer_f1
from stackpilot.observation_geometry import render_retrieval_observation
from stackpilot.react_agent_eval import SYSTEM_PROMPT
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.trace_common import (
    atomic_write_json,
    file_sha256,
    read_jsonl,
)

RUN_SCHEMA = 1
FORMAT_CORRECTION = (
    "Invalid format. Output exactly one <search>query</search> or "
    "<answer>short answer</answer> action."
)
_THREAD_LOCAL = threading.local()


def _client(cfg: dict[str, Any]) -> OpenAI:
    current = getattr(_THREAD_LOCAL, "client", None)
    if current is None:
        current = OpenAI(
            base_url=str(cfg["model"]["api_base"]),
            api_key=str(cfg["model"]["api_key"]),
            timeout=float(cfg["continuation"]["request_timeout"]),
            max_retries=int(cfg["continuation"]["request_retries"]),
        )
        _THREAD_LOCAL.client = current
    return current


def _complete(
    cfg: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    response = _client(cfg).chat.completions.create(
        model=str(cfg["model"]["served_model_name"]),
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        seed=int(seed),
    )
    return response.choices[0].message.content or ""


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _best_answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    return (
        max(answer_em(prediction, answer) for answer in answers),
        max(answer_f1(prediction, answer) for answer in answers),
    )


def _service_check(cfg: dict[str, Any]) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for backend in ("bm25", "e5"):
        port = int(cfg["retrieval"][f"{backend}_port"])
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or payload.get("backend") != backend:
            raise RuntimeError(f"Unexpected {backend} health response: {payload}")
        if backend == "e5":
            if payload.get("faiss_gpu") is not True:
                raise RuntimeError(f"E5 is not using FAISS GPU: {payload}")
            expected_gpu = str(cfg["retrieval"]["e5_gpu"])
            if str(payload.get("cuda_visible_devices")) != expected_gpu:
                raise RuntimeError(
                    f"E5 uses GPU {payload.get('cuda_visible_devices')}, expected {expected_gpu}"
                )
        identities[backend] = payload
    models = requests.get(
        str(cfg["model"]["api_base"]).rstrip("/") + "/models", timeout=30
    )
    models.raise_for_status()
    model_ids = {str(row.get("id")) for row in models.json().get("data", [])}
    expected = str(cfg["model"]["served_model_name"])
    if expected not in model_ids:
        raise RuntimeError(f"vLLM does not serve {expected}; available={sorted(model_ids)}")
    identities["served_models"] = sorted(model_ids)
    return identities


def _retrievers(cfg: dict[str, Any]) -> dict[str, RetrievalClient]:
    return {
        backend: RetrievalClient(
            backend,
            f"http://127.0.0.1:{int(cfg['retrieval'][f'{backend}_port'])}/retrieve",
            timeout=int(cfg["continuation"]["request_timeout"]),
            retries=int(cfg["continuation"]["request_retries"]),
        )
        for backend in ("bm25", "e5")
    }


def _render_query(
    retriever: RetrievalClient,
    tokenizer: Any,
    query: str,
    *,
    topk: int,
    token_budget: int,
) -> tuple[list[dict[str, Any]], Any]:
    results = retriever.search(query, topk)
    rendered = render_retrieval_observation(results, tokenizer, token_budget)
    return results, rendered


def _title_jaccard(left: list[str], right: list[str]) -> float:
    return jaccard(
        {normalize_title(value) for value in left},
        {normalize_title(value) for value in right},
    )


def reconstruct_prefix(
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
    retriever: RetrievalClient,
    tokenizer: Any,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {state['question']}"},
    ]
    cumulative_observed: set[str] = set()
    prefix_records: list[dict[str, Any]] = []
    token_budget = int(cfg["agent"]["observation_token_budget"])
    tolerance = float(cfg["validation"]["recall_tolerance"])
    minimum_jaccard = float(cfg["validation"]["minimum_prefix_title_jaccard"])

    for raw_turn in state["prior_turns"]:
        query = str(raw_turn["query"])
        _results, rendered = _render_query(
            retriever,
            tokenizer,
            query,
            topk=int(state["topk"]),
            token_budget=token_budget,
        )
        observed_titles = list(rendered.observed_titles)
        cumulative_observed.update(normalize_title(value) for value in observed_titles)
        recall = support_recall(state["support_titles"], cumulative_observed)
        raw_titles = _strings(raw_turn.get("observed_titles"))
        title_match = _title_jaccard(raw_titles, observed_titles) if raw_titles else 1.0
        if title_match < minimum_jaccard:
            raise RuntimeError(
                f"State {state['state_id']} prefix turn {raw_turn['turn']} title mismatch: "
                f"jaccard={title_match:.4f}"
            )
        if abs(recall - float(raw_turn.get("support_recall", recall))) > tolerance:
            raise RuntimeError(
                f"State {state['state_id']} prefix recall mismatch at turn "
                f"{raw_turn['turn']}: replay={recall}, raw={raw_turn.get('support_recall')}"
            )
        messages.append({"role": "assistant", "content": f"<search>{query}</search>"})
        messages.append({"role": "user", "content": rendered.visible_text})
        prefix_records.append(
            {
                "turn": int(raw_turn["turn"]),
                "query": query,
                "retrieved_titles": list(rendered.retrieved_titles),
                "observed_titles": observed_titles,
                "support_recall": recall,
                "observation_token_count": int(rendered.token_count),
                "observation_truncated": int(rendered.truncated),
                "raw_title_jaccard": title_match,
            }
        )
    return {
        "messages": messages,
        "records": prefix_records,
        "observed_titles": cumulative_observed,
        "prefix_recall": (
            float(prefix_records[-1]["support_recall"]) if prefix_records else 0.0
        ),
    }


def _alternative_prompt(state: dict[str, Any], styles: list[str]) -> str:
    history_lines = []
    for turn in state["prior_turns"]:
        history_lines.append(f"Previous query {turn['turn']}: {turn['query']}")
        titles = _strings(turn.get("observed_titles"))
        if titles:
            history_lines.append("Observed titles: " + " | ".join(titles[:10]))
    fields = ", ".join(f'"{style}": "..."' for style in styles)
    return "\n".join(
        [
            "Generate state-matched alternative NEXT search queries.",
            "Each query must pursue the same unresolved information need as the factual query.",
            "Do not answer the question. Do not copy the factual query exactly.",
            "Keep known entities, avoid invented facts, and return only one JSON object.",
            f"Required JSON keys: {fields}",
            "Styles:",
            "- lexical: concise rare entities and relation keywords",
            "- semantic: fluent natural-language paraphrase of the same search intent",
            "- entity: focus on the most useful discovered entity and missing relation",
            "",
            f"Question: {state['question']}",
            *history_lines,
            f"Factual next query: {state['factual_query']}",
        ]
    )


def generate_alternatives(
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
    count: int,
) -> list[dict[str, str]]:
    styles = [str(value) for value in cfg["alternatives"]["styles"]][:count]
    if len(styles) != count:
        raise RuntimeError(f"Config provides only {len(styles)} styles for {count} alternatives")
    prompt = _alternative_prompt(state, styles)
    accumulated: dict[str, str] = {}
    observed_titles = [
        title
        for turn in state["prior_turns"]
        for title in _strings(turn.get("observed_titles"))
    ]
    for attempt in range(int(cfg["alternatives"]["attempts"])):
        output = _complete(
            cfg,
            [
                {
                    "role": "system",
                    "content": "You create controlled counterfactual search queries and output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(cfg["alternatives"]["temperature"]),
            max_tokens=int(cfg["alternatives"]["max_tokens"]),
            seed=stable_seed("alternatives", state["state_id"], attempt),
        )
        parsed = parse_alternative_queries(output, styles)
        accumulated.update(parsed)
        valid = validate_alternatives(
            accumulated,
            styles=styles,
            factual_query=str(state["factual_query"]),
            question=str(state["question"]),
            observed_titles=observed_titles,
            minimum_tokens=int(cfg["alternatives"]["minimum_tokens"]),
            length_ratio_low=float(cfg["alternatives"]["length_ratio_low"]),
            length_ratio_high=float(cfg["alternatives"]["length_ratio_high"]),
        )
        if len(valid) == count:
            return [
                {"style": style, "query": valid[style], "origin": "alternative"}
                for style in styles
            ]
        missing = [style for style in styles if style not in valid]
        prompt += "\nRegenerate all keys. Previously invalid or missing styles: " + ", ".join(missing)
    raise RuntimeError(
        f"State {state['state_id']} produced only {len(valid)}/{count} valid alternatives: {valid}"
    )


def run_branch(
    state: dict[str, Any],
    candidate: dict[str, str],
    *,
    cfg: dict[str, Any],
    retriever: RetrievalClient,
    tokenizer: Any,
    prefix: dict[str, Any],
) -> dict[str, Any]:
    messages = copy.deepcopy(prefix["messages"])
    cumulative_observed = set(prefix["observed_titles"])
    prior_observed = set(cumulative_observed)
    prior_text_parts = [str(state["question"])]
    prior_text_parts.extend(str(turn["query"]) for turn in state["prior_turns"])
    prior_text_parts.extend(
        title
        for turn in state["prior_turns"]
        for title in _strings(turn.get("observed_titles"))
    )
    prior_text = " ".join(prior_text_parts)
    token_budget = int(cfg["agent"]["observation_token_budget"])
    max_turns = int(cfg["agent"]["max_search_turns"])
    candidate_query = str(candidate["query"])

    branch_records: list[dict[str, Any]] = []
    _results, rendered = _render_query(
        retriever,
        tokenizer,
        candidate_query,
        topk=int(state["topk"]),
        token_budget=token_budget,
    )
    candidate_observed = list(rendered.observed_titles)
    cumulative_observed.update(normalize_title(value) for value in candidate_observed)
    recall_after_candidate = support_recall(state["support_titles"], cumulative_observed)
    immediate_gain = recall_after_candidate - float(prefix["prefix_recall"])
    new_titles = sorted(
        {normalize_title(value) for value in candidate_observed} - prior_observed
    )
    messages.append(
        {"role": "assistant", "content": f"<search>{candidate_query}</search>"}
    )
    messages.append({"role": "user", "content": rendered.visible_text})
    branch_records.append(
        {
            "turn": int(state["source_turn"]),
            "query": candidate_query,
            "retrieved_titles": list(rendered.retrieved_titles),
            "observed_titles": candidate_observed,
            "support_recall": recall_after_candidate,
            "evidence_gain": immediate_gain,
            "new_observed_titles": new_titles,
            "observation_token_count": int(rendered.token_count),
            "observation_truncated": int(rendered.truncated),
        }
    )

    prediction = ""
    protocol_failure = 0
    invalid_action_count = 0
    attempts = int(state["source_turn"])
    next_query = ""
    next_query_gain = 0.0
    while attempts < max_turns:
        attempts += 1
        content = _complete(
            cfg,
            messages,
            temperature=float(cfg["continuation"]["temperature"]),
            max_tokens=int(cfg["continuation"]["max_tokens"]),
            seed=stable_seed(
                "suffix",
                state["state_id"],
                candidate["style"],
                candidate_query,
                attempts,
            ),
        )
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search" or not value:
            invalid_action_count += 1
            messages.append({"role": "user", "content": FORMAT_CORRECTION})
            continue

        previous_recall = support_recall(state["support_titles"], cumulative_observed)
        _results, suffix_rendered = _render_query(
            retriever,
            tokenizer,
            value,
            topk=int(state["topk"]),
            token_budget=token_budget,
        )
        suffix_observed = list(suffix_rendered.observed_titles)
        cumulative_observed.update(normalize_title(title) for title in suffix_observed)
        current_recall = support_recall(state["support_titles"], cumulative_observed)
        evidence_gain = current_recall - previous_recall
        if not next_query:
            next_query = value
            next_query_gain = evidence_gain
        branch_records.append(
            {
                "turn": len(state["prior_turns"]) + len(branch_records) + 1,
                "query": value,
                "retrieved_titles": list(suffix_rendered.retrieved_titles),
                "observed_titles": suffix_observed,
                "support_recall": current_recall,
                "evidence_gain": evidence_gain,
                "new_observed_titles": sorted(
                    {normalize_title(title) for title in suffix_observed}
                    - set().union(
                        *[
                            {normalize_title(title) for title in record["observed_titles"]}
                            for record in branch_records
                        ]
                    )
                ),
                "observation_token_count": int(suffix_rendered.token_count),
                "observation_truncated": int(suffix_rendered.truncated),
            }
        )
        messages.append({"role": "user", "content": suffix_rendered.visible_text})

    if not prediction:
        messages.append(
            {
                "role": "user",
                "content": "The search budget is exhausted. Give your best final answer now as <answer>short answer</answer>.",
            }
        )
        content = _complete(
            cfg,
            messages,
            temperature=float(cfg["continuation"]["temperature"]),
            max_tokens=int(cfg["continuation"]["max_tokens"]),
            seed=stable_seed(
                "suffix-final", state["state_id"], candidate["style"], candidate_query
            ),
        )
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
        else:
            protocol_failure = 1
            if action == "invalid":
                invalid_action_count += 1

    answers = _strings(state["answers"])
    em, f1 = (0.0, 0.0) if protocol_failure else _best_answer_scores(prediction, answers)
    final_recall = support_recall(state["support_titles"], cumulative_observed)
    bridge_tokens = transferred_bridge_tokens(
        next_query=next_query,
        intervention_titles=candidate_observed,
        prior_text=prior_text,
    )
    previous_query = (
        str(state["prior_turns"][-1]["query"]) if state["prior_turns"] else ""
    )
    question_tokens = token_set(str(state["question"]), content_only=True)
    candidate_tokens = token_set(candidate_query, content_only=True)
    previous_tokens = token_set(previous_query, content_only=True)
    return {
        "candidate_id": stable_hash(state["state_id"], candidate["style"], candidate_query),
        "style": str(candidate["style"]),
        "origin": str(candidate["origin"]),
        "query": candidate_query,
        "query_token_count": len(word_tokens(candidate_query)),
        "query_question_overlap": len(question_tokens & candidate_tokens)
        / max(1, len(question_tokens)),
        "query_previous_change": 1.0
        - len(candidate_tokens & previous_tokens)
        / max(1, len(candidate_tokens | previous_tokens)),
        "intervention_retrieved_titles": list(rendered.retrieved_titles),
        "intervention_observed_titles": candidate_observed,
        "intervention_result_novelty": 1.0
        - jaccard(
            {normalize_title(value) for value in candidate_observed},
            prior_observed,
        ),
        "immediate_support_gain": immediate_gain,
        "recall_after_intervention": recall_after_candidate,
        "next_query": next_query,
        "next_query_evidence_gain": next_query_gain,
        "transferred_bridge_tokens": bridge_tokens,
        "transferred_bridge_token_count": len(bridge_tokens),
        "final_support_recall": final_recall,
        "answer_prediction": prediction,
        "answer_em": em,
        "answer_f1": f1,
        "protocol_failure": protocol_failure,
        "invalid_action_count": invalid_action_count,
        "total_search_count": len(state["prior_turns"]) + len(branch_records),
        "suffix_search_count": max(0, len(branch_records) - 1),
        "branch_turns": branch_records,
    }


def process_state(
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
    tokenizer: Any,
    run_signature: str,
    output_root: Path,
    alternatives_per_state: int,
) -> dict[str, Any]:
    destination = output_root / str(state["backend"]) / f"{state['state_id']}.json"
    state_signature = canonical_signature(
        {
            "run_signature": run_signature,
            "state": state,
            "alternatives_per_state": alternatives_per_state,
        }
    )
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("state_signature") == state_signature:
            return existing
        stale = destination.with_suffix(f".stale.{int(time.time())}.json")
        destination.rename(stale)

    retriever = _retrievers(cfg)[str(state["backend"])]
    prefix = reconstruct_prefix(
        state,
        cfg=cfg,
        retriever=retriever,
        tokenizer=tokenizer,
    )
    alternatives = generate_alternatives(
        state,
        cfg=cfg,
        count=alternatives_per_state,
    )
    candidates = [
        {
            "style": "factual",
            "query": str(state["factual_query"]),
            "origin": "factual",
        },
        *alternatives,
    ]
    branches = [
        run_branch(
            state,
            candidate,
            cfg=cfg,
            retriever=retriever,
            tokenizer=tokenizer,
            prefix=prefix,
        )
        for candidate in candidates
    ]

    factual = branches[0]
    tolerance = float(cfg["validation"]["recall_tolerance"])
    factual_title_match = _title_jaccard(
        _strings(state.get("raw_factual_observed_titles")),
        _strings(factual.get("intervention_observed_titles")),
    )
    if factual_title_match < float(cfg["validation"]["minimum_factual_title_jaccard"]):
        raise RuntimeError(
            f"State {state['state_id']} factual replay title mismatch: {factual_title_match:.4f}"
        )
    if abs(
        float(factual["immediate_support_gain"])
        - float(state["raw_factual_evidence_gain"])
    ) > tolerance:
        raise RuntimeError(
            f"State {state['state_id']} factual gain mismatch: "
            f"replay={factual['immediate_support_gain']} raw={state['raw_factual_evidence_gain']}"
        )
    if abs(
        float(factual["recall_after_intervention"])
        - float(state["raw_factual_support_recall"])
    ) > tolerance:
        raise RuntimeError(
            f"State {state['state_id']} factual recall mismatch: "
            f"replay={factual['recall_after_intervention']} raw={state['raw_factual_support_recall']}"
        )

    branches = attach_query_effects(
        branches,
        answer_weight=float(cfg["agent"]["answer_weight"]),
        search_cost=float(cfg["agent"]["search_cost"]),
        epsilon=float(cfg["analysis"]["epsilon"]),
        bridge_min_support_tqe=float(cfg["analysis"]["bridge_min_support_tqe"]),
    )
    result = {
        "schema": RUN_SCHEMA,
        "state_signature": state_signature,
        "run_signature": run_signature,
        "state": state,
        "prefix": {
            "records": prefix["records"],
            "prefix_recall": prefix["prefix_recall"],
        },
        "factual_title_jaccard": factual_title_match,
        "candidates": branches,
    }
    atomic_write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run state-matched query interventions and suffix replay for EXP-013."
    )
    parser.add_argument("--config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--states-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_causal_query_config(args.config)
    if args.model:
        cfg["model"]["base_model"] = args.model
    profile = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    states_root = Path(args.states_root or work_root / "states" / args.profile)
    states_path = states_root / "states.jsonl"
    manifest_path = states_root / "manifest.json"
    if not states_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing prepared states under {states_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("states_sha256") != file_sha256(states_path):
        raise RuntimeError("Prepared causal-query states do not match their manifest")
    if manifest.get("config_sha256") != file_sha256(args.config):
        raise RuntimeError("Causal-query config changed after state preparation; rerun prepare")

    identities = _service_check(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        local_files_only=Path(str(cfg["model"]["base_model"])).is_dir(),
    )
    states = read_jsonl(states_path)
    evaluator_files = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in (
            "causal_query_common.py",
            "causal_query_replay.py",
            "action_protocol.py",
            "observation_geometry.py",
            "retrieval_clients.py",
            "react_agent_eval.py",
        )
    }
    run_signature = canonical_signature(
        {
            "schema": RUN_SCHEMA,
            "profile": args.profile,
            "config": cfg,
            "states_manifest_signature": manifest.get("signature"),
            "services": identities,
            "evaluator_files": evaluator_files,
        }
    )
    output_root = Path(
        args.output_root or work_root / "results" / args.profile / "states"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or profile["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    alternatives_per_state = int(profile["alternatives_per_state"])

    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                process_state,
                state,
                cfg=cfg,
                tokenizer=tokenizer,
                run_signature=run_signature,
                output_root=output_root,
                alternatives_per_state=alternatives_per_state,
            ): state
            for state in states
        }
        for future in as_completed(future_map):
            state = future_map[future]
            try:
                future.result()
                completed += 1
                print(
                    f"completed {completed}/{len(states)}: {state['backend']} {state['state_id']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - preserve all state failures
                failures.append(
                    {
                        "state_id": str(state["state_id"]),
                        "backend": str(state["backend"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"FAILED {state['state_id']}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
    summary = {
        "schema": RUN_SCHEMA,
        "profile": args.profile,
        "run_signature": run_signature,
        "states": len(states),
        "completed": completed,
        "failures": failures,
        "output_root": str(output_root),
    }
    atomic_write_json(output_root.parent / "run_summary.json", summary)
    if failures:
        raise SystemExit(
            f"{len(failures)} causal-query states failed; see {output_root.parent / 'run_summary.json'}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

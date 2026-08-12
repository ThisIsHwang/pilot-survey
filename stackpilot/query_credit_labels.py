from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from stackpilot.action_protocol import parse_action
from stackpilot.causal_query_common import (
    load_causal_query_config,
    normalize_title,
    support_recall,
)
from stackpilot.causal_query_replay import (
    FORMAT_CORRECTION,
    _best_answer_scores,
    _complete,
    _retrievers,
    _service_check,
    reconstruct_prefix,
)
from stackpilot.interface_causality_common import load_state_results
from stackpilot.observation_geometry import render_retrieval_observation
from stackpilot.query_credit_common import (
    aggregate_document_credit,
    atomic_write_json,
    atomic_write_jsonl,
    behavior_signature,
    best_replacement_gap,
    centered_action_advantage,
    composite_reward,
    load_config,
    normalize_text,
    stable_hash,
    stable_seed,
)


def discover_inputs(cfg: dict[str, Any], provided: Sequence[str] | None) -> list[str]:
    patterns = list(provided or [])
    if not patterns:
        environment = os.environ.get("QUERY_CREDIT_INPUTS", "").strip()
        if environment:
            patterns = [part for part in environment.split(os.pathsep) if part]
        else:
            patterns = [str(value) for value in cfg["source"]["state_globs"]]
    paths: set[str] = set()
    for pattern in patterns:
        for value in glob.glob(os.path.expanduser(pattern), recursive=True):
            if Path(value).is_file():
                paths.add(str(Path(value).resolve()))
    if not paths:
        raise RuntimeError("No causal-query state files were found")
    return sorted(paths)


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _document_parts(item: dict[str, Any], rank: int) -> dict[str, Any]:
    document = item.get("document")
    title = ""
    text = ""
    if isinstance(document, dict):
        contents = str(document.get("contents") or "")
        title, separator, text = contents.partition("\n")
        if not separator:
            text = ""
    else:
        title = str(item.get("title") or item.get("document_title") or "")
        text = str(item.get("text") or item.get("content") or "")
    score = item.get("score", item.get("retrieval_score", 0.0))
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    return {
        "document_rank": rank,
        "document_title": title.strip(),
        "document_text": text.strip(),
        "retriever_score": numeric_score,
    }


def _candidate_rows(result: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in result.get("candidates", []):
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        query = str(candidate.get("query", "")).strip()
        normalized = normalize_text(query)
        if not query or normalized in seen:
            continue
        seen.add(normalized)
        output.append(
            {
                "query": query,
                "style": str(candidate.get("style", candidate.get("origin", "unknown"))),
                "origin": str(candidate.get("origin", "alternative")),
            }
        )
        if len(output) >= maximum:
            break
    return output


def _reward_from_replay(replay: dict[str, Any], weights: dict[str, float]) -> float:
    return composite_reward(
        support_recall=float(replay["final_support_recall"]),
        answer_f1=float(replay["answer_f1"]),
        search_count=float(replay["search_count"]),
        invalid_action_count=float(replay["invalid_action_count"]),
        weights=weights,
    )


def replay_after_observation(
    state: dict[str, Any],
    *,
    causal_cfg: dict[str, Any],
    retriever: Any,
    tokenizer: Any,
    prefix: dict[str, Any],
    query: str,
    results: list[dict[str, Any]],
    continuation_seed: int,
) -> dict[str, Any]:
    token_budget = int(causal_cfg["agent"]["observation_token_budget"])
    rendered = render_retrieval_observation(results, tokenizer, token_budget)
    messages = copy.deepcopy(prefix["messages"])
    messages.append({"role": "assistant", "content": f"<search>{query}</search>"})
    messages.append({"role": "user", "content": rendered.visible_text})
    cumulative = set(prefix["observed_titles"])
    cumulative.update(normalize_title(value) for value in rendered.observed_titles)
    records = [
        {
            "turn": int(state["source_turn"]),
            "query": query,
            "observed_titles": list(rendered.observed_titles),
            "support_recall": support_recall(state["support_titles"], cumulative),
        }
    ]
    prediction = ""
    invalid = 0
    attempts = int(state["source_turn"])
    max_turns = int(causal_cfg["agent"]["max_search_turns"])
    while attempts < max_turns:
        attempts += 1
        content = _complete(
            causal_cfg,
            messages,
            temperature=float(causal_cfg["continuation"]["temperature"]),
            max_tokens=int(causal_cfg["continuation"]["max_tokens"]),
            seed=stable_seed(
                "query-credit-common-random-number",
                state["state_id"],
                continuation_seed,
                attempts,
            ),
        )
        messages.append({"role": "assistant", "content": content})
        action, value = parse_action(content)
        if action == "answer":
            prediction = value
            break
        if action != "search" or not value:
            invalid += 1
            messages.append({"role": "user", "content": FORMAT_CORRECTION})
            continue
        suffix_results = retriever.search(value, int(state["topk"]))
        suffix_rendered = render_retrieval_observation(
            suffix_results, tokenizer, token_budget
        )
        cumulative.update(normalize_title(title) for title in suffix_rendered.observed_titles)
        records.append(
            {
                "turn": attempts,
                "query": value,
                "observed_titles": list(suffix_rendered.observed_titles),
                "support_recall": support_recall(state["support_titles"], cumulative),
            }
        )
        messages.append({"role": "user", "content": suffix_rendered.visible_text})
    answers = _strings(state.get("answers"))
    if not answers and state.get("answer"):
        answers = [str(state["answer"])]
    answer_em, answer_f1 = _best_answer_scores(prediction, answers) if answers else (0.0, 0.0)
    return {
        "prediction": prediction,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "final_support_recall": support_recall(state["support_titles"], cumulative),
        "search_count": len(records),
        "invalid_action_count": invalid,
        "records": records,
        "observed_titles": list(rendered.observed_titles),
        "retrieved_titles": list(rendered.retrieved_titles),
        "observation_token_count": int(rendered.token_count),
        "observation_truncated": int(rendered.truncated),
    }


def process_state(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    retrievers: dict[str, Any],
    tokenizer: Any,
    profile_name: str,
    cache_dir: Path,
) -> dict[str, Any]:
    state = result["state"]
    state_id = str(state["state_id"])
    cache_path = cache_dir / f"{state_id}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    profile = cfg["profiles"][profile_name]
    maximum = int(profile.get("maximum_candidates_per_state", cfg["labeling"]["maximum_candidates_per_state"]))
    candidates = _candidate_rows(result, maximum)
    minimum = int(cfg["labeling"]["minimum_candidates_per_state"])
    if len(candidates) < minimum:
        return {"state_id": state_id, "excluded": "too-few-candidates", "candidate_count": len(candidates)}
    backend = str(state["backend"])
    retriever = retrievers[backend]
    prefix = reconstruct_prefix(
        state, cfg=causal_cfg, retriever=retriever, tokenizer=tokenizer
    )
    seeds = [int(value) for value in profile.get("continuation_seeds", cfg["labeling"]["continuation_seeds"])]
    weights = {key: float(value) for key, value in cfg["labeling"]["reward_weights"].items()}
    maximum_documents = int(cfg["labeling"]["maximum_documents_per_query"])
    candidate_payloads: list[dict[str, Any]] = []
    document_payloads: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        query = candidate["query"]
        results = retriever.search(query, int(state["topk"]))
        documents = [_document_parts(item, rank) for rank, item in enumerate(results, start=1)]
        scores = np.asarray([row["retriever_score"] for row in documents], dtype=np.float64)
        if len(scores):
            std = float(scores.std())
            score_z = np.zeros_like(scores) if std <= 1e-12 else (scores - scores.mean()) / std
            for row, z in zip(documents, score_z, strict=True):
                row["retriever_score_z"] = float(z)
        full_replays = []
        omission_replays: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(min(maximum_documents, len(results)))
        }
        for continuation_seed in seeds:
            full = replay_after_observation(
                state,
                causal_cfg=causal_cfg,
                retriever=retriever,
                tokenizer=tokenizer,
                prefix=prefix,
                query=query,
                results=results,
                continuation_seed=continuation_seed,
            )
            full["reward"] = _reward_from_replay(full, weights)
            full["continuation_seed"] = continuation_seed
            full_replays.append(full)
            for document_index in omission_replays:
                omission_results = results[:document_index] + results[document_index + 1 :]
                omission = replay_after_observation(
                    state,
                    causal_cfg=causal_cfg,
                    retriever=retriever,
                    tokenizer=tokenizer,
                    prefix=prefix,
                    query=query,
                    results=omission_results,
                    continuation_seed=continuation_seed,
                )
                omission["reward"] = _reward_from_replay(omission, weights)
                omission["continuation_seed"] = continuation_seed
                omission_replays[document_index].append(omission)
        mean_reward = float(np.mean([row["reward"] for row in full_replays]))
        mean_support = float(np.mean([row["final_support_recall"] for row in full_replays]))
        mean_f1 = float(np.mean([row["answer_f1"] for row in full_replays]))
        mean_searches = float(np.mean([row["search_count"] for row in full_replays]))
        mean_invalid = float(np.mean([row["invalid_action_count"] for row in full_replays]))
        document_utilities: list[float] = []
        for document_index, omission_rows in omission_replays.items():
            utility = float(
                np.mean(
                    [
                        full_replays[seed_index]["reward"] - omission_rows[seed_index]["reward"]
                        for seed_index in range(len(seeds))
                    ]
                )
            )
            document_utilities.append(utility)
            metadata = documents[document_index]
            document_payloads.append(
                {
                    "state_id": state_id,
                    "question_id": str(state["question_id"]),
                    "backend": backend,
                    "dataset": str(state["dataset"]),
                    "source_turn": int(state["source_turn"]),
                    "candidate_index": candidate_index,
                    "query": query,
                    "style": candidate["style"],
                    **metadata,
                    "document_utility": utility,
                    "full_reward": mean_reward,
                }
            )
        aggregations = {
            mode: aggregate_document_credit(document_utilities, mode)
            for mode in cfg["labeling"]["document_credit_aggregations"]
        }
        observed_titles = full_replays[0]["observed_titles"]
        candidate_payloads.append(
            {
                "state_id": state_id,
                "question_id": str(state["question_id"]),
                "backend": backend,
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "candidate_index": candidate_index,
                "query": query,
                "style": candidate["style"],
                "origin": candidate["origin"],
                "full_reward": mean_reward,
                "final_support_recall": mean_support,
                "answer_f1": mean_f1,
                "search_count": mean_searches,
                "invalid_action_count": mean_invalid,
                "retrieved_titles": full_replays[0]["retrieved_titles"],
                "observed_titles": observed_titles,
                "behavior_signature": behavior_signature(observed_titles),
                "document_utilities": document_utilities,
                "document_credit": aggregations,
            }
        )

    rewards = [float(row["full_reward"]) for row in candidate_payloads]
    gaps = best_replacement_gap(rewards)
    centered = centered_action_advantage(rewards)
    signatures: dict[str, int] = {}
    for row in candidate_payloads:
        signature = str(row["behavior_signature"])
        signatures[signature] = signatures.get(signature, 0) + 1
    epsilon = float(cfg["labeling"]["replacement_epsilon"])
    for row, gap, advantage in zip(candidate_payloads, gaps, centered, strict=True):
        row["query_indispensability"] = float(gap)
        row["query_action_advantage"] = float(advantage)
        row["replaceable"] = int(gap <= epsilon)
        row["alias_class_size"] = int(signatures[str(row["behavior_signature"])])
        row["question"] = str(state["question"])
        row["prior_turns"] = state.get("prior_turns", [])
    payload = {
        "state_id": state_id,
        "question_id": str(state["question_id"]),
        "backend": backend,
        "dataset": str(state["dataset"]),
        "prefix_messages": prefix["messages"],
        "candidates": candidate_payloads,
        "documents": document_payloads,
        "signature": stable_hash(state_id, len(candidate_payloads), len(document_payloads)),
    }
    atomic_write_json(cache_path, payload)
    return payload


def run(cfg: dict[str, Any], causal_cfg: dict[str, Any], profile_name: str, inputs: Sequence[str] | None) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    paths = discover_inputs(cfg, inputs)
    states = load_state_results(paths)
    states = sorted(states, key=lambda row: str(row["state"]["state_id"]))[: int(profile["states"])]
    _service_check(causal_cfg)
    retrievers = _retrievers(causal_cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        causal_cfg["model"]["base_model"],
        revision=causal_cfg["model"].get("revision"),
        trust_remote_code=bool(causal_cfg["model"].get("trust_remote_code", False)),
    )
    root = Path(cfg["work_dir"]).resolve() / "labels" / profile_name
    cache_dir = root / "states"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(profile["workers"])) as executor:
        futures = {
            executor.submit(
                process_state,
                result,
                cfg=cfg,
                causal_cfg=causal_cfg,
                retrievers=retrievers,
                tokenizer=tokenizer,
                profile_name=profile_name,
                cache_dir=cache_dir,
            ): result
            for result in states
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    included = [row for row in outputs if "excluded" not in row]
    excluded = [row for row in outputs if "excluded" in row]
    candidates = [candidate for state in included for candidate in state["candidates"]]
    documents = [document for state in included for document in state["documents"]]
    prefixes = [
        {
            "state_id": state["state_id"],
            "question_id": state["question_id"],
            "backend": state["backend"],
            "dataset": state["dataset"],
            "prefix_messages": state["prefix_messages"],
        }
        for state in included
    ]
    atomic_write_jsonl(root / "candidate_credits.jsonl", sorted(candidates, key=lambda row: (row["state_id"], row["candidate_index"])))
    atomic_write_jsonl(root / "document_credits.jsonl", sorted(documents, key=lambda row: (row["state_id"], row["candidate_index"], row["document_rank"])))
    atomic_write_jsonl(root / "state_prefixes.jsonl", sorted(prefixes, key=lambda row: row["state_id"]))
    atomic_write_jsonl(root / "exclusions.jsonl", sorted(excluded, key=lambda row: row["state_id"]))
    manifest = {
        "schema": 1,
        "experiment_id": "EXP-050",
        "profile": profile_name,
        "source_files": len(paths),
        "requested_states": len(states),
        "included_states": len(included),
        "excluded_states": len(excluded),
        "candidate_rows": len(candidates),
        "document_rows": len(documents),
        "service_identity": _service_check(causal_cfg),
    }
    atomic_write_json(root / "manifest.json", manifest)
    report_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile_name / "EXP-050"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# EXP-050 Matched query/document counterfactual labels",
        "",
        f"Profile: `{profile_name}`.",
        "",
        f"- Included states: **{len(included)}**",
        f"- Excluded states: **{len(excluded)}**",
        f"- Query candidates: **{len(candidates)}**",
        f"- Document omission rows: **{len(documents)}**",
        f"- Continuation seeds: **{len(profile.get('continuation_seeds', cfg['labeling']['continuation_seeds']))}**",
        "",
        "Every query candidate is executed from the same frozen state. Query indispensability is computed against the best alternative, while document utility is measured by matched document omission under common continuation seeds.",
        "",
    ]
    (report_dir / "EXP050_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    atomic_write_json(report_dir / "decision.json", {**manifest, "go": len(included) > 0})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit.yaml")
    parser.add_argument("--causal-config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(
        load_config(args.config),
        load_causal_query_config(args.causal_config),
        args.profile,
        args.input,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

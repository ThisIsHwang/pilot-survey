from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.causal_query_replay import reconstruct_prefix
from stackpilot.query_credit_common import (
    atomic_write_json,
    behavior_signature,
    stable_hash,
)
from stackpilot.query_credit_labels import replay_after_observation
from stackpilot.query_credit_weekend_collect_support import (
    SCHEMA,
    _candidate_bank,
    _compact_replay,
    _corpus_probe,
    _reward_views,
    _serializable_documents,
)
from stackpilot.query_credit_weekend_common import (
    aggregate_swap_credit,
    apply_fixed_cardinality_swap,
    choose_length_matched_replacements,
)


def process_state(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    profile_name: str,
    retrievers: dict[str, Any],
    tokenizer: Any,
    cache_root: Path,
    omission_state_ids: set[str],
    run_signature: str,
) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    state = result["state"]
    state_id = str(state["state_id"])
    backend = str(state["backend"]).lower()
    dataset = str(state["dataset"]).lower()
    cache_path = cache_root / backend / f"{state_id}.json"
    signature = stable_hash(
        "query-credit-weekend-state-v2",
        run_signature,
        json.dumps(state, sort_keys=True),
        length=32,
    )
    if cache_path.is_file():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            return existing

    retriever = retrievers[backend]
    prefix = reconstruct_prefix(
        state,
        cfg=causal_cfg,
        retriever=retriever,
        tokenizer=tokenizer,
    )
    candidates = _candidate_bank(
        result,
        state=state,
        prefix=prefix,
        causal_cfg=causal_cfg,
        profile=profile,
    )
    seeds = [int(value) for value in profile["continuation_seeds"]]
    visible_documents = int(profile["visible_documents"])
    retrieval_depth = int(profile["retrieval_depth"])
    primary_view = str(cfg["analysis"]["primary_reward_view"])
    raw_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        query = str(candidate["query"])
        results = retriever.search(query, retrieval_depth)
        if len(results) < retrieval_depth:
            raise RuntimeError(
                f"State {state_id} query {candidate_index} returned only {len(results)} documents"
            )
        replacements = choose_length_matched_replacements(
            results,
            visible_documents=visible_documents,
            pool_start_rank=int(profile["replacement_pool_start_rank"]),
            pool_end_rank=int(profile["replacement_pool_end_rank"]),
            tokenizer=tokenizer,
        )
        visible_results = [dict(value) for value in results[:visible_documents]]
        full_seed_rewards: dict[str, list[float]] = {}
        swap_seed_credit: dict[str, list[list[float]]] = {}
        omission_seed_credit: dict[str, list[list[float]]] = {}
        for continuation_seed in seeds:
            full = replay_after_observation(
                state,
                causal_cfg=causal_cfg,
                retriever=retriever,
                tokenizer=tokenizer,
                prefix=prefix,
                query=query,
                results=visible_results,
                continuation_seed=continuation_seed,
            )
            full["continuation_seed"] = continuation_seed
            full_rewards = _reward_views(full, cfg)
            compact_full = _compact_replay(full, full_rewards)
            for name, value in full_rewards.items():
                full_seed_rewards.setdefault(name, []).append(float(value))
            raw_rows.append(
                {
                    "state_id": state_id,
                    "question_id": str(state["question_id"]),
                    "dataset": dataset,
                    "backend": backend,
                    "candidate_index": candidate_index,
                    "query": query,
                    "intervention": "full",
                    "document_slot": None,
                    "replacement_rank": None,
                    **compact_full,
                }
            )

            per_view_swap: dict[str, list[float]] = {name: [] for name in full_rewards}
            for replacement in replacements:
                swapped_results = apply_fixed_cardinality_swap(
                    results,
                    visible_documents=visible_documents,
                    slot=int(replacement["slot"]),
                    replacement_index=int(replacement["replacement_index"]),
                )
                swapped = replay_after_observation(
                    state,
                    causal_cfg=causal_cfg,
                    retriever=retriever,
                    tokenizer=tokenizer,
                    prefix=prefix,
                    query=query,
                    results=swapped_results,
                    continuation_seed=continuation_seed,
                )
                swapped["continuation_seed"] = continuation_seed
                swapped_rewards = _reward_views(swapped, cfg)
                compact_swapped = _compact_replay(swapped, swapped_rewards)
                for name in full_rewards:
                    per_view_swap[name].append(
                        float(full_rewards[name]) - float(swapped_rewards[name])
                    )
                raw_rows.append(
                    {
                        "state_id": state_id,
                        "question_id": str(state["question_id"]),
                        "dataset": dataset,
                        "backend": backend,
                        "candidate_index": candidate_index,
                        "query": query,
                        "intervention": "fixed-cardinality-swap",
                        "document_slot": int(replacement["slot"]),
                        "replacement_rank": int(replacement["replacement_rank"]),
                        "original_title": str(replacement["original_title"]),
                        "replacement_title": str(replacement["replacement_title"]),
                        **compact_swapped,
                    }
                )
            for name, values in per_view_swap.items():
                swap_seed_credit.setdefault(name, []).append(values)

            if state_id in omission_state_ids:
                per_view_omission: dict[str, list[float]] = {
                    name: [] for name in full_rewards
                }
                for slot in range(visible_documents):
                    omitted_results = visible_results[:slot] + visible_results[slot + 1 :]
                    omitted = replay_after_observation(
                        state,
                        causal_cfg=causal_cfg,
                        retriever=retriever,
                        tokenizer=tokenizer,
                        prefix=prefix,
                        query=query,
                        results=omitted_results,
                        continuation_seed=continuation_seed,
                    )
                    omitted["continuation_seed"] = continuation_seed
                    omitted_rewards = _reward_views(omitted, cfg)
                    compact_omitted = _compact_replay(omitted, omitted_rewards)
                    for name in full_rewards:
                        per_view_omission[name].append(
                            float(full_rewards[name]) - float(omitted_rewards[name])
                        )
                    raw_rows.append(
                        {
                            "state_id": state_id,
                            "question_id": str(state["question_id"]),
                            "dataset": dataset,
                            "backend": backend,
                            "candidate_index": candidate_index,
                            "query": query,
                            "intervention": "omission-sensitivity",
                            "document_slot": slot,
                            "replacement_rank": None,
                            **compact_omitted,
                        }
                    )
                for name, values in per_view_omission.items():
                    omission_seed_credit.setdefault(name, []).append(values)

        swap_credit = {
            name: aggregate_swap_credit(values)
            for name, values in swap_seed_credit.items()
        }
        omission_credit = {
            name: aggregate_swap_credit(values)
            for name, values in omission_seed_credit.items()
        }
        candidate_rows.append(
            {
                "schema": SCHEMA,
                "state_id": state_id,
                "question_id": str(state["question_id"]),
                "dataset": dataset,
                "backend": backend,
                "source_turn": int(state["source_turn"]),
                "candidate_index": candidate_index,
                "query": query,
                "question": str(state["question"]),
                "answers": [
                    str(value)
                    for value in (state.get("answers") or [state.get("answer", "")])
                    if str(value).strip()
                ],
                "origin": str(candidate["origin"]),
                "source_seed": candidate.get("source_seed"),
                "visible_documents": _serializable_documents(visible_results),
                "full_seed_rewards": full_seed_rewards,
                "mean_reward": {
                    name: float(np.mean(values)) for name, values in full_seed_rewards.items()
                },
                "swap_seed_credit": swap_seed_credit,
                "swap_credit": swap_credit,
                "omission_seed_credit": omission_seed_credit,
                "omission_credit": omission_credit,
                "replacement_plan": replacements,
                "retrieved_titles": [
                    str(value.get("document", {}).get("contents", "")).split("\n", 1)[0]
                    if isinstance(value.get("document"), dict)
                    else str(value.get("title", ""))
                    for value in results[:visible_documents]
                ],
                "behavior_signature": behavior_signature(
                    [
                        str(value.get("document", {}).get("contents", "")).split("\n", 1)[0]
                        if isinstance(value.get("document"), dict)
                        else str(value.get("title", ""))
                        for value in results[:visible_documents]
                    ]
                ),
            }
        )

    primary_rewards = [float(row["mean_reward"][primary_view]) for row in candidate_rows]
    primary_mean = float(np.mean(primary_rewards))
    for row, reward in zip(candidate_rows, primary_rewards, strict=True):
        row["query_action_advantage"] = float(reward - primary_mean)
        row["full_reward"] = float(reward)
        # Compatibility with the prior query-credit tooling.
        row["document_credit"] = {
            "signed-mean": float(row["swap_credit"][primary_view]["signed_mean"]),
            "positive-sum": float(row["swap_credit"][primary_view]["positive_sum"]),
            "signed-sum": float(row["swap_credit"][primary_view]["signed_sum"]),
        }

    payload = {
        "schema": SCHEMA,
        "signature": signature,
        "run_signature": run_signature,
        "state_id": state_id,
        "question_id": str(state["question_id"]),
        "dataset": dataset,
        "backend": backend,
        "model": str(cfg["model"]["base_model"]),
        "prefix_messages": prefix["messages"],
        "candidate_count": len(candidate_rows),
        "direct_policy_candidate_fraction": float(
            np.mean(
                [
                    row["origin"] in {"factual", "direct-policy-sibling"}
                    for row in candidate_rows
                ]
            )
        ),
        # BM25 exact-title retrieval is used only as a shared-corpus health
        # probe. It never affects inclusion, query credit, or document choice.
        "corpus_probe": _corpus_probe(state, retrievers["bm25"]),
        "candidates": candidate_rows,
        "raw_replays": raw_rows,
        "created_at_unix": time.time(),
    }
    atomic_write_json(cache_path, payload)
    return payload

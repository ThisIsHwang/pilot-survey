from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from stackpilot.behavior_quotient_common import balanced_subset, load_state_results
from stackpilot.causal_query_common import load_causal_query_config
from stackpilot.causal_query_replay import _retrievers, _service_check, reconstruct_prefix
from stackpilot.credit_routing_common import (
    FEATURE_NAMES,
    atomic_write_json,
    atomic_write_jsonl,
    env_patterns,
    feature_rows,
    fixed_budget_contexts,
    load_config,
    markdown_table,
    matched_swap_utilities,
    normalize_document,
    safe_spearman,
    stable_hash,
)
from stackpilot.interface_document_ctu import factual_candidate, replay_after_observation

EXPERIMENT_ID = "EXP-045"


def _context_reward(replay: dict[str, Any], weights: dict[str, Any]) -> float:
    return (
        float(weights["support"]) * float(replay["final_support_recall"])
        + float(weights["answer_f1"]) * float(replay["answer_f1"])
        - float(weights["search_efficiency"]) * float(replay["search_count"])
        - float(weights["invalid_action"]) * float(replay["invalid_action_count"])
    )


def process_state(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    retrievers: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state = result["state"]
    backend = str(state["backend"])
    retriever = retrievers[backend]
    prefix = reconstruct_prefix(
        state,
        cfg=causal_cfg,
        retriever=retriever,
        tokenizer=tokenizer,
    )
    factual = factual_candidate(result)
    query = str(factual["query"])
    upstream_topk = int(cfg["labeling"]["upstream_topk"])
    keep_k = int(cfg["labeling"]["observation_k"])
    retrieval = list(retriever.search(query, upstream_topk))
    if len(retrieval) < upstream_topk:
        raise RuntimeError(
            f"State {state['state_id']} returned {len(retrieval)} documents, "
            f"expected {upstream_topk}"
        )
    if upstream_topk <= keep_k:
        raise RuntimeError("upstream_topk must exceed observation_k")

    contexts = fixed_budget_contexts(len(retrieval), keep_k)
    replays: dict[tuple[int, ...], dict[str, Any]] = {}
    weights = cfg["labeling"]["weights"]
    context_rows: list[dict[str, Any]] = []
    for indices in contexts:
        replay = replay_after_observation(
            state,
            causal_cfg=causal_cfg,
            retriever=retriever,
            tokenizer=tokenizer,
            prefix=prefix,
            query=query,
            results=[retrieval[index] for index in indices],
            branch_name="fixed-budget-" + "-".join(map(str, indices)),
        )
        replays[indices] = replay
        context_rows.append(
            {
                "schema": 1,
                "experiment_id": EXPERIMENT_ID,
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": backend,
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "query": query,
                "context_indices": list(indices),
                "context_titles": [normalize_document(retrieval[index])[0] for index in indices],
                "final_support_recall": float(replay["final_support_recall"]),
                "answer_f1": float(replay["answer_f1"]),
                "search_count": int(replay["search_count"]),
                "invalid_action_count": int(replay["invalid_action_count"]),
                "context_reward": _context_reward(replay, weights),
            }
        )

    support_values = {
        context: float(replay["final_support_recall"])
        for context, replay in replays.items()
    }
    answer_values = {
        context: float(replay["answer_f1"])
        for context, replay in replays.items()
    }
    # Costs are negated so every matched-swap component is higher-is-better.
    search_values = {
        context: -float(replay["search_count"])
        for context, replay in replays.items()
    }
    invalid_values = {
        context: -float(replay["invalid_action_count"])
        for context, replay in replays.items()
    }
    composite_values = {
        context: _context_reward(replay, weights)
        for context, replay in replays.items()
    }
    support_utilities, comparison_counts = matched_swap_utilities(
        support_values,
        candidate_count=len(retrieval),
        keep_k=keep_k,
    )
    answer_utilities, _ = matched_swap_utilities(
        answer_values,
        candidate_count=len(retrieval),
        keep_k=keep_k,
    )
    search_utilities, _ = matched_swap_utilities(
        search_values,
        candidate_count=len(retrieval),
        keep_k=keep_k,
    )
    invalid_utilities, _ = matched_swap_utilities(
        invalid_values,
        candidate_count=len(retrieval),
        keep_k=keep_k,
    )
    document_utilities, _ = matched_swap_utilities(
        composite_values,
        candidate_count=len(retrieval),
        keep_k=keep_k,
    )

    rows: list[dict[str, Any]] = []
    features = feature_rows(query, retrieval, backend)
    for index, item in enumerate(retrieval):
        title, text, score = normalize_document(item)
        contexts_with = [context for context in contexts if index in context]
        contexts_without = [context for context in contexts if index not in context]
        rows.append(
            {
                "schema": 2,
                "experiment_id": EXPERIMENT_ID,
                "utility_design": "exact-matched-swaps",
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "question": str(state["question"]),
                "dataset": str(state["dataset"]),
                "backend": backend,
                "source_turn": int(state["source_turn"]),
                "query": query,
                "candidate_count": len(retrieval),
                "observation_k": keep_k,
                "document_rank": index + 1,
                "document_title": title,
                "document_text": text[: int(cfg["labeling"]["maximum_document_chars"])],
                "retriever_score": score,
                "support_utility": float(support_utilities[index]),
                "answer_utility": float(answer_utilities[index]),
                "search_utility": float(search_utilities[index]),
                "invalid_utility": float(invalid_utilities[index]),
                "document_utility": float(document_utilities[index]),
                "matched_swap_comparisons": int(comparison_counts[index]),
                "contexts_with_document": len(contexts_with),
                "contexts_without_document": len(contexts_without),
                "mean_context_reward_with_document": float(
                    np.mean([composite_values[context] for context in contexts_with])
                ),
                "mean_context_reward_without_document": float(
                    np.mean([composite_values[context] for context in contexts_without])
                ),
                "factual_query_support_tqe": float(factual.get("support_tqe", 0.0)),
                "factual_query_composite_tqe": float(factual.get("composite_tqe", 0.0)),
                **features[index],
            }
        )

    state_summary = {
        "state_id": str(state["state_id"]),
        "question_id": str(state["question_id"]),
        "backend": backend,
        "dataset": str(state["dataset"]),
        "source_turn": int(state["source_turn"]),
        "query": query,
        "documents": len(rows),
        "unique_context_replays": len(replays),
        "matched_swap_comparisons_per_document": int(comparison_counts[0]),
        "document_utility_sum": float(document_utilities.sum()),
        "label_signature": stable_hash(
            state["state_id"],
            query,
            [(row["document_title"], row["document_utility"]) for row in rows],
            length=32,
        ),
    }
    return state_summary, rows, context_rows


def run(
    cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    profile_name: str,
    *,
    inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    patterns = env_patterns(
        "CREDIT_ROUTING_INPUTS",
        cfg["source"]["state_globs"],
        inputs,
    )
    results = balanced_subset(
        load_state_results(patterns),
        int(profile["label_states"]),
    )
    service_identity = _service_check(causal_cfg)
    retrievers = _retrievers(causal_cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        causal_cfg["model"]["base_model"],
        trust_remote_code=bool(causal_cfg["model"].get("trust_remote_code", False)),
    )
    output_dir = Path(cfg["work_dir"]).resolve() / "labels" / profile_name
    cache_dir = output_dir / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_signature = stable_hash(
        "credit-routing-label-v2",
        cfg["labeling"],
        causal_cfg["model"].get("base_model"),
        length=32,
    )

    cached_by_state: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for result in results:
        state_id = str(result["state"]["state_id"])
        cache_path = cache_dir / f"{state_id}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_signature") != cache_signature:
                raise RuntimeError(
                    f"Stale fixed-budget label cache: {cache_path}. "
                    "Remove the cache or use a new work directory."
                )
            cached_by_state[state_id] = payload
        else:
            pending.append(result)

    with ThreadPoolExecutor(max_workers=int(profile["label_workers"])) as executor:
        futures = {
            executor.submit(
                process_state,
                result,
                cfg=cfg,
                causal_cfg=causal_cfg,
                retrievers=retrievers,
                tokenizer=tokenizer,
            ): str(result["state"]["state_id"])
            for result in pending
        }
        for future in as_completed(futures):
            state_id = futures[future]
            summary, state_rows, state_contexts = future.result()
            payload = {
                "schema": 1,
                "cache_signature": cache_signature,
                "summary": summary,
                "document_rows": state_rows,
                "context_rows": state_contexts,
            }
            atomic_write_json(cache_dir / f"{state_id}.json", payload)
            cached_by_state[state_id] = payload

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for result in results:
        state_id = str(result["state"]["state_id"])
        payload = cached_by_state.get(state_id)
        if payload is None:
            raise RuntimeError(f"Missing completed label cache for {state_id}")
        summaries.append(dict(payload["summary"]))
        rows.extend(dict(row) for row in payload["document_rows"])
        context_rows.extend(dict(row) for row in payload["context_rows"])
    summaries.sort(key=lambda row: row["state_id"])
    rows.sort(key=lambda row: (row["state_id"], int(row["document_rank"])))
    context_rows.sort(
        key=lambda row: (row["state_id"], tuple(row["context_indices"]))
    )
    labels_path = output_dir / "budgeted_document_utility.jsonl"
    contexts_path = output_dir / "fixed_budget_contexts.jsonl"
    atomic_write_jsonl(labels_path, rows)
    atomic_write_jsonl(contexts_path, context_rows)
    atomic_write_jsonl(output_dir / "state_summaries.jsonl", summaries)
    manifest = {
        "schema": 2,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "utility_design": "exact-matched-swaps",
        "states": len(summaries),
        "document_rows": len(rows),
        "context_rows": len(context_rows),
        "contexts_per_state": (
            int(summaries[0]["unique_context_replays"]) if summaries else 0
        ),
        "upstream_topk": int(cfg["labeling"]["upstream_topk"]),
        "observation_k": int(cfg["labeling"]["observation_k"]),
        "feature_names": list(FEATURE_NAMES),
        "labels": str(labels_path),
        "contexts": str(contexts_path),
        "service_identity": service_identity,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    report_dir = (
        Path(cfg["work_dir"]).resolve()
        / "reports"
        / profile_name
        / EXPERIMENT_ID
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    positive_rate = (
        float((frame["document_utility"] > 0.0).mean())
        if not frame.empty
        else 0.0
    )
    query_document_spearman = (
        safe_spearman(
            frame["document_utility"].to_numpy(dtype=float),
            frame["factual_query_composite_tqe"].to_numpy(dtype=float),
        )
        if len(frame) > 1
        else 0.0
    )
    subgroups = (
        frame.groupby(["backend", "dataset"], as_index=False).agg(
            states=("state_id", "nunique"),
            documents=("state_id", "size"),
            positive_rate=(
                "document_utility",
                lambda values: float((values > 0.0).mean()),
            ),
            mean_absolute_utility=(
                "document_utility",
                lambda values: float(np.abs(values).mean()),
            ),
        )
        if not frame.empty
        else pd.DataFrame()
    )
    subgroups.to_csv(report_dir / "subgroup_metrics.csv", index=False)
    state_ranges = (
        frame.groupby("state_id")["document_utility"].agg(
            lambda values: float(values.max() - values.min())
        )
        if not frame.empty
        else pd.Series(dtype=float)
    )
    gate = cfg["gates"][EXPERIMENT_ID]
    informative_state_rate = (
        float(
            (
                state_ranges
                >= float(gate["minimum_state_utility_range"])
            ).mean()
        )
        if len(state_ranges)
        else 0.0
    )
    go = bool(
        len(summaries) >= int(gate["minimum_states"])
        and positive_rate >= float(gate["minimum_positive_document_rate"])
        and informative_state_rate
        >= float(gate["minimum_informative_state_rate"])
    )
    decision = {
        **manifest,
        "positive_document_rate": positive_rate,
        "informative_state_rate": informative_state_rate,
        "minimum_state_utility_range": float(
            gate["minimum_state_utility_range"]
        ),
        "document_query_spearman": query_document_spearman,
        "go": go,
    }
    atomic_write_json(report_dir / "decision.json", decision)
    report = [
        "# EXP-045 Fixed-budget document utility labels",
        "",
        f"Profile: `{profile_name}`. Every top-{manifest['upstream_topk']} candidate is evaluated by exact matched swaps across all {manifest['contexts_per_state']} equal-cardinality top-{manifest['observation_k']} contexts.",
        "",
        f"States: **{len(summaries)}**. Document rows: **{len(rows)}**. Context replays: **{len(context_rows)}**. Positive document-utility rate: **{positive_rate:.4f}**. Informative-state rate: **{informative_state_rate:.4f}**.",
        "",
        f"Document utility versus factual-query composite TQE Spearman: **{query_document_spearman:.4f}**.",
        "",
        "## Backend × dataset",
        "",
        markdown_table(subgroups),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "This gate checks only that a comparable fixed-budget utility signal exists. The same frozen signal is subsequently routed either to the producing policy reward or to observation selection.",
        "",
    ]
    (report_dir / "EXP045_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact matched-swap document utility labels."
    )
    parser.add_argument("--config", default="configs/credit_routing.yaml")
    parser.add_argument("--causal-config", default="configs/causal_query_audit.yaml")
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(
        load_config(args.config),
        load_causal_query_config(args.causal_config),
        args.profile,
        inputs=args.input,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

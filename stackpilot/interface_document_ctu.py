from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from stackpilot.action_protocol import parse_action
from stackpilot.causal_query_common import (
    load_causal_query_config,
    normalize_title,
    stable_seed,
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
from stackpilot.interface_causality_common import (
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    load_state_results,
    source_patterns,
)
from stackpilot.observation_geometry import render_retrieval_observation


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def factual_candidate(result: dict[str, Any]) -> dict[str, Any]:
    candidate = next(
        (
            row
            for row in result["candidates"]
            if str(row.get("origin", "")) == "factual"
            or str(row.get("style", "")) == "factual"
        ),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"State {result['state']['state_id']} has no factual candidate")
    return candidate


def extract_title(row: dict[str, Any]) -> str:
    for key in ("title", "document_title", "wikipedia_title"):
        value = row.get(key)
        if value:
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("title", "document_title", "wikipedia_title"):
            value = metadata.get(key)
            if value:
                return str(value)
    document = row.get("document")
    if isinstance(document, dict) and document.get("title"):
        return str(document["title"])
    return ""


def replay_after_observation(
    state: dict[str, Any],
    *,
    causal_cfg: dict[str, Any],
    retriever: Any,
    tokenizer: Any,
    prefix: dict[str, Any],
    query: str,
    results: list[dict[str, Any]],
    branch_name: str,
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
            seed=stable_seed("doc-ctu", state["state_id"], attempts),
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
            suffix_results,
            tokenizer,
            token_budget,
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
        "intervention_observed_titles": list(rendered.observed_titles),
    }


def process_state(
    result: dict[str, Any],
    *,
    interface_cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    retrievers: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    results = retriever.search(query, int(state["topk"]))
    maximum_documents = min(
        len(results), int(interface_cfg["document_ctu"]["maximum_documents_per_state"])
    )
    baseline = replay_after_observation(
        state,
        causal_cfg=causal_cfg,
        retriever=retriever,
        tokenizer=tokenizer,
        prefix=prefix,
        query=query,
        results=results,
        branch_name="full",
    )
    expected_titles = {
        normalize_title(value)
        for value in _strings(factual.get("intervention_observed_titles"))
    }
    replayed_titles = {
        normalize_title(value)
        for value in baseline["intervention_observed_titles"]
    }
    union = expected_titles | replayed_titles
    title_jaccard = (len(expected_titles & replayed_titles) / len(union)) if union else 1.0
    minimum_jaccard = float(causal_cfg["validation"]["minimum_factual_title_jaccard"])
    if title_jaccard < minimum_jaccard:
        raise RuntimeError(
            f"State {state['state_id']} factual replay mismatch: "
            f"title_jaccard={title_jaccard:.4f} < {minimum_jaccard:.4f}"
        )
    rows = []
    weights = interface_cfg["document_ctu"]["weights"]
    for index in range(maximum_documents):
        omission_results = results[:index] + results[index + 1 :]
        omission = replay_after_observation(
            state,
            causal_cfg=causal_cfg,
            retriever=retriever,
            tokenizer=tokenizer,
            prefix=prefix,
            query=query,
            results=omission_results,
            branch_name=f"omit-{index}",
        )
        support_ctu = baseline["final_support_recall"] - omission["final_support_recall"]
        answer_ctu = baseline["answer_f1"] - omission["answer_f1"]
        search_ctu = omission["search_count"] - baseline["search_count"]
        document_ctu = (
            float(weights["support"]) * support_ctu
            + float(weights["answer_f1"]) * answer_ctu
            + float(weights["search_efficiency"]) * search_ctu
        )
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": backend,
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "query": query,
                "document_rank": index + 1,
                "document_title": extract_title(results[index]),
                "support_ctu": support_ctu,
                "answer_ctu": answer_ctu,
                "search_ctu": search_ctu,
                "document_ctu": document_ctu,
                "baseline_final_support_recall": baseline["final_support_recall"],
                "omission_final_support_recall": omission["final_support_recall"],
                "factual_query_tqe": float(factual.get("support_tqe", 0.0)),
                "factual_query_composite_tqe": float(factual.get("composite_tqe", 0.0)),
            }
        )
    summary = {
        "state_id": str(state["state_id"]),
        "question_id": str(state["question_id"]),
        "backend": backend,
        "dataset": str(state["dataset"]),
        "documents_omitted": maximum_documents,
        "baseline": baseline,
        "factual_title_jaccard": title_jaccard,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate document-level CTU labels for EXP-021.")
    parser.add_argument("--interface-config", default="configs/interface_causality.yaml")
    parser.add_argument("--causal-config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    interface_cfg = load_config(args.interface_config)
    causal_cfg = load_causal_query_config(args.causal_config)
    profile = interface_cfg["profiles"][args.profile]
    _service_check(causal_cfg)
    retrievers = _retrievers(causal_cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        causal_cfg["model"]["base_model"],
        trust_remote_code=bool(causal_cfg["model"].get("trust_remote_code", False)),
    )
    results = load_state_results(source_patterns(interface_cfg, args.inputs))
    results = sorted(results, key=lambda row: str(row["state"]["state_id"]))[: int(profile["document_ctu_states"])]
    output_dir = Path(
        args.output_dir
        or Path(interface_cfg["work_dir"]) / "document_ctu" / args.profile
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    rows = []
    with ThreadPoolExecutor(max_workers=int(profile["document_ctu_workers"])) as executor:
        futures = {
            executor.submit(
                process_state,
                result,
                interface_cfg=interface_cfg,
                causal_cfg=causal_cfg,
                retrievers=retrievers,
                tokenizer=tokenizer,
            ): result
            for result in results
        }
        for future in as_completed(futures):
            summary, state_rows = future.result()
            summaries.append(summary)
            rows.extend(state_rows)
    atomic_write_jsonl(output_dir / "document_ctu.jsonl", rows)
    manifest = {
        "profile": args.profile,
        "states": len(summaries),
        "document_rows": len(rows),
        "output": str(output_dir / "document_ctu.jsonl"),
        "service_identity": _service_check(causal_cfg),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

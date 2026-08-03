from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.interface_causality_common import (
    atomic_write_json,
    balanced_state_subset,
    cluster_bootstrap,
    jaccard,
    load_config,
    load_state_results,
    markdown_table,
    normalize_title,
    source_patterns,
    token_set,
    word_tokens,
)

EXPERIMENT_ID = "EXP-022"


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def prefix_titles(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    prefix = result.get("prefix", {})
    for record in prefix.get("records", []) or []:
        if isinstance(record, dict):
            values.extend(_strings(record.get("observed_titles")))
    if not values:
        for record in result["state"].get("prior_turns", []) or []:
            if isinstance(record, dict):
                values.extend(_strings(record.get("observed_titles")))
    seen: set[str] = set()
    output = []
    for value in values:
        normalized = normalize_title(value)
        if normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def relation_tokens(question: str, titles: list[str], maximum: int) -> list[str]:
    title_tokens = token_set(" ".join(titles), content_only=True)
    output = []
    seen = set()
    for token in word_tokens(question, content_only=True):
        if token in title_tokens or token in seen:
            continue
        seen.add(token)
        output.append(token)
        if len(output) >= maximum:
            break
    return output


def menu_queries(result: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    state = result["state"]
    titles = prefix_titles(result)
    maximum_titles = int(cfg["interface_audit"]["maximum_prefix_titles"])
    maximum_menu = int(cfg["interface_audit"]["maximum_menu_queries"])
    relations = relation_tokens(
        str(state["question"]),
        titles,
        int(cfg["interface_audit"]["relation_tokens"]),
    )
    relation = " ".join(relations)
    candidates: list[dict[str, str]] = []
    for title in titles[:maximum_titles]:
        candidates.append({"style": "menu-title", "query": title})
        if relation:
            candidates.append({"style": "menu-title-relation", "query": f"{title} {relation}"})
    if relation:
        candidates.append({"style": "menu-question-relation", "query": relation})
    if not candidates:
        question_terms = " ".join(word_tokens(str(state["question"]), content_only=True)[:8])
        if question_terms:
            candidates.append({"style": "menu-question", "query": question_terms})
    output, seen = [], set()
    for candidate in candidates:
        query = " ".join(candidate["query"].split())
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        output.append({**candidate, "query": query, "origin": "finite-menu"})
        if len(output) >= maximum_menu:
            break
    return output


def free_queries(result: dict[str, Any]) -> list[dict[str, str]]:
    output, seen = [], set()
    for candidate in result["candidates"]:
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        query = " ".join(str(candidate.get("query", "")).split())
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        output.append(
            {
                "style": str(candidate.get("style", "free")),
                "query": query,
                "origin": "free-form",
            }
        )
    return output


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
    if isinstance(document, dict):
        value = document.get("title")
        if value:
            return str(value)
    return ""


def execute_query(
    client: Any,
    query: dict[str, str],
    *,
    topk: int,
    state: dict[str, Any],
    prefix_observed: set[str],
) -> dict[str, Any]:
    results = client.search(query["query"], topk)
    titles = [extract_title(row) for row in results]
    titles = [title for title in titles if title]
    normalized = [normalize_title(value) for value in titles]
    gold = {normalize_title(value) for value in _strings(state.get("support_titles"))}
    before = len(gold & prefix_observed) / max(1, len(gold))
    after = len(gold & (prefix_observed | set(normalized))) / max(1, len(gold))
    return {
        **query,
        "observed_titles": titles,
        "ranked_signature": tuple(normalized),
        "immediate_support_gain": after - before,
        "support_recall_after": after,
    }


def state_interface_rows(
    result: dict[str, Any],
    cfg: dict[str, Any],
    clients: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = result["state"]
    backend = str(state["backend"])
    free = free_queries(result)
    menu = menu_queries(result, cfg)
    all_queries = free + menu
    if not free or not menu:
        return [], {}
    prefix_observed = {normalize_title(value) for value in prefix_titles(result)}
    executed = [
        execute_query(
            clients[backend],
            query,
            topk=int(state["topk"]),
            state=state,
            prefix_observed=prefix_observed,
        )
        for query in all_queries
    ]
    interfaces = {
        "free-form": [row for row in executed if row["origin"] == "free-form"],
        "finite-menu": [row for row in executed if row["origin"] == "finite-menu"],
        "hybrid": executed,
    }
    rows = []
    for interface, members in interfaces.items():
        signatures = {tuple(row["ranked_signature"]) for row in members}
        best = max(members, key=lambda row: (row["immediate_support_gain"], row["support_recall_after"]))
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": backend,
                "dataset": str(state["dataset"]),
                "source_turn": int(state["source_turn"]),
                "interface": interface,
                "query_count": len(members),
                "unique_behaviors": len(signatures),
                "alias_rate": 1.0 - len(signatures) / max(1, len(members)),
                "oracle_immediate_gain": float(best["immediate_support_gain"]),
                "oracle_support_recall": float(best["support_recall_after"]),
                "any_gain": int(float(best["immediate_support_gain"]) > 0.0),
                "best_query": str(best["query"]),
                "best_style": str(best["style"]),
                "best_signature": list(best["ranked_signature"]),
            }
        )
    free_best = max(interfaces["free-form"], key=lambda row: (row["immediate_support_gain"], row["support_recall_after"]))
    menu_signatures = {tuple(row["ranked_signature"]) for row in interfaces["finite-menu"]}
    gold_tokens = token_set(" ".join(_strings(state.get("support_titles"))), content_only=True)
    known_tokens = token_set(str(state["question"]) + " " + " ".join(prefix_titles(result)), content_only=True)
    free_best_tokens = token_set(str(free_best["query"]), content_only=True)
    state_summary = {
        "state_id": str(state["state_id"]),
        "question_id": str(state["question_id"]),
        "backend": backend,
        "dataset": str(state["dataset"]),
        "source_turn": int(state["source_turn"]),
        "menu_covers_free_best_behavior": int(tuple(free_best["ranked_signature"]) in menu_signatures),
        "free_best_gain": float(free_best["immediate_support_gain"]),
        "novel_query_token_rate": (
            len(free_best_tokens - known_tokens) / max(1, len(free_best_tokens))
        ),
        "prefix_contains_gold_title": int(
            bool({normalize_title(value) for value in prefix_titles(result)} & {normalize_title(value) for value in _strings(state.get("support_titles"))})
        ),
        "gold_token_coverage_in_known_state": len(gold_tokens & known_tokens) / max(1, len(gold_tokens)),
    }
    return rows, state_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-022: compare free-form, finite-menu, and hybrid search interfaces.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from stackpilot.retrieval_clients import RetrievalClient

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    clients = {
        "bm25": RetrievalClient(
            "bm25",
            str(cfg["retrieval"]["bm25_url"]),
            timeout=int(cfg["retrieval"]["timeout"]),
            retries=int(cfg["retrieval"]["retries"]),
        ),
        "e5": RetrievalClient(
            "e5",
            str(cfg["retrieval"]["e5_url"]),
            timeout=int(cfg["retrieval"]["timeout"]),
            retries=int(cfg["retrieval"]["retries"]),
        ),
    }
    results = load_state_results(source_patterns(cfg, args.inputs))
    results = balanced_state_subset(results, int(profile["interface_states"]))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    workers = int(profile["retrieval_workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(state_interface_rows, result, cfg, clients): result
            for result in results
        }
        for future in as_completed(future_map):
            interface_rows, state_summary = future.result()
            rows.extend(interface_rows)
            if state_summary:
                summaries.append(state_summary)
    if not rows:
        raise RuntimeError("No states had both free-form and finite-menu queries")
    frame = pd.DataFrame(rows)
    state_frame = pd.DataFrame(summaries)
    output_dir = Path(args.output_dir or Path(cfg["work_dir"]) / "reports" / args.profile / "EXP-022").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "interface_state_metrics.csv", index=False)
    state_frame.to_csv(output_dir / "state_characteristics.csv", index=False)

    pivot = frame.pivot_table(
        index=["state_id", "backend", "dataset", "source_turn"],
        columns="interface",
        values=["oracle_immediate_gain", "oracle_support_recall", "alias_rate", "unique_behaviors"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{interface}" for metric, interface in pivot.columns]
    pivot = pivot.reset_index().merge(state_frame, on=["state_id", "backend", "dataset", "source_turn"], how="left")
    pivot["free_minus_menu_gain"] = pivot["oracle_immediate_gain__free-form"] - pivot["oracle_immediate_gain__finite-menu"]
    pivot["hybrid_minus_free_gain"] = pivot["oracle_immediate_gain__hybrid"] - pivot["oracle_immediate_gain__free-form"]
    pivot.to_csv(output_dir / "paired_interface_contrasts.csv", index=False)

    free_advantage = cluster_bootstrap(
        pivot.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([row["free_minus_menu_gain"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=22101,
    )
    menu_miss = cluster_bootstrap(
        state_frame.to_dict("records"),
        cluster_key="state_id",
        statistic=lambda records: float(np.mean([1 - row["menu_covers_free_best_behavior"] for row in records])),
        samples=int(profile["bootstrap_samples"]),
        seed=22102,
    )
    ood = pivot[pivot["prefix_contains_gold_title"] == 0]
    ood_advantage = (
        cluster_bootstrap(
            ood.to_dict("records"),
            cluster_key="state_id",
            statistic=lambda records: float(np.mean([row["free_minus_menu_gain"] for row in records])),
            samples=int(profile["bootstrap_samples"]),
            seed=22103,
        )
        if len(ood)
        else {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0.0, "n_rows": 0.0, "finite_draws": 0.0}
    )
    gates = cfg["gates"]["EXP-022"]
    go = bool(
        menu_miss["estimate"] >= float(gates["minimum_menu_behavior_miss_rate"])
        and menu_miss["ci_low"] > 0.0
        and ood_advantage["estimate"] >= float(gates["minimum_freeform_ood_gain_advantage"])
        and ood_advantage["ci_low"] > 0.0
    )
    summary = (
        frame.groupby(["backend", "interface"], as_index=False)[
            ["oracle_immediate_gain", "oracle_support_recall", "alias_rate", "unique_behaviors", "any_gain"]
        ].mean()
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "profile": args.profile,
        "go": go,
        "free_form_minus_finite_menu_gain": free_advantage,
        "finite_menu_behavior_miss_rate": menu_miss,
        "free_form_ood_gain_advantage": ood_advantage,
        "states": int(pivot["state_id"].nunique()),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = [
        "# EXP-022 Interface-expressivity audit",
        "",
        f"Profile: `{args.profile}`. Free-form query candidates, deterministic finite-menu actions, and their hybrid are executed against the same backend and prefix state.",
        "",
        "## Interface summary",
        "",
        markdown_table(summary),
        "",
        "## Primary effects",
        "",
        "```text",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
        "A GO supports a top-conference comparison between free-form quotient policies and finite-menu harnesses: the finite interface reduces aliases but leaves a measurable oracle-utility or behavior-coverage gap, especially when the needed entity is not already present in the prefix.",
        "",
    ]
    (output_dir / "EXP022_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

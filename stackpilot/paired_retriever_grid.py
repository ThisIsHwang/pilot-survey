from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    balanced_subset,
    cluster_bootstrap,
    load_config,
    load_state_results,
    markdown_table,
    normalize_title,
    source_patterns,
    stable_hash,
)
from stackpilot.interface_expressivity_audit import prefix_titles
from stackpilot.retrieval_clients import RetrievalClient

EXPERIMENT_ID = "EXP-033"


def query_bank(result: dict[str, Any], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    candidates = sorted(
        result["candidates"],
        key=lambda row: (
            0 if str(row.get("style", "")) == "factual" else 1,
            str(row.get("style", "")),
            str(row.get("candidate_id", "")),
        ),
    )
    for candidate in candidates:
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        query = " ".join(str(candidate.get("query", "")).split())
        normalized = query.lower()
        if not query or normalized in seen:
            continue
        seen.add(normalized)
        output.append(query)
        if len(output) >= int(limit):
            break
    return output


def rrf_fuse(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    rrf_k: int,
    topk: int,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}
    for results in (left, right):
        for rank, row in enumerate(results, start=1):
            key = normalize_title(row.get("title", "")) or str(row.get("id", ""))
            if not key:
                continue
            scores[key] += 1.0 / (int(rrf_k) + rank)
            documents.setdefault(key, dict(row))
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[: int(topk)]
    return [
        {**documents[key], "score": scores[key], "rank": index + 1}
        for index, key in enumerate(ordered)
    ]


def execute_query(
    query: str,
    clients: dict[str, RetrievalClient],
    cfg: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    topk = int(cfg["retrieval"]["topk"])
    bm25 = clients["bm25"].search(query, topk)
    e5 = clients["e5"].search(query, topk)
    output = {
        "bm25": bm25,
        "e5": e5,
        "hybrid": rrf_fuse(
            bm25,
            e5,
            rrf_k=int(cfg["retrieval"]["rrf_k"]),
            topk=topk,
        ),
    }
    if "colbert" in clients:
        output["colbert"] = clients["colbert"].search(query, topk)
    return output


def process_state(
    result: dict[str, Any],
    *,
    clients: dict[str, RetrievalClient],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    state = result["state"]
    queries = query_bank(result, int(cfg["paired_grid"]["query_limit_per_state"]))
    if len(queries) < 2:
        return []
    prefix = {normalize_title(value) for value in prefix_titles(result)}
    gold = {normalize_title(value) for value in state.get("support_titles", [])}
    before = len(gold & prefix) / max(1, len(gold))
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        by_backend = execute_query(query, clients, cfg)
        for backend, results in by_backend.items():
            titles = [str(row.get("title", "")).strip() for row in results]
            titles = [title for title in titles if title]
            signature = tuple(normalize_title(title) for title in titles)
            after = len(gold & (prefix | set(signature))) / max(1, len(gold))
            rows.append(
                {
                    "state_id": str(state["state_id"]),
                    "question_id": str(state["question_id"]),
                    "dataset": str(state["dataset"]),
                    "source_backend": str(state["backend"]),
                    "backend": backend,
                    "query_index": query_index,
                    "query": query,
                    "signature": json.dumps(signature, ensure_ascii=False),
                    "observed_titles": json.dumps(list(signature), ensure_ascii=False),
                    "prefix_titles": json.dumps(sorted(prefix), ensure_ascii=False),
                    "gold_titles": json.dumps(sorted(gold), ensure_ascii=False),
                    "prefix_support_recall": float(before),
                    "evidence_gain": float(after - before),
                    "support_recall": float(after),
                }
            )
    return rows


def equivalence_edges(group: pd.DataFrame) -> set[tuple[int, int]]:
    signatures = {
        int(row.query_index): str(row.signature) for row in group.itertuples()
    }
    output = set()
    for left, right in itertools.combinations(sorted(signatures), 2):
        if signatures[left] == signatures[right]:
            output.add((left, right))
    return output


def relation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id, state_group in frame.groupby("state_id"):
        backends = sorted(state_group["backend"].unique())
        backend_groups = {
            backend: state_group[state_group["backend"] == backend]
            for backend in backends
        }
        query_ids = sorted(
            set.intersection(
                *[
                    set(group["query_index"].astype(int))
                    for group in backend_groups.values()
                ]
            )
        )
        all_pairs = set(itertools.combinations(query_ids, 2))
        edges = {
            backend: equivalence_edges(group[group["query_index"].isin(query_ids)])
            for backend, group in backend_groups.items()
        }
        for left, right in itertools.combinations(backends, 2):
            union = edges[left] | edges[right]
            agreement = (
                np.mean(
                    [
                        int((pair in edges[left]) == (pair in edges[right]))
                        for pair in all_pairs
                    ]
                )
                if all_pairs
                else 1.0
            )
            jaccard = len(edges[left] & edges[right]) / len(union) if union else 1.0
            best_left = set(
                backend_groups[left]
                .loc[
                    backend_groups[left]["evidence_gain"]
                    == backend_groups[left]["evidence_gain"].max(),
                    "query_index",
                ]
                .astype(int)
            )
            best_right = set(
                backend_groups[right]
                .loc[
                    backend_groups[right]["evidence_gain"]
                    == backend_groups[right]["evidence_gain"].max(),
                    "query_index",
                ]
                .astype(int)
            )
            rows.append(
                {
                    "state_id": state_id,
                    "backend_left": left,
                    "backend_right": right,
                    "relation_agreement": float(agreement),
                    "positive_edge_jaccard": float(jaccard),
                    "best_query_agreement": int(bool(best_left & best_right)),
                    "query_pairs": len(all_pairs),
                }
            )
    return pd.DataFrame(rows)


def _balanced_indices(group: pd.DataFrame, budget: int, seed: int) -> list[int]:
    classes: dict[str, list[int]] = defaultdict(list)
    for row in group.itertuples():
        classes[str(row.signature)].append(int(row.query_index))
    for members in classes.values():
        members.sort(key=lambda value: stable_hash(seed, value))
    class_names = sorted(classes, key=lambda value: stable_hash(seed, value))
    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for name in class_names:
            if depth < len(classes[name]):
                selected.append(classes[name][depth])
                added = True
                if len(selected) >= budget:
                    break
        if not added:
            break
        depth += 1
    return selected


def sampling_rows(
    frame: pd.DataFrame,
    budgets: Sequence[int],
    draws: int = 100,
) -> pd.DataFrame:
    rows = []
    for (state_id, backend), group in frame.groupby(["state_id", "backend"]):
        query_ids = sorted(group["query_index"].astype(int).unique())
        for budget_value in budgets:
            budget = min(int(budget_value), len(query_ids))
            if budget <= 0:
                continue
            for draw in range(int(draws)):
                rng = np.random.default_rng(
                    int(stable_hash(state_id, backend, budget, draw, length=15), 16)
                )
                random_indices = sorted(
                    map(int, rng.choice(query_ids, size=budget, replace=False))
                )
                balanced_indices = _balanced_indices(group, budget, draw)
                for method, indices in (
                    ("random", random_indices),
                    ("behavior-balanced", balanced_indices),
                ):
                    selected = group[group["query_index"].isin(indices)]
                    signatures = set(selected["signature"])
                    first_row = selected.iloc[0]
                    gold = set(json.loads(str(first_row["gold_titles"])))
                    observed = set(json.loads(str(first_row["prefix_titles"])))
                    for raw_titles in selected["observed_titles"]:
                        observed.update(json.loads(str(raw_titles)))
                    union_recall = len(gold & observed) / max(1, len(gold))
                    rows.append(
                        {
                            "state_id": state_id,
                            "backend": backend,
                            "budget": budget,
                            "draw": draw,
                            "method": method,
                            "behavior_coverage": len(signatures) / max(1, budget),
                            "best_evidence_gain": float(
                                selected["evidence_gain"].max()
                            ),
                            "union_support_recall": float(union_recall),
                            "union_evidence_gain": float(
                                union_recall
                                - float(first_row["prefix_support_recall"])
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def run(
    cfg: dict[str, Any],
    profile_name: str,
    inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
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
    colbert_url = os.environ.get("COLBERT_URL", "").strip() or str(
        cfg["retrieval"].get("colbert_url", "")
    ).strip()
    if colbert_url:
        clients["colbert"] = RetrievalClient(
            "colbert",
            colbert_url,
            timeout=int(cfg["retrieval"]["timeout"]),
            retries=int(cfg["retrieval"]["retries"]),
        )
    results = balanced_subset(
        load_state_results(source_patterns(cfg, inputs)),
        int(profile["paired_states"]),
    )
    rows: list[dict[str, Any]] = []
    workers = min(16, max(1, len(results)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_state, result, clients=clients, cfg=cfg): result
            for result in results
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    if not rows:
        raise RuntimeError("Paired retriever grid produced no query executions")
    frame = pd.DataFrame(rows)
    relation = relation_metrics(frame)
    sampling = sampling_rows(
        frame,
        cfg["paired_grid"]["budgets"],
        draws=50 if profile_name == "smoke" else 200,
    )

    output_dir = (
        Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "query_backend_rows.csv", index=False)
    relation.to_csv(output_dir / "relation_stability.csv", index=False)
    sampling.to_csv(output_dir / "sampling_rows.csv", index=False)
    backend_means = frame.groupby("backend", as_index=False).agg(
        evidence_gain=("evidence_gain", "mean"),
        support_recall=("support_recall", "mean"),
        states=("state_id", "nunique"),
    )
    relation_means = relation.groupby(
        ["backend_left", "backend_right"], as_index=False
    ).agg(
        relation_agreement=("relation_agreement", "mean"),
        positive_edge_jaccard=("positive_edge_jaccard", "mean"),
        best_query_agreement=("best_query_agreement", "mean"),
    )
    contrasts = []
    maximum_budget = sampling["budget"].max()
    for backend in sorted(sampling["backend"].unique()):
        subset = sampling[
            (sampling["backend"] == backend)
            & (sampling["budget"] == maximum_budget)
        ]
        pivot = subset.pivot_table(
            index=["state_id", "draw"],
            columns="method",
            values=[
                "behavior_coverage",
                "union_support_recall",
                "union_evidence_gain",
            ],
            aggfunc="first",
        )
        for metric in (
            "behavior_coverage",
            "union_support_recall",
            "union_evidence_gain",
        ):
            paired = pivot[metric][["behavior-balanced", "random"]].dropna().reset_index()
            bootstrap_rows = [
                {
                    "cluster": str(row["state_id"]),
                    "difference": float(
                        row["behavior-balanced"] - row["random"]
                    ),
                }
                for _, row in paired.iterrows()
            ]
            estimate = cluster_bootstrap(
                bootstrap_rows,
                cluster_key="cluster",
                statistic=lambda values: float(
                    np.mean([item["difference"] for item in values])
                ),
                samples=int(profile["bootstrap_samples"]),
                seed=33033 + len(contrasts),
            )
            contrasts.append({"backend": backend, "metric": metric, **estimate})
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output_dir / "sampling_contrasts.csv", index=False)

    gate = cfg["gates"][EXPERIMENT_ID]
    coverage = contrast_frame[
        contrast_frame["metric"] == "union_evidence_gain"
    ]
    positive = coverage[
        (
            coverage["estimate"]
            >= float(gate["minimum_worst_backend_gain"])
        )
        & (coverage["ci_low"] > 0)
    ]
    go = bool(
        frame["backend"].nunique() >= int(gate["minimum_backends"])
        and len(positive) >= int(gate["minimum_positive_backends"])
        and coverage["estimate"].mean()
        >= float(gate["minimum_average_coverage_gain"])
        and coverage["estimate"].min()
        >= float(gate["minimum_worst_backend_gain"])
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "states": int(frame["state_id"].nunique()),
        "backends": sorted(frame["backend"].unique()),
        "colbert_attached": "colbert" in clients,
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-033 Fully paired multi-retriever behavior grid",
        "",
        f"Profile: `{profile_name}`. Identical normalized query strings from each state are executed against BM25, E5, an RRF hybrid, and optional ColBERT.",
        "",
        "## Backend means",
        "",
        markdown_table(backend_means),
        "",
        "## Pairwise behavior-relation stability",
        "",
        markdown_table(relation_means),
        "",
        "## Behavior-balanced minus random sampling",
        "",
        markdown_table(contrast_frame),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP033_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, args.input)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stackpilot.query_credit_common import (
    atomic_write_json,
    bootstrap_by_cluster,
    load_config,
    markdown_table,
    read_jsonl,
    spearman,
)

EXPERIMENT_ID = "EXP-051"


def _candidate_path(cfg: dict[str, Any], profile: str, provided: str | None) -> Path:
    return Path(provided or Path(cfg["work_dir"]) / "labels" / profile / "candidate_credits.jsonl").resolve()


def _top_agreement(rows: Sequence[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    matches = []
    for group in grouped.values():
        best_reward = max(group, key=lambda row: (float(row["full_reward"]), -int(row["candidate_index"])))
        best_credit = max(group, key=lambda row: (float(row[field]), -int(row["candidate_index"])))
        matches.append(float(best_reward["candidate_index"] == best_credit["candidate_index"]))
    return float(np.mean(matches)) if matches else 0.0


def _metric_rows(rows: Sequence[dict[str, Any]], aggregation: str, epsilon: float) -> list[dict[str, Any]]:
    field = f"document_credit_{aggregation}"
    enriched = []
    for row in rows:
        copy = dict(row)
        credit = float(row["document_credit"][aggregation])
        copy[field] = credit
        copy["doc_positive"] = int(credit > epsilon)
        copy["query_indispensable"] = int(float(row["query_indispensability"]) > epsilon)
        copy["false_positive_action_credit"] = int(copy["doc_positive"] and not copy["query_indispensable"])
        copy["alias_bin"] = "1" if int(row["alias_class_size"]) == 1 else ("2" if int(row["alias_class_size"]) == 2 else "3+")
        enriched.append(copy)
    return enriched


def _rate(rows: Sequence[dict[str, Any]], column: str, condition: str | None = None) -> float:
    selected = [row for row in rows if condition is None or int(row[condition]) == 1]
    if not selected:
        return 0.0
    return float(np.mean([float(row[column]) for row in selected]))


def run(cfg: dict[str, Any], profile: str, candidate_path: str | None = None) -> dict[str, Any]:
    path = _candidate_path(cfg, profile, candidate_path)
    rows = read_jsonl([path])
    if not rows:
        raise RuntimeError(f"No query-credit candidate rows in {path}")
    epsilon = float(cfg["labeling"]["replacement_epsilon"])
    samples = int(cfg["profiles"][profile]["bootstrap_samples"])
    summaries: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for aggregation_index, aggregation in enumerate(cfg["labeling"]["document_credit_aggregations"]):
        enriched = _metric_rows(rows, aggregation, epsilon)
        field = f"document_credit_{aggregation}"
        correlation = spearman(
            [float(row[field]) for row in enriched],
            [float(row["query_indispensability"]) for row in enriched],
        )
        fp = bootstrap_by_cluster(
            enriched,
            cluster_key="state_id",
            statistic=lambda values: _rate(values, "false_positive_action_credit", "doc_positive"),
            samples=samples,
            seed=51000 + aggregation_index,
        )
        top_agreement = _top_agreement(enriched, field)
        sign_agreement = float(
            np.mean(
                [
                    float(int(row["doc_positive"]) == int(row["query_indispensable"]))
                    for row in enriched
                ]
            )
        )
        summaries.append(
            {
                "aggregation": aggregation,
                "spearman_doc_vs_indispensability": correlation,
                "false_positive_rate": fp["estimate"],
                "fp_ci_low": fp["ci_low"],
                "fp_ci_high": fp["ci_high"],
                "sign_agreement": sign_agreement,
                "top_query_agreement": top_agreement,
                "states": int(len({row["state_id"] for row in enriched})),
                "queries": len(enriched),
            }
        )
        for alias_bin in ("1", "2", "3+"):
            subset = [row for row in enriched if row["alias_bin"] == alias_bin]
            alias_rows.append(
                {
                    "aggregation": aggregation,
                    "alias_bin": alias_bin,
                    "queries": len(subset),
                    "doc_positive_rate": _rate(subset, "doc_positive"),
                    "false_positive_rate_conditional": _rate(subset, "false_positive_action_credit", "doc_positive"),
                    "mean_indispensability": float(np.mean([float(row["query_indispensability"]) for row in subset])) if subset else 0.0,
                    "mean_document_credit": float(np.mean([float(row[field]) for row in subset])) if subset else 0.0,
                }
            )
        alias_one = next(row for row in alias_rows if row["aggregation"] == aggregation and row["alias_bin"] == "1")
        alias_high = next(row for row in alias_rows if row["aggregation"] == aggregation and row["alias_bin"] == "3+")
        gate = cfg["gates"][EXPERIMENT_ID]
        go = bool(
            correlation <= float(gate["maximum_doc_query_spearman"])
            and fp["estimate"] >= float(gate["minimum_false_positive_rate"])
            and (
                float(alias_high["false_positive_rate_conditional"])
                - float(alias_one["false_positive_rate_conditional"])
                >= float(gate["minimum_alias_high_excess"])
            )
        )
        decisions.append(
            {
                "aggregation": aggregation,
                "go": go,
                "alias_high_excess": float(alias_high["false_positive_rate_conditional"])
                - float(alias_one["false_positive_rate_conditional"]),
            }
        )
    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(summaries).to_csv(output_dir / "surrogate_metrics.csv", index=False)
    pd.DataFrame(alias_rows).to_csv(output_dir / "alias_strata.csv", index=False)
    primary = next(item for item in decisions if item["aggregation"] == "positive-sum")
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile,
        "candidate_file": str(path),
        "primary_aggregation": "positive-sum",
        "primary_go": bool(primary["go"]),
        "aggregations": decisions,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-051 Retrieval-derived query-credit validity audit",
        "",
        f"Profile: `{profile}`. Query indispensability is the reward gap to the best state-matched replacement. Document-derived query credit aggregates document-omission utility from the same query trajectory.",
        "",
        "## Surrogate metrics",
        "",
        markdown_table(summaries),
        "",
        "## Alias-stratified false-positive credit",
        "",
        markdown_table(alias_rows),
        "",
        f"Primary decision (`positive-sum`): **{'GO' if primary['go'] else 'NO-GO'}**.",
        "",
        "A GO means document-derived credit frequently praises replaceable queries, correlates weakly with query indispensability, and becomes more over-attributed in large retrieval-equivalence classes.",
        "",
    ]
    (output_dir / "EXP051_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/query_credit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--candidate-file")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.profile, args.candidate_file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.causal_query_common import (
    bootstrap_by_state,
    finite_number,
    leave_one_out_effect,
    load_causal_query_config,
    spearman,
)
from stackpilot.trace_common import atomic_write_json


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def load_results(result_root: Path, expected_states: int) -> list[dict[str, Any]]:
    paths = sorted(result_root.glob("*/*.json"))
    if len(paths) != expected_states:
        raise RuntimeError(
            f"Expected {expected_states} completed state results under {result_root}; found {len(paths)}"
        )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    signatures = {str(row.get("run_signature", "")) for row in rows}
    if len(signatures) != 1 or not next(iter(signatures)):
        raise RuntimeError(f"State results have inconsistent run signatures: {signatures}")
    return rows


def flatten_candidates(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        state = result["state"]
        for candidate in result["candidates"]:
            row = {
                "state_id": state["state_id"],
                "question_id": state["question_id"],
                "question": state["question"],
                "dataset": state["dataset"],
                "backend": state["backend"],
                "topk": int(state["topk"]),
                "source_turn": int(state["source_turn"]),
                "policy_tag": state["policy_tag"],
                "policy_seed": int(state["policy_seed"]),
                **candidate,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    required_numeric = [
        "immediate_support_gain",
        "recall_after_intervention",
        "final_support_recall",
        "answer_f1",
        "support_tqe",
        "direct_effect",
        "downstream_effect",
        "composite_tqe",
        "intervention_result_novelty",
        "query_question_overlap",
        "query_previous_change",
        "transferred_bridge_token_count",
    ]
    for column in required_numeric:
        if column not in frame or not frame[column].map(finite_number).all():
            raise RuntimeError(f"Candidate report contains missing or non-finite {column}")
        frame[column] = frame[column].astype(float)

    proxy_columns = [
        "intervention_result_novelty",
        "query_question_overlap",
        "query_previous_change",
        "transferred_bridge_token_count",
    ]
    for column in proxy_columns:
        effects = pd.Series(index=frame.index, dtype=float)
        for _state, group in frame.groupby("state_id", sort=False):
            values = leave_one_out_effect(group[column].to_list())
            effects.loc[group.index] = values
        frame[f"{column}_effect"] = effects.astype(float)
    return frame


def _state_group_key(row: dict[str, Any]) -> str:
    return str(row.get("_bootstrap_state") or row["state_id"])


def correlation_stat(rows: list[dict[str, Any]], left: str, right: str) -> float:
    return spearman([float(row[left]) for row in rows], [float(row[right]) for row in rows])


def rate_stat(
    rows: list[dict[str, Any]],
    *,
    numerator: str,
    denominator: callable,
) -> float:
    eligible = [row for row in rows if denominator(row)]
    if not eligible:
        raise RuntimeError(f"No eligible rows for rate {numerator}")
    return float(np.mean([float(row[numerator]) for row in eligible]))


def mean_stat(rows: list[dict[str, Any]], column: str, predicate: callable) -> float:
    values = [float(row[column]) for row in rows if predicate(row)]
    if not values:
        raise RuntimeError(f"No eligible rows for mean {column}")
    return float(np.mean(values))


def top1_accuracy(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_state_group_key(row), []).append(row)
    correct = []
    for candidates in grouped.values():
        best_direct = max(float(row["immediate_support_gain"]) for row in candidates)
        best_final = max(float(row["final_support_recall"]) for row in candidates)
        direct_ids = {
            row["candidate_id"]
            for row in candidates
            if float(row["immediate_support_gain"]) == best_direct
        }
        final_ids = {
            row["candidate_id"]
            for row in candidates
            if float(row["final_support_recall"]) == best_final
        }
        correct.append(float(bool(direct_ids & final_ids)))
    return float(np.mean(correct)) if correct else 0.0


def bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    samples: int,
    epsilon: float,
) -> dict[str, dict[str, float]]:
    records = frame.to_dict("records")
    metrics = {
        "direct_total_spearman": lambda rows: correlation_stat(
            rows, "direct_effect", "support_tqe"
        ),
        "bridge_rate_zero_direct": lambda rows: rate_stat(
            rows,
            numerator="mediated_bridge",
            denominator=lambda row: float(row["immediate_support_gain"]) <= epsilon,
        ),
        "positive_causal_bridge_rate_zero_direct": lambda rows: rate_stat(
            rows,
            numerator="positive_causal_bridge",
            denominator=lambda row: float(row["immediate_support_gain"]) <= epsilon,
        ),
        "redundant_direct_rate": lambda rows: rate_stat(
            rows,
            numerator="redundant_direct",
            denominator=lambda row: float(row["immediate_support_gain"]) > epsilon,
        ),
        "bridge_downstream_effect": lambda rows: mean_stat(
            rows,
            "downstream_effect",
            lambda row: int(row["mediated_bridge"]) == 1,
        ),
        "immediate_top1_accuracy": top1_accuracy,
    }
    return {
        name: bootstrap_by_state(
            records,
            state_key="state_id",
            statistic=statistic,
            samples=samples,
            seed=13000 + index,
        )
        for index, (name, statistic) in enumerate(metrics.items())
    }


def proxy_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    proxies = {
        "immediate_support_gain": "direct_effect",
        "result_novelty": "intervention_result_novelty_effect",
        "question_overlap": "query_question_overlap_effect",
        "query_change": "query_previous_change_effect",
        "bridge_token_transfer": "transferred_bridge_token_count_effect",
    }
    rows = []
    for name, column in proxies.items():
        rows.append(
            {
                "proxy": name,
                "spearman_with_support_tqe": spearman(
                    frame[column].astype(float).to_list(),
                    frame["support_tqe"].astype(float).to_list(),
                ),
                "spearman_with_composite_tqe": spearman(
                    frame[column].astype(float).to_list(),
                    frame["composite_tqe"].astype(float).to_list(),
                ),
            }
        )
    return pd.DataFrame(rows)


def prevalence_table(frame: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    scopes = [("combined", frame)] + [
        (str(backend), group) for backend, group in frame.groupby("backend")
    ]
    rows = []
    for scope, group in scopes:
        zero = group[group["immediate_support_gain"] <= epsilon]
        direct = group[group["immediate_support_gain"] > epsilon]
        rows.append(
            {
                "scope": scope,
                "candidates": len(group),
                "zero_direct_candidates": len(zero),
                "mediated_bridge_rate_zero_direct": (
                    float(zero["mediated_bridge"].mean()) if len(zero) else float("nan")
                ),
                "positive_causal_bridge_rate_zero_direct": (
                    float(zero["positive_causal_bridge"].mean())
                    if len(zero)
                    else float("nan")
                ),
                "direct_candidates": len(direct),
                "redundant_direct_rate": (
                    float(direct["redundant_direct"].mean())
                    if len(direct)
                    else float("nan")
                ),
                "mean_bridge_downstream_effect": (
                    float(group.loc[group["mediated_bridge"] == 1, "downstream_effect"].mean())
                    if int(group["mediated_bridge"].sum())
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def subgroup_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["backend", "dataset", "source_turn"]):
        zero = group[group["immediate_support_gain"] <= 1e-8]
        rows.append(
            {
                "backend": keys[0],
                "dataset": keys[1],
                "source_turn": int(keys[2]),
                "states": group["state_id"].nunique(),
                "candidates": len(group),
                "direct_total_spearman": spearman(
                    group["direct_effect"].to_list(), group["support_tqe"].to_list()
                ),
                "bridge_rate_zero_direct": (
                    float(zero["mediated_bridge"].mean()) if len(zero) else float("nan")
                ),
                "mean_support_tqe": float(group["support_tqe"].mean()),
            }
        )
    return pd.DataFrame(rows)


def example_table(frame: pd.DataFrame, *, kind: str, count: int) -> pd.DataFrame:
    if kind == "bridge":
        subset = frame[frame["mediated_bridge"] == 1].sort_values(
            ["support_tqe", "next_query_evidence_gain"], ascending=False
        )
    elif kind == "redundant":
        subset = frame[frame["redundant_direct"] == 1].sort_values(
            ["immediate_support_gain", "support_tqe"], ascending=[False, True]
        )
    else:
        raise ValueError(kind)
    columns = [
        "backend",
        "dataset",
        "question",
        "style",
        "query",
        "immediate_support_gain",
        "support_tqe",
        "next_query",
        "next_query_evidence_gain",
        "transferred_bridge_tokens",
    ]
    return subset.head(count)[columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the EXP-013 causal query signal audit.")
    parser.add_argument("--config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--states-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_causal_query_config(args.config)
    profile = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    states_root = Path(args.states_root or work_root / "states" / args.profile)
    state_manifest_path = states_root / "manifest.json"
    if not state_manifest_path.is_file():
        raise RuntimeError(f"Missing state manifest: {state_manifest_path}")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    expected_states = int(state_manifest["selected_states"])
    result_root = Path(
        args.result_root or work_root / "results" / args.profile / "states"
    )
    output_dir = Path(args.output_dir or work_root / "reports" / args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(result_root, expected_states)
    frame = flatten_candidates(results)
    epsilon = float(cfg["analysis"]["epsilon"])
    bootstrap = bootstrap_metrics(
        frame,
        samples=int(profile["bootstrap_samples"]),
        epsilon=epsilon,
    )
    proxy_frame = proxy_correlations(frame)
    prevalence = prevalence_table(frame, epsilon)
    subgroups = subgroup_table(frame)
    style_summary = (
        frame.groupby(["backend", "style"], as_index=False)[
            [
                "immediate_support_gain",
                "final_support_recall",
                "support_tqe",
                "downstream_effect",
                "answer_f1",
            ]
        ]
        .mean()
        .sort_values(["backend", "style"])
    )
    bridge_examples = example_table(
        frame, kind="bridge", count=int(cfg["analysis"]["top_examples"])
    )
    redundant_examples = example_table(
        frame, kind="redundant", count=int(cfg["analysis"]["top_examples"])
    )

    frame.to_csv(output_dir / "candidate_metrics.csv", index=False)
    proxy_frame.to_csv(output_dir / "proxy_correlations.csv", index=False)
    prevalence.to_csv(output_dir / "prevalence.csv", index=False)
    subgroups.to_csv(output_dir / "subgroups.csv", index=False)
    style_summary.to_csv(output_dir / "style_summary.csv", index=False)
    bridge_examples.to_csv(output_dir / "bridge_examples.csv", index=False)
    redundant_examples.to_csv(output_dir / "redundant_examples.csv", index=False)
    atomic_write_json(output_dir / "bootstrap_metrics.json", bootstrap)

    combined_prevalence = prevalence[prevalence["scope"] == "combined"].iloc[0]
    backend_bridge_counts = (
        frame[frame["mediated_bridge"] == 1].groupby("backend").size().to_dict()
    )
    gate_cfg = cfg["gate"]
    direct_supported = bool(
        bootstrap["direct_total_spearman"]["estimate"]
        <= float(gate_cfg["maximum_direct_total_spearman"])
    )
    bridge_supported = bool(
        float(combined_prevalence["mediated_bridge_rate_zero_direct"])
        >= float(gate_cfg["minimum_bridge_rate_among_zero_direct"])
        and bootstrap["bridge_downstream_effect"]["estimate"]
        >= float(gate_cfg["minimum_bridge_downstream_effect"])
        and bootstrap["bridge_downstream_effect"]["ci_low"] > 0.0
    )
    redundant_supported = bool(
        float(combined_prevalence["redundant_direct_rate"])
        >= float(gate_cfg["minimum_redundant_direct_rate"])
    )
    both_backends = all(int(backend_bridge_counts.get(name, 0)) > 0 for name in ("bm25", "e5"))
    if not bool(gate_cfg.get("require_bridge_examples_in_both_backends", True)):
        both_backends = True
    decision = {
        "local_signal_incomplete": direct_supported,
        "bridge_mediation_present": bridge_supported,
        "redundant_direct_queries_present": redundant_supported,
        "bridge_examples_in_both_backends": both_backends,
        "all_conditions": direct_supported
        and bridge_supported
        and redundant_supported
        and both_backends,
    }
    atomic_write_json(output_dir / "decision.json", decision)

    metrics_rows = pd.DataFrame(
        [
            {"metric": name, **values}
            for name, values in bootstrap.items()
        ]
    )
    report = [
        "# EXP-013 Causal Query Signal Audit",
        "",
        f"Profile: `{args.profile}`. States: {expected_states}. Candidate branches: {len(frame)}.",
        "",
        "Each intervention state uses one factual query plus state-matched alternatives. Every candidate is executed against the same retriever snapshot and prefix, then the same frozen Qwen2.5-7B continuation policy replays the remaining suffix.",
        "",
        "## Primary diagnostic metrics",
        "",
        markdown_table(metrics_rows),
        "",
        "The direct-total correlation compares leave-one-out immediate support effect with leave-one-out final support effect. A low value means local evidence gain is not a sufficient causal-credit proxy.",
        "",
        "## Proxy correlations",
        "",
        markdown_table(proxy_frame),
        "",
        "## Bridge and redundancy prevalence",
        "",
        markdown_table(prevalence),
        "",
        "A mediated bridge candidate has zero immediate support gain, transfers at least one newly observed title token into the next generated query, and the next search gains supporting evidence. A redundant direct candidate gains evidence immediately but has non-positive total support effect relative to its state-matched alternatives.",
        "",
        "## Subgroups",
        "",
        markdown_table(subgroups),
        "",
        "## Candidate style means",
        "",
        markdown_table(style_summary),
        "",
        "## High-utility bridge examples",
        "",
        markdown_table(bridge_examples, digits=3),
        "",
        "## Redundant direct-evidence examples",
        "",
        markdown_table(redundant_examples, digits=3),
        "",
        "## Decision",
        "",
        f"- Immediate gain is incomplete: **{'PASS' if direct_supported else 'FAIL'}**",
        f"- Bridge mediation is frequent and useful: **{'PASS' if bridge_supported else 'FAIL'}**",
        f"- Redundant direct queries are nontrivial: **{'PASS' if redundant_supported else 'FAIL'}**",
        f"- Bridge examples occur in both backends: **{'PASS' if both_backends else 'FAIL'}**",
        "",
        ("**GO:** develop a query-level causal mediation estimator and policy-credit method."
         if decision["all_conditions"]
         else "**NO-GO:** the audit does not justify a costly causal query-credit method."),
        "",
        "This is a signal audit, not a final policy result. A GO must still be followed by a learned low-cost estimator and an interactive policy comparison against immediate evidence gain, STAMP-style provenance, information gain, and outcome-only RL under matched search-call budgets.",
    ]
    (output_dir / "CAUSAL_QUERY_AUDIT_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()

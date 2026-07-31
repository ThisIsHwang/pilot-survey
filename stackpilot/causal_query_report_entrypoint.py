from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from stackpilot import causal_query_report as implementation


def rate_stat(
    rows: list[dict[str, Any]],
    *,
    numerator: str,
    denominator: Callable[[dict[str, Any]], bool],
) -> float:
    eligible = [row for row in rows if denominator(row)]
    if not eligible:
        return 0.0
    return float(np.mean([float(row[numerator]) for row in eligible]))


def mean_stat(
    rows: list[dict[str, Any]],
    column: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    values = [float(row[column]) for row in rows if predicate(row)]
    return float(np.mean(values)) if values else 0.0


def top1_accuracy(rows: list[dict[str, Any]]) -> float:
    """Tie-aware probability that an immediate-gain maximizer is final-optimal."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("_bootstrap_state") or row["state_id"])
        grouped.setdefault(key, []).append(row)
    scores = []
    for candidates in grouped.values():
        best_direct = max(float(row["immediate_support_gain"]) for row in candidates)
        best_final = max(float(row["final_support_recall"]) for row in candidates)
        direct_ids = {
            str(row["candidate_id"])
            for row in candidates
            if float(row["immediate_support_gain"]) == best_direct
        }
        final_ids = {
            str(row["candidate_id"])
            for row in candidates
            if float(row["final_support_recall"]) == best_final
        }
        scores.append(len(direct_ids & final_ids) / max(1, len(direct_ids)))
    return float(np.mean(scores)) if scores else 0.0


def prevalence_table(frame: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    table = implementation._ORIGINAL_PREVALENCE_TABLE(frame, epsilon)
    columns = [
        "mediated_bridge_rate_zero_direct",
        "positive_causal_bridge_rate_zero_direct",
        "redundant_direct_rate",
        "mean_bridge_downstream_effect",
    ]
    for column in columns:
        if column in table:
            table[column] = table[column].fillna(0.0).astype(float)
    return table


def subgroup_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = implementation._ORIGINAL_SUBGROUP_TABLE(frame)
    if "bridge_rate_zero_direct" in table:
        table["bridge_rate_zero_direct"] = (
            table["bridge_rate_zero_direct"].fillna(0.0).astype(float)
        )
    return table


# Save the implementations before replacing their module globals. Main and the
# bootstrap lambdas resolve these names dynamically, so the hardened definitions
# below apply to every CLI report without duplicating the analysis pipeline.
implementation._ORIGINAL_PREVALENCE_TABLE = implementation.prevalence_table
implementation._ORIGINAL_SUBGROUP_TABLE = implementation.subgroup_table
implementation.rate_stat = rate_stat
implementation.mean_stat = mean_stat
implementation.top1_accuracy = top1_accuracy
implementation.prevalence_table = prevalence_table
implementation.subgroup_table = subgroup_table


if __name__ == "__main__":
    implementation.main()

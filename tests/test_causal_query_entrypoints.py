from __future__ import annotations

import unittest

import pandas as pd

from stackpilot.causal_query_model_contract import validate_parameter_count
from stackpilot.causal_query_prepare_entrypoint import build_candidate_states
from stackpilot.causal_query_report_entrypoint import (
    mean_stat,
    prevalence_table,
    rate_stat,
    top1_accuracy,
)


class CausalQueryEntrypointTests(unittest.TestCase):
    def test_unresolved_cohort_filter_removes_solved_prefixes(self) -> None:
        cfg = {
            "source": {
                "datasets": ["toy"],
                "policy_tags": [],
                "topks": [3],
                "intervention_turns": [2],
                "require_protocol_success": True,
                "maximum_prefix_support_recall": 0.5,
            },
            "agent": {"max_search_turns": 4},
        }

        def episode(question_id: str, prefix_recall: float) -> dict:
            return {
                "question_id": question_id,
                "question": "Who wrote it?",
                "answers": ["Author"],
                "support_titles": ["Book", "Author"],
                "dataset": "toy",
                "backend": "bm25",
                "topk": 3,
                "policy_tag": "teacher",
                "seed": 13,
                "protocol_failure": 0,
                "turns": [
                    {
                        "query": "book title",
                        "observed_titles": ["Book"],
                        "support_recall": prefix_recall,
                        "evidence_gain": prefix_recall,
                    },
                    {
                        "query": "Book author",
                        "observed_titles": ["Author"],
                        "support_recall": 1.0,
                        "evidence_gain": 1.0 - prefix_recall,
                    },
                ],
            }

        rows = [(episode("unresolved", 0.5), "raw"), (episode("solved", 1.0), "raw")]
        states = build_candidate_states(rows, cfg=cfg)
        self.assertEqual([row["question_id"] for row in states], ["unresolved"])
        self.assertEqual(states[0]["prefix_support_recall"], 0.5)

    def test_sparse_rates_and_means_return_zero(self) -> None:
        rows = [{"eligible": 0, "value": 1.0}]
        self.assertEqual(
            rate_stat(
                rows,
                numerator="value",
                denominator=lambda row: bool(row["eligible"]),
            ),
            0.0,
        )
        self.assertEqual(
            mean_stat(rows, "value", lambda row: bool(row["eligible"])),
            0.0,
        )

    def test_top1_accuracy_is_tie_aware(self) -> None:
        rows = [
            {
                "state_id": "s",
                "candidate_id": "a",
                "immediate_support_gain": 0.0,
                "final_support_recall": 1.0,
            },
            {
                "state_id": "s",
                "candidate_id": "b",
                "immediate_support_gain": 0.0,
                "final_support_recall": 0.0,
            },
        ]
        self.assertEqual(top1_accuracy(rows), 0.5)

    def test_prevalence_table_has_no_nan_for_empty_categories(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "backend": "bm25",
                    "immediate_support_gain": 0.0,
                    "mediated_bridge": 0,
                    "positive_causal_bridge": 0,
                    "redundant_direct": 0,
                    "downstream_effect": 0.0,
                }
            ]
        )
        table = prevalence_table(frame, 1e-8)
        numeric = table.select_dtypes(include=["number"])
        self.assertFalse(numeric.isna().any().any())
        combined = table[table["scope"] == "combined"].iloc[0]
        self.assertEqual(float(combined["redundant_direct_rate"]), 0.0)
        self.assertEqual(float(combined["mean_bridge_downstream_effect"]), 0.0)

    def test_model_contract_rejects_stale_3b_checkpoint(self) -> None:
        validate_parameter_count(
            7_600_000_000,
            minimum_parameters=6_000_000_000,
            maximum_parameters=9_000_000_000,
        )
        with self.assertRaises(RuntimeError):
            validate_parameter_count(
                3_100_000_000,
                minimum_parameters=6_000_000_000,
                maximum_parameters=9_000_000_000,
            )


if __name__ == "__main__":
    unittest.main()

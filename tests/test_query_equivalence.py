from __future__ import annotations

import unittest

import pandas as pd

from stackpilot.query_equivalence_common import (
    EquivalenceThresholds,
    candidates_equivalent,
    class_summary,
    equivalence_classes,
    support_set,
)
from stackpilot.query_equivalence_plan import _training_rows
from stackpilot.query_equivalence_report import class_metrics


def candidate(
    candidate_id: str,
    *,
    style: str,
    support_titles: list[str],
    immediate_gain: float,
    answer_f1: float = 1.0,
    searches: int = 2,
    origin: str = "alternative",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "style": style,
        "origin": origin,
        "query": f"query {candidate_id}",
        "prompt": "prompt",
        "immediate_support_gain": immediate_gain,
        "final_support_recall": len(support_titles) / 2,
        "answer_em": 1,
        "answer_f1": answer_f1,
        "total_search_count": searches,
        "protocol_failure": 0,
        "invalid_action_count": 0,
        "branch_turns": [
            {"turn": 2, "observed_titles": support_titles, "support_recall": len(support_titles) / 2}
        ],
    }


STATE = {
    "state_id": "s1", "question_id": "q1", "question": "question",
    "dataset": "hotpotqa", "backend": "bm25", "topk": 3,
    "source_turn": 2, "policy_tag": "base", "policy_seed": 13,
    "support_titles": ["Gold A", "Gold B"], "prior_turns": [],
}


class QueryEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = EquivalenceThresholds()

    def test_strict_equivalence_uses_final_support_set(self) -> None:
        left = candidate("a", style="factual", support_titles=["Gold A"], immediate_gain=0.5, origin="factual")
        right = candidate("b", style="semantic", support_titles=["Gold A"], immediate_gain=0.5)
        different = candidate("c", style="lexical", support_titles=["Gold B"], immediate_gain=0.5)
        self.assertTrue(candidates_equivalent(STATE, left, right, self.thresholds))
        self.assertFalse(candidates_equivalent(STATE, left, different, self.thresholds))
        self.assertEqual(support_set(STATE, left), ("gold a",))

    def test_equivalence_class_and_credit_overallocation(self) -> None:
        rows = [
            candidate("a", style="factual", support_titles=["Gold A"], immediate_gain=0.5, origin="factual"),
            candidate("b", style="semantic", support_titles=["Gold A"], immediate_gain=0.5),
            candidate("c", style="lexical", support_titles=["Gold B"], immediate_gain=0.5),
        ]
        groups = equivalence_classes(STATE, rows, self.thresholds)
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        group = next(value for value in groups if len(value) == 2)
        summary = class_summary(STATE, rows, group, epsilon=1e-8)
        self.assertTrue(summary["factual_member"])
        self.assertEqual(summary["class_size"], 2)
        self.assertAlmostEqual(summary["exclusive_credit_overallocation"], 0.5)

    def test_training_variants_use_same_state_credit(self) -> None:
        candidates = {
            "a": {"candidate_id": "a", "style": "factual", "origin": "factual", "prompt": "p", "query": "qa"},
            "b": {"candidate_id": "b", "style": "semantic", "origin": "alternative", "prompt": "p", "query": "qb"},
        }
        state = {"state_id": "s1", "selected_class": {"class_id": "c1", "member_candidate_ids": ["a", "b"]}}
        factual = _training_rows([state], candidates, variant="factual-onehot", seed=13)
        random_rows = _training_rows([state], candidates, variant="random-onehot", seed=13)
        equivalent = _training_rows([state], candidates, variant="equivalence-normalized", seed=13)
        self.assertEqual(len(factual), 1)
        self.assertEqual(len(random_rows), 1)
        self.assertEqual(len(equivalent), 2)
        self.assertEqual({row["state_id"] for row in equivalent}, {"s1"})
        self.assertTrue(all(row["weight"] == 1.0 for row in equivalent))

    def test_class_metrics_preserve_worst_member_and_variance(self) -> None:
        frame = pd.DataFrame([
            {
                "direction": "bm25-to-e5", "variant": "equivalence-normalized",
                "seed": 13, "class_id": "c1", "origin": "factual",
                "baseline_nll": 2.0, "adapted_nll": 1.8, "heldout_gain": 0.2,
            },
            {
                "direction": "bm25-to-e5", "variant": "equivalence-normalized",
                "seed": 13, "class_id": "c1", "origin": "alternative",
                "baseline_nll": 2.2, "adapted_nll": 2.1, "heldout_gain": 0.1,
            },
        ])
        result = class_metrics(frame).iloc[0]
        self.assertAlmostEqual(result["class_mean_gain"], 0.15)
        self.assertAlmostEqual(result["class_worst_gain"], 0.1)
        self.assertAlmostEqual(result["class_gain_std"], 0.05)
        self.assertEqual(result["positive_member_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

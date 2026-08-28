from __future__ import annotations

import unittest

import numpy as np

from stackpilot.query_credit_weekend_common import (
    aggregate_swap_credit,
    apply_fixed_cardinality_swap,
    choose_length_matched_replacements,
    pairwise_preference_accuracy,
    shaped_signal,
    stable_balanced_sample,
    state_audit_metrics,
    top1_regret,
    two_way_paired_bootstrap,
)


class FakeTokenizer:
    def __call__(self, value: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(len(value.split())))}


class WeekendCommonTest(unittest.TestCase):
    def test_balanced_sampling_is_result_blind_and_cell_balanced(self) -> None:
        rows = []
        for dataset in ("a", "b"):
            for backend in ("bm25", "e5"):
                for index in range(5):
                    rows.append(
                        {
                            "state": {
                                "dataset": dataset,
                                "backend": backend,
                                "state_id": f"{dataset}-{backend}-{index}",
                                "question_id": f"q-{dataset}-{index}",
                                "outcome_that_must_not_be_used": 1000 - index,
                            }
                        }
                    )
        first, counts = stable_balanced_sample(
            rows,
            datasets=["a", "b"],
            backends=["bm25", "e5"],
            per_cell=2,
            salt="test",
        )
        second, _ = stable_balanced_sample(
            list(reversed(rows)),
            datasets=["a", "b"],
            backends=["bm25", "e5"],
            per_cell=2,
            salt="test",
        )
        self.assertEqual(counts, {"a/bm25": 2, "a/e5": 2, "b/bm25": 2, "b/e5": 2})
        self.assertEqual(
            [row["state"]["state_id"] for row in first],
            [row["state"]["state_id"] for row in second],
        )
        by_cell = {}
        for row in first:
            state = row["state"]
            by_cell.setdefault((state["dataset"], state["backend"]), set()).add(state["question_id"])
        self.assertEqual(by_cell[("a", "bm25")], by_cell[("a", "e5")])
        self.assertEqual(by_cell[("b", "bm25")], by_cell[("b", "e5")])

    def test_replacement_keeps_three_documents_and_never_uses_visible_rank(self) -> None:
        results = [
            {"title": f"D{index}", "text": "x " * (index + 1), "score": 10 - index}
            for index in range(10)
        ]
        plan = choose_length_matched_replacements(
            results,
            visible_documents=3,
            pool_start_rank=4,
            pool_end_rank=10,
            tokenizer=FakeTokenizer(),
        )
        self.assertEqual(len(plan), 3)
        self.assertEqual(len({row["replacement_index"] for row in plan}), 3)
        self.assertTrue(all(row["replacement_index"] >= 3 for row in plan))
        swapped = apply_fixed_cardinality_swap(
            results,
            visible_documents=3,
            slot=1,
            replacement_index=plan[1]["replacement_index"],
        )
        self.assertEqual(len(swapped), 3)
        self.assertEqual(swapped[0]["title"], "D0")
        self.assertEqual(swapped[2]["title"], "D2")
        self.assertNotEqual(swapped[1]["title"], "D1")

    def test_pairwise_ties_receive_half_credit(self) -> None:
        self.assertAlmostEqual(pairwise_preference_accuracy([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(pairwise_preference_accuracy([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(pairwise_preference_accuracy([1, 0], [0, 0]), 0.5)

    def test_top1_regret_is_normalized_by_state_range(self) -> None:
        result = top1_regret([0.0, 0.5, 1.0], [1.0, 0.0, -1.0])
        self.assertAlmostEqual(result["regret"], 1.0)
        self.assertAlmostEqual(result["normalized_regret"], 1.0)
        self.assertEqual(result["agreement"], 0.0)

    def test_swap_credit_preserves_negative_effects(self) -> None:
        result = aggregate_swap_credit([[0.4, -0.4, 0.0], [0.2, -0.2, 0.0]])
        self.assertAlmostEqual(result["signed_mean"], 0.0)
        self.assertAlmostEqual(result["positive_sum"], 0.3)

    def test_shaped_signal_matches_outcome_rms(self) -> None:
        outcome = [0.0, 1.0, 2.0]
        document = [2.0, 0.0, 1.0]
        shaped = shaped_signal(outcome, document, alpha=0.5)
        baseline = (np.asarray(outcome) - np.mean(outcome)) / np.std(outcome)
        self.assertAlmostEqual(float(np.mean(shaped)), 0.0, places=6)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(shaped**2))),
            float(np.sqrt(np.mean(baseline**2))),
            places=5,
        )

    def test_two_way_bootstrap_preserves_seed_and_question_axes(self) -> None:
        rows = [
            {"seed": seed, "question": question, "value": 0.2}
            for seed in (1, 2, 3)
            for question in ("a", "b", "c")
        ]
        result = two_way_paired_bootstrap(
            rows,
            seed_key="seed",
            item_key="question",
            value_key="value",
            samples=200,
            seed=7,
        )
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertEqual(result["seeds"], 3.0)
        self.assertEqual(result["items"], 3.0)

    def test_state_audit_uses_seed_halves_and_signed_swap(self) -> None:
        rows = [
            {
                "full_seed_rewards": {"task": [1.0, 1.1, 0.9, 1.0]},
                "swap_credit": {"task": {"signed_mean": 0.0, "per_seed_signed_mean": [0.0, 0.0, 0.0, 0.0]}},
            },
            {
                "full_seed_rewards": {"task": [0.0, 0.1, -0.1, 0.0]},
                "swap_credit": {"task": {"signed_mean": 1.0, "per_seed_signed_mean": [1.0, 1.0, 1.0, 1.0]}},
            },
        ]
        metrics = state_audit_metrics(
            rows,
            reward_view="task",
            document_signal="signed_mean",
            epsilon=1e-9,
        )
        self.assertEqual(metrics["action_self_pairwise"], 1.0)
        self.assertEqual(metrics["document_action_pairwise"], 0.0)
        self.assertEqual(metrics["reliability_gap"], 1.0)


if __name__ == "__main__":
    unittest.main()

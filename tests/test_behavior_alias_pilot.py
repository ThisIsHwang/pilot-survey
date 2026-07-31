from __future__ import annotations

import unittest

from stackpilot.behavior_alias_common import (
    behavior_key,
    build_injected_pool,
    choose_injection_class,
    natural_alias_metrics,
    select_queries,
    selected_metrics,
    valid_query,
)
from stackpilot.behavior_alias_simulation import contrast_stat


def toy_state() -> dict:
    return {
        "state_id": "s",
        "question_id": "q",
        "question": "Who is older, Alpha or Beta?",
        "backend": "bm25",
        "dataset": "toy",
        "source_turn": 2,
        "support_titles": ["Alpha", "Beta"],
        "prior_turns": [
            {
                "turn": 1,
                "query": "Alpha age",
                "observed_titles": [],
            }
        ],
        "prefix_support_recall": 0.0,
        "injection_class_id": "bad",
        "classes": [
            {
                "class_id": "good-a",
                "queries": ["Alpha birth date"],
                "observed_titles": ["Alpha"],
                "natural_alias_count": 1,
                "support_gain": 0.5,
                "reward": 0.5,
            },
            {
                "class_id": "good-b",
                "queries": ["Beta birth date"],
                "observed_titles": ["Beta"],
                "natural_alias_count": 1,
                "support_gain": 0.5,
                "reward": 0.5,
            },
            {
                "class_id": "bad",
                "queries": ["Alpha biography", "Alpha profile"],
                "observed_titles": ["Unrelated"],
                "natural_alias_count": 2,
                "support_gain": 0.0,
                "reward": 0.0,
            },
        ],
    }


class BehaviorAliasPilotTests(unittest.TestCase):
    def test_behavior_key_is_rank_sensitive_and_exact(self) -> None:
        self.assertEqual(behavior_key(["Alpha", "Beta"]), behavior_key(["Alpha", "Beta"]))
        self.assertNotEqual(behavior_key(["Alpha", "Beta"]), behavior_key(["Beta", "Alpha"]))

    def test_query_validation_requires_known_anchor(self) -> None:
        self.assertTrue(
            valid_query(
                "Alpha birth date",
                question="Who is older, Alpha or Beta?",
                observed_titles=[],
                minimum_tokens=2,
                maximum_tokens=10,
            )
        )
        self.assertFalse(
            valid_query(
                "unrelated banana catalog",
                question="Who is older, Alpha or Beta?",
                observed_titles=[],
                minimum_tokens=2,
                maximum_tokens=10,
            )
        )

    def test_injection_class_prefers_largest_nonbest_class(self) -> None:
        classes = toy_state()["classes"]
        self.assertEqual(
            choose_injection_class(classes, minimum_reward_gap=0.1),
            "bad",
        )

    def test_natural_alias_metrics_count_behavior_classes(self) -> None:
        candidates = [
            {"class_id": "a"},
            {"class_id": "a"},
            {"class_id": "b"},
            {"class_id": "c"},
        ]
        metrics = natural_alias_metrics(candidates)
        self.assertEqual(metrics["surface_queries"], 4.0)
        self.assertEqual(metrics["behavior_classes"], 3.0)
        self.assertAlmostEqual(metrics["alias_fraction"], 0.25)

    def test_quotient_selection_covers_classes_before_repeating(self) -> None:
        state = toy_state()
        pool = build_injected_pool(state, multiplicity=8)
        surface = select_queries(pool, method="surface", budget=3, seed=1)
        quotient = select_queries(pool, method="quotient", budget=3, seed=1)
        self.assertLessEqual(len({row["class_id"] for row in surface}), 3)
        self.assertEqual(len({row["class_id"] for row in quotient}), 3)

    def test_selected_metrics_measure_union_evidence(self) -> None:
        state = toy_state()
        pool = build_injected_pool(state, multiplicity=1)
        selected = [
            next(row for row in pool if row["class_id"] == "good-a"),
            next(row for row in pool if row["class_id"] == "good-b"),
        ]
        metrics = selected_metrics(state, selected)
        self.assertEqual(metrics["unique_classes"], 2.0)
        self.assertEqual(metrics["union_support_recall"], 1.0)
        self.assertEqual(metrics["union_support_gain"], 1.0)

    def test_contrast_stat_is_state_paired(self) -> None:
        rows = []
        for state in ("a", "b"):
            rows.extend(
                [
                    {
                        "state_id": state,
                        "method": "surface",
                        "multiplicity": 1,
                        "class_coverage": 1.0,
                    },
                    {
                        "state_id": state,
                        "method": "surface",
                        "multiplicity": 8,
                        "class_coverage": 0.5,
                    },
                ]
            )
        self.assertAlmostEqual(
            contrast_stat(
                rows,
                metric="class_coverage",
                method_a="surface",
                multiplicity_a=1,
                method_b="surface",
                multiplicity_b=8,
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()

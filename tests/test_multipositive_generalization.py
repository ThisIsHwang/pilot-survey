from __future__ import annotations

import unittest

from stackpilot.multipositive_common import select_pair
from stackpilot.multipositive_plan import core_specs, style_specs
from stackpilot.multipositive_interactive_job import mean_pairwise_query_distance, transition_signature


def candidate(candidate_id: str, query: str, style: str, *, origin: str = "alternative", direct: int = 1, strict: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "query": query,
        "style": style,
        "origin": origin,
        "factual": int(origin == "factual"),
        "protocol_failure": 0,
        "direct": direct,
        "immediate_support_set": ["a"] if direct else [],
        "final_support_set": ["a", "b"] if strict else ["a"],
        "answer_em": 1.0 if strict else 0.0,
        "immediate_support_gain": 0.5 if direct else 0.0,
        "final_support_recall": 1.0 if strict else 0.5,
    }


def state() -> dict:
    return {
        "state_id": "s1",
        "question_id": "q1",
        "candidates": [
            candidate("f", "alpha creator date", "factual", origin="factual", strict=True),
            candidate("l", "alpha maker birthday", "lexical", strict=True),
            candidate("s", "when was alpha's creator born", "semantic", strict=False),
            candidate("e", "alpha creator", "entity", strict=False),
        ],
    }


class MultipositiveGeneralizationTests(unittest.TestCase):
    def test_style_holdout_never_selects_excluded_style(self) -> None:
        rows = select_pair(state(), selector="all_direct", seed=13, excluded_style="semantic", maximum_random_query_jaccard=0.95)
        self.assertNotIn("semantic", [row["style"] for row in rows])

    def test_strict_selector_uses_functional_partner(self) -> None:
        rows = select_pair(state(), selector="strict", seed=13, excluded_style=None, maximum_random_query_jaccard=0.95)
        self.assertEqual({row["candidate_id"] for row in rows}, {"f", "l"})

    def test_every_core_selector_has_uniform_and_consistency_controls(self) -> None:
        specs = core_specs()
        for stem in ("random", "diversity", "all-direct", "strict"):
            self.assertIn(f"{stem}-uniform", specs)
            self.assertIn(f"{stem}-consistency", specs)

    def test_style_matrix_has_three_methods_per_fold(self) -> None:
        specs = style_specs(["lexical", "semantic", "entity"])
        self.assertEqual(len(specs), 9)
        self.assertTrue(all(spec["experiment_id"] == "EXP-024" for spec in specs.values()))

    def test_behavior_signature_is_order_sensitive(self) -> None:
        self.assertNotEqual(transition_signature(["A", "B"]), transition_signature(["B", "A"]))

    def test_query_diversity_is_finite(self) -> None:
        value = mean_pairwise_query_distance(["alpha creator", "when was alpha made"])
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

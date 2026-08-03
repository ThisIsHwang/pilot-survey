from __future__ import annotations

import unittest

import numpy as np

from stackpilot.interface_alias_audit import state_alias_rows
from stackpilot.interface_causality_common import (
    behavior_signature,
    classification_metrics,
    group_candidates,
    normalize_advantages,
)
from stackpilot.interface_credit_granularity import state_credit_rows
from stackpilot.interface_equivalence_predictor import feature_sets, pair_rows, split_rows
from stackpilot.interface_expressivity_audit import menu_queries, prefix_titles, state_interface_rows


def candidate(
    candidate_id: str,
    query: str,
    titles: list[str],
    *,
    style: str,
    origin: str = "alternative",
    gain: float = 0.5,
    final_recall: float = 1.0,
    support_tqe: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "query": query,
        "style": style,
        "origin": origin,
        "intervention_observed_titles": titles,
        "immediate_support_gain": gain,
        "final_support_recall": final_recall,
        "answer_em": 1.0,
        "answer_f1": 1.0,
        "total_search_count": 2,
        "protocol_failure": 0,
        "support_tqe": support_tqe,
        "composite_tqe": support_tqe,
        "branch_turns": [
            {"observed_titles": titles, "support_recall": final_recall}
        ],
    }


def result() -> dict:
    return {
        "run_signature": "run",
        "state": {
            "state_id": "s1",
            "question_id": "q1",
            "question": "When did the creator of Gold A die?",
            "dataset": "hotpotqa",
            "backend": "bm25",
            "topk": 3,
            "source_turn": 2,
            "support_titles": ["Gold A", "Gold B"],
            "policy_tag": "base",
            "policy_seed": 13,
            "prior_turns": [],
        },
        "prefix": {
            "records": [
                {"turn": 1, "query": "Gold A creator", "observed_titles": ["Creator X"]}
            ]
        },
        "candidates": [
            candidate(
                "factual", "Creator X death date", ["Gold A"],
                style="factual", origin="factual", support_tqe=-0.1,
            ),
            candidate(
                "semantic", "When did Creator X die?", ["Gold A"],
                style="semantic", support_tqe=0.1,
            ),
            candidate(
                "lexical", "Creator X biography", ["Other"],
                style="lexical", gain=0.0, final_recall=0.0, support_tqe=-0.2,
            ),
        ],
    }


CFG = {
    "reward": {
        "support": 1.0,
        "answer_f1": 0.5,
        "immediate_gain": 0.25,
        "search_cost": 0.02,
        "protocol_cost": 1.0,
    },
    "alias_audit": {"behavior_signature": "ranked-transition"},
    "credit_granularity": {"behavior_signature": "gold-transition"},
    "interface_audit": {
        "maximum_prefix_titles": 4,
        "maximum_menu_queries": 8,
        "relation_tokens": 5,
    },
    "equivalence_predictor": {"label_signature": "ranked-transition"},
}


class InterfaceCausalityTests(unittest.TestCase):
    def test_behavior_signature_clusters_retrieval_aliases(self) -> None:
        payload = result()
        classes = group_candidates(
            payload["state"], payload["candidates"], mode="ranked-transition"
        )
        self.assertEqual(sorted(map(len, classes)), [1, 2])
        self.assertEqual(
            behavior_signature(payload["candidates"][0], payload["state"], mode="ranked-transition"),
            behavior_signature(payload["candidates"][1], payload["state"], mode="ranked-transition"),
        )

    def test_surface_alias_injection_changes_surface_but_not_quotient_credit(self) -> None:
        rows = state_alias_rows(result(), CFG, multiplicities=[1, 8])
        self.assertEqual(len(rows), 2)
        max_row = rows[-1]
        self.assertGreater(max_row["surface_class_advantage_drift"], 0.0)
        self.assertEqual(max_row["quotient_class_advantage_drift"], 0.0)

    def test_credit_granularity_detects_action_class_disagreement(self) -> None:
        rows = state_credit_rows(result(), CFG)
        factual = next(row for row in rows if row["factual"] == 1)
        self.assertLess(factual["query_tqe"], 0.0)
        self.assertAlmostEqual(factual["factual_class_tqe"], 0.0)

    def test_menu_uses_prefix_entity_and_relation(self) -> None:
        payload = result()
        self.assertEqual(prefix_titles(payload), ["Creator X"])
        queries = menu_queries(payload, CFG)
        self.assertTrue(any("Creator X" in row["query"] for row in queries))
        self.assertTrue(any(row["style"] == "menu-title-relation" for row in queries))

    def test_interface_rows_compare_free_menu_and_hybrid(self) -> None:
        class FakeClient:
            def search(self, query: str, topk: int):
                if "Creator X" in query and "death" in query:
                    return [{"title": "Gold A"}, {"title": "Gold B"}][:topk]
                if "Creator X" in query:
                    return [{"title": "Gold A"}, {"title": "Other"}][:topk]
                return [{"title": "Other"}][:topk]

        rows, summary = state_interface_rows(
            result(), CFG, {"bm25": FakeClient(), "e5": FakeClient()}
        )
        self.assertEqual({row["interface"] for row in rows}, {"free-form", "finite-menu", "hybrid"})
        self.assertIn("menu_covers_free_best_behavior", summary)
        hybrid = next(row for row in rows if row["interface"] == "hybrid")
        menu = next(row for row in rows if row["interface"] == "finite-menu")
        self.assertGreaterEqual(hybrid["oracle_immediate_gain"], menu["oracle_immediate_gain"])

    def test_predictor_examples_and_question_hash_split(self) -> None:
        rows = pair_rows([result()], CFG)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row["label"] for row in rows), 1)
        train, test = split_rows(rows, seed=1, train_ratio=0.5)
        self.assertIn(len(train), (0, 3))
        self.assertEqual(len(train) + len(test), 3)
        features = feature_sets(list(rows[0]))
        self.assertIn("semantic-only", features)
        self.assertIn("response-conditioned", features)

    def test_metrics_are_finite(self) -> None:
        metrics = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4])
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        advantages = normalize_advantages([0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(advantages.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()

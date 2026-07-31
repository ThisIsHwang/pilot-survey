from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stackpilot.query_equivalence_common import (
    build_equivalence_state,
    class_is_nontrivial,
    group_equivalence_classes,
)
from stackpilot.query_equivalence_plan import VARIANTS, _targets_for_variant
from stackpilot.query_equivalence_report import paired_effect_rows, state_metrics
from stackpilot.query_equivalence_scheduler import completed


def candidate(
    candidate_id: str,
    query: str,
    *,
    style: str,
    origin: str = "alternative",
    immediate_titles: list[str] | None = None,
    final_titles: list[str] | None = None,
    immediate_gain: float = 0.5,
    final_recall: float = 1.0,
    answer_em: float = 1.0,
) -> dict:
    immediate_titles = immediate_titles or ["Gold A"]
    final_titles = final_titles or ["Gold A", "Gold B"]
    return {
        "candidate_id": candidate_id,
        "query": query,
        "style": style,
        "origin": origin,
        "intervention_observed_titles": immediate_titles,
        "immediate_support_gain": immediate_gain,
        "final_support_recall": final_recall,
        "answer_em": answer_em,
        "answer_f1": answer_em,
        "total_search_count": 2,
        "suffix_search_count": 1,
        "protocol_failure": 0,
        "support_tqe": 0.0,
        "composite_tqe": 0.0,
        "branch_turns": [
            {
                "observed_titles": final_titles,
                "support_recall": final_recall,
                "evidence_gain": immediate_gain,
            }
        ],
    }


def config() -> dict:
    return {
        "equivalence": {
            "epsilon": 1e-8,
            "minimum_class_size": 2,
            "include_answer_in_signature": True,
            "maximum_nontrivial_query_jaccard": 0.8,
            "require_nontrivial_class": True,
            "require_factual_in_best_class": True,
        }
    }


def result_payload() -> dict:
    return {
        "run_signature": "run",
        "state_signature": "state-sig",
        "state": {
            "state_id": "s1",
            "question_id": "q1",
            "question": "Who created Gold A and when?",
            "support_titles": ["Gold A", "Gold B"],
            "dataset": "hotpotqa",
            "backend": "bm25",
            "topk": 3,
            "source_turn": 2,
            "policy_tag": "base",
            "policy_seed": 13,
        },
        "prefix": {
            "records": [
                {"turn": 1, "query": "Gold A creator", "observed_titles": ["Other"]}
            ]
        },
        "candidates": [
            candidate(
                "factual",
                "Gold A creator date",
                style="factual",
                origin="factual",
            ),
            candidate(
                "semantic",
                "Who made Gold A and on which date?",
                style="semantic",
            ),
            candidate(
                "other",
                "unrelated zero result",
                style="lexical",
                immediate_titles=[],
                final_titles=[],
                immediate_gain=0.0,
                final_recall=0.0,
                answer_em=0.0,
            ),
        ],
    }


class QueryEquivalenceTests(unittest.TestCase):
    def test_distinct_queries_form_one_functional_class(self) -> None:
        state = build_equivalence_state(result_payload(), cfg=config())
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["best_class_size"], 2)
        self.assertEqual(state["factual_in_best_class"], 1)
        self.assertEqual(state["nontrivial_best_class"], 1)
        self.assertLessEqual(state["best_class_min_query_jaccard"], 0.8)

    def test_answer_mismatch_breaks_strict_equivalence(self) -> None:
        rows = [
            {
                "candidate_id": "a",
                "immediate_support_set": ["gold a"],
                "final_support_set": ["gold a", "gold b"],
                "answer_em": 1.0,
                "final_support_recall": 1.0,
                "total_search_count": 2,
            },
            {
                "candidate_id": "b",
                "immediate_support_set": ["gold a"],
                "final_support_set": ["gold a", "gold b"],
                "answer_em": 0.0,
                "final_support_recall": 1.0,
                "total_search_count": 2,
            },
        ]
        classes = group_equivalence_classes(rows, include_answer=True)
        self.assertEqual(sorted(len(value) for value in classes), [1, 1])

    def test_nontrivial_class_requires_distinct_wording(self) -> None:
        rows = [
            {"query": "Gold A creator date"},
            {"query": "Gold A creator date"},
        ]
        self.assertFalse(class_is_nontrivial(rows, maximum_query_jaccard=0.8))

    def test_all_variants_preserve_one_unit_of_state_credit(self) -> None:
        state = build_equivalence_state(result_payload(), cfg=config())
        assert state is not None
        for variant in VARIANTS:
            targets = _targets_for_variant(state, variant=variant, seed=13)
            self.assertAlmostEqual(sum(float(row["weight"]) for row in targets), 1.0)
        self.assertEqual(len(_targets_for_variant(state, variant="first-exposure", seed=13)), 1)
        self.assertEqual(len(_targets_for_variant(state, variant="equivalence-pool", seed=13)), 2)

    def test_report_constructs_class_metrics_and_paired_effect(self) -> None:
        rows = []
        for variant, gains in {
            "first-exposure": [0.1, 0.0, -0.1],
            "random-member": [0.08, 0.01, -0.1],
            "equivalence-pool": [0.2, 0.15, -0.05],
            "all-direct-pool": [0.18, 0.12, 0.0],
        }.items():
            for index, gain in enumerate(gains):
                rows.append(
                    {
                        "direction": "bm25-to-e5",
                        "seed": 13,
                        "state_id": "state",
                        "variant": variant,
                        "target_id": f"t{index}",
                        "dataset": "hotpotqa",
                        "backend": "e5",
                        "best_class_member": int(index < 2),
                        "direct": 1,
                        "baseline_nll": 1.0 + 0.1 * index,
                        "adapted_nll": 1.0 + 0.1 * index - gain,
                        "heldout_gain": gain,
                    }
                )
        metrics = state_metrics(pd.DataFrame(rows), 1e-8)
        self.assertEqual(len(metrics), 4)
        effects = paired_effect_rows(
            metrics,
            left="equivalence-pool",
            right="first-exposure",
            value="class_mean_gain",
        )
        self.assertAlmostEqual(float(effects.iloc[0]["effect"]), 0.125)

    def test_scheduler_completion_requires_matching_signature(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            job = {"output_dir": str(root), "job_signature": "expected"}
            self.assertFalse(completed(job))
            (root / "metrics.json").write_text('{"job_signature":"other"}')
            self.assertFalse(completed(job))
            (root / "metrics.json").write_text('{"job_signature":"expected"}')
            self.assertTrue(completed(job))


if __name__ == "__main__":
    unittest.main()

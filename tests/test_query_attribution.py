from __future__ import annotations

import unittest

import pandas as pd

from stackpilot.query_attribution_common import candidate_key, class_members, query_jaccard, relative_imbalance, select_targets
from stackpilot.query_attribution_plan import training_groups
from stackpilot.query_attribution_report import paired_effects, state_metrics


def candidate(candidate_id: str, query: str, *, factual: int = 0, direct: int = 1, immediate: tuple[str, ...] = ("gold-a",), final: tuple[str, ...] = ("gold-a", "gold-b"), answer_em: float = 1.0, style: str = "semantic") -> dict:
    return {"candidate_id": candidate_id, "query": query, "style": style, "origin": "factual" if factual else "alternative", "factual": factual, "direct": direct, "protocol_failure": 0, "immediate_support_set": list(immediate), "final_support_set": list(final), "final_support_recall": len(final) / 2, "answer_em": answer_em, "total_search_count": 2}


def state() -> dict:
    return {"state_id": "s1", "question_id": "q1", "dataset": "hotpotqa", "backend": "bm25", "prompt": "Question and history", "candidates": [candidate("f", "Gold A creator date", factual=1, style="factual"), candidate("eq", "Who created Gold A and when?", style="semantic"), candidate("imm", "Gold A creator biography", immediate=("gold-a",), final=("gold-a",), answer_em=0.0, style="lexical"), candidate("fin", "Gold B date linked to Gold A", immediate=("gold-b",), final=("gold-a", "gold-b"), answer_em=1.0, style="entity"), candidate("direct", "Gold B direct evidence", immediate=("gold-b",), final=("gold-b",), answer_em=0.0, style="lexical"), candidate("zero", "unrelated query", direct=0, immediate=(), final=(), answer_em=0.0, style="semantic")]}


class QueryAttributionTests(unittest.TestCase):
    def test_class_definitions_differ(self) -> None:
        row = state(); self.assertEqual(len(class_members(row, "strict")), 2); self.assertEqual(len(class_members(row, "immediate")), 3); self.assertEqual(len(class_members(row, "final")), 3); self.assertNotEqual(candidate_key(row["candidates"][0], "strict"), candidate_key(row["candidates"][2], "strict"))

    def test_all_target_selectors_keep_two_sequences(self) -> None:
        row = state(); selectors = ["factual_replicated", "strict", "random_outside_strict", "diversity_matched_outside_strict", "direct_outside_strict", "immediate_only", "final_only"]
        for selector in selectors:
            self.assertEqual(len(select_targets(row, selector=selector, seed=13, maximum_random_query_jaccard=0.95)), 2, selector)
        repeated = select_targets(row, selector="factual_replicated", seed=13, maximum_random_query_jaccard=0.95); self.assertEqual(repeated[0]["query"], repeated[1]["query"])

    def test_state_credit_is_normalized(self) -> None:
        spec = {"selector": "strict", "variant": "strict-uniform", "experiment_id": "EXP-016", "family": "attribution-controls", "objective": "uniform"}; group = training_groups([state()], spec=spec, seed=13, cfg={"selection": {"maximum_random_query_jaccard": 0.95}})[0]; self.assertAlmostEqual(sum(target["weight"] for target in group["targets"]), 1.0)

    def test_diversity_distance_is_defined(self) -> None:
        self.assertLess(query_jaccard("Gold A creator date", "Who made Gold A and when"), 1.0); self.assertEqual(relative_imbalance([10, 12, 11]), 0.2)

    def test_report_uses_final_nll_dispersion(self) -> None:
        rows = []
        for variant, gains in {"strict-uniform": [0.8, 0.2, 0.0], "random-k": [0.4, 0.1, 0.0]}.items():
            for index, gain in enumerate(gains):
                rows.append({"direction": "bm25-to-e5", "variant": variant, "seed": 13, "eval_scope": "cross", "state_id": "s", "question_id": "q", "dataset": "hotpotqa", "backend": "e5", "target_id": f"t{index}", "best_class_member": int(index < 2), "direct": 1, "factual": int(index == 0), "synthetic": int(index != 0), "strict_class_size": 2, "strict_min_jaccard": 0.3, "baseline_nll": 2.0 if index == 0 else 1.0, "adapted_nll": (2.0 if index == 0 else 1.0) - gain, "heldout_gain": gain})
        metrics = state_metrics(pd.DataFrame(rows), 1e-8); strict = metrics[metrics["variant"] == "strict-uniform"].iloc[0]; self.assertAlmostEqual(strict["baseline_class_std"], 0.5); self.assertAlmostEqual(strict["adapted_class_std"], 0.2); self.assertAlmostEqual(strict["dispersion_reduction"], 0.3); self.assertGreater(float(paired_effects(metrics, "strict-uniform", "random-k", "class_mean_gain").iloc[0]["effect"]), 0.0)


if __name__ == "__main__":
    unittest.main()

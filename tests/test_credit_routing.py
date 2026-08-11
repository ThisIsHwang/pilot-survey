from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hard_rq0.patch_searchr1_credit_routing import patch
from stackpilot.credit_routing_common import (
    aggregate_document_utility,
    apply_standardizer,
    feature_rows,
    fit_ridge,
    fit_standardizer,
    fixed_budget_contexts,
    matched_swap_utilities,
    matrix_from_feature_rows,
    paired_context_indices,
    predict_ridge,
    selection_indices,
)
from stackpilot.credit_routing_eval_split import question_identifier
from stackpilot.credit_routing_proxy import attach_metadata
from stackpilot.credit_routing_runtime import normalized_group_shaping


class CreditRoutingTests(unittest.TestCase):
    def test_budgeted_ctu_contexts_have_equal_cardinality(self) -> None:
        for index in range(8):
            with_document, without_document = paired_context_indices(index, 8, 3)
            self.assertEqual(len(with_document), 3)
            self.assertEqual(len(without_document), 3)
            self.assertIn(index, with_document)
            self.assertNotIn(index, without_document)
        self.assertEqual(paired_context_indices(0, 8, 3), ((0, 1, 2), (1, 2, 3)))
        self.assertEqual(paired_context_indices(6, 8, 3), ((0, 1, 6), (0, 1, 2)))

    def test_exact_matched_swaps_recover_document_order(self) -> None:
        weights = np.asarray([0.9, 0.4, 0.2, 0.0, -0.1, -0.2, -0.4, -0.8])
        contexts = fixed_budget_contexts(8, 3)
        self.assertEqual(len(contexts), 56)
        values = {context: float(weights[list(context)].sum()) for context in contexts}
        utilities, counts = matched_swap_utilities(
            values, candidate_count=8, keep_k=3
        )
        self.assertTrue(np.all(counts == 105))
        self.assertAlmostEqual(float(utilities.sum()), 0.0, places=8)
        self.assertEqual(list(np.argsort(-utilities)), list(np.argsort(-weights)))

    def test_shared_utility_selects_without_changing_action_aggregate(self) -> None:
        scores = [0.1, 0.9, 0.2, 0.8]
        self.assertEqual(selection_indices(scores, 2, mode="rank"), [0, 1])
        self.assertEqual(selection_indices(scores, 2, mode="utility"), [1, 3])
        expected = aggregate_document_utility(scores, k=2, mode="mean-topk")
        self.assertAlmostEqual(expected, 0.85)

    def test_ridge_round_trip(self) -> None:
        items = [
            {"score": 1.0, "document": {"contents": "Alpha\nalpha evidence"}},
            {"score": 0.2, "document": {"contents": "Beta\nbeta noise"}},
            {"score": 0.1, "document": {"contents": "Gamma\ngamma noise"}},
        ]
        rows = feature_rows("alpha evidence", items, "bm25")
        matrix = matrix_from_feature_rows(rows)
        mean, scale = fit_standardizer(matrix)
        standardized = apply_standardizer(matrix, mean, scale)
        weights = fit_ridge(standardized, np.array([1.0, 0.0, -0.1]), l2=0.1)
        predictions = predict_ridge(standardized, weights)
        self.assertEqual(predictions.shape, (3,))
        self.assertGreater(predictions[0], predictions[1])

    def test_endpoint_question_identifier_supports_hard_rq0_rows(self) -> None:
        self.assertEqual(question_identifier({"id": "dataset:42"}), "dataset:42")
        self.assertEqual(
            question_identifier({"extra_info": {"question_id": "dataset:43"}}),
            "dataset:43",
        )

    def test_group_shaping_is_centered_and_clipped(self) -> None:
        shaping, rows = normalized_group_shaping(
            [0.0, 1.0, 2.0, 10.0],
            ["a", "a", "b", "b"],
            coefficient=0.5,
            clip=1.0,
        )
        self.assertAlmostEqual(float(shaping[:2].mean()), 0.0, places=7)
        self.assertAlmostEqual(float(shaping[2:].mean()), 0.0, places=7)
        self.assertLessEqual(float(np.abs(shaping).max()), 0.5)
        self.assertEqual(len(rows), 2)

    def test_proxy_metadata_preserves_result(self) -> None:
        source = {"score": 0.5, "document": {"contents": "Doc\nText"}}
        result = attach_metadata(
            source,
            predicted_utility=0.4,
            action_utility=0.3,
            upstream_rank=4,
            observation_route="utility",
        )
        self.assertEqual(result["document"], source["document"])
        self.assertEqual(result["stackpilot_upstream_rank"], 4)
        self.assertAlmostEqual(result["stackpilot_action_utility"], 0.3)
        self.assertNotIn("stackpilot_action_utility", source)

    def test_searchr1_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "search_r1" / "llm_agent" / "generation.py"
            trainer = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            generation.parent.mkdir(parents=True)
            trainer.parent.mkdir(parents=True)
            generation.write_text(
                "# STACKPILOT_OBSERVATION_GEOMETRY_V1\n"
                "# STACKPILOT_BEHAVIOR_QUOTIENT_GENERATION_V1\n"
                "        self._stackpilot_search_title_batches = [\n"
                "            [] for _ in range(protocol_batch_size)\n"
                "        ]\n\n"
                "        self._stackpilot_last_search_titles = []\n"
                "        self._stackpilot_last_observed_titles = []\n"
                "        for result_index, retrieval_result in enumerate(results):\n"
                "            titles = []\n"
                "            self._stackpilot_last_search_titles.append(titles)\n"
                "        return [self._passages2string(result) for result in results]\n"
                "            search_observed_title_batches = getattr(\n"
                "                self, '_stackpilot_last_observed_titles', None\n"
                "            )\n"
                "                or not isinstance(search_observed_title_batches, list)\n"
                "                or len(search_observed_title_batches) != len(search_results)\n"
                "            search_observed_title_batches = [\n"
                "                list(titles) for titles in search_observed_title_batches\n"
                "            ]\n"
                "            search_observed_title_batches = [[] for _ in search_results]\n"
                "                    current_search_titles = list(search_title_batches.pop(0))\n"
                "                    self._stackpilot_search_title_batches[i].append(\n"
                "                        current_search_titles\n"
                "                    )\n"
                "                    search_observed_title_batches.pop(0)\n"
                "                    next_obs.append('')\n"
                "        assert len(search_observed_title_batches) == 0\n"
                "            'stackpilot_search_title_batches': self._stackpilot_search_title_batches,\n",
                encoding="utf-8",
            )
            trainer.write_text(
                "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1\n"
                "from stackpilot.behavior_quotient_runtime import (\n"
                "    append_behavior_telemetry,\n"
                "    compute_behavior_advantages,\n"
                "    select_behavior_rows,\n"
                ")\n"
                "        query_batches = data.non_tensor_batch.get('stackpilot_search_queries')\n"
                "        signature_array = np.empty(len(bq_signatures), dtype=object)\n",
                encoding="utf-8",
            )
            patch(root)
            first_generation = generation.read_text(encoding="utf-8")
            first_trainer = trainer.read_text(encoding="utf-8")
            patch(root)
            self.assertEqual(first_generation, generation.read_text(encoding="utf-8"))
            self.assertEqual(first_trainer, trainer.read_text(encoding="utf-8"))
            self.assertIn("stackpilot_search_action_utilities", first_generation)
            self.assertIn("apply_action_utility_shaping", first_trainer)


if __name__ == "__main__":
    unittest.main()

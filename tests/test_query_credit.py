from __future__ import annotations

import json
import os
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stackpilot.experiment_registry import experiment_by_id, load_registry
from stackpilot.query_credit_common import (
    LinearUtilityModel,
    aggregate_document_credit,
    best_replacement_gap,
    build_search_span_ids,
    document_features,
    fit_ridge_ensemble,
    question_split,
)
from stackpilot.query_credit_modeling import signal_values
from stackpilot.query_credit_runtime import _normalize_turn_scores


class QueryCreditCommonTest(unittest.TestCase):
    def test_best_replacement_gap_marks_tied_best_as_replaceable(self) -> None:
        self.assertEqual(best_replacement_gap([1.0, 1.0, 0.0]), [0.0, 0.0, -1.0])

    def test_document_credit_aggregations(self) -> None:
        values = [0.6, -0.2, 0.1]
        self.assertAlmostEqual(aggregate_document_credit(values, "positive-sum"), 0.7)
        self.assertAlmostEqual(aggregate_document_credit(values, "signed-sum"), 0.5)
        self.assertAlmostEqual(aggregate_document_credit(values, "max"), 0.6)
        self.assertAlmostEqual(aggregate_document_credit(values, "top2-sum"), 0.7)

    def test_search_span_ids_find_multiple_actions(self) -> None:
        open_ids = [10, 11]
        close_ids = [12, 13]
        responses = np.asarray([[0, 10, 11, 5, 12, 13, 9, 10, 11, 7, 8, 12, 13, 0]])
        mask = build_search_span_ids(
            responses,
            pad_token_id=0,
            open_ids=open_ids,
            close_ids=close_ids,
        )
        self.assertEqual(mask.tolist(), [[0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0]])

    def test_alias_normalization_conserves_class_mass(self) -> None:
        raw = [[2.0], [2.0], [0.0]]
        index = ["q", "q", "q"]
        titles = [[['A']], [['A']], [['B']]]
        normalized = _normalize_turn_scores(
            raw,
            index=index,
            title_batches=titles,
            mode="alias-normalized",
            seed=1,
        )
        class_a = normalized[0][0] + normalized[1][0]
        class_b = normalized[2][0]
        self.assertAlmostEqual(class_a, 1.0, places=5)
        self.assertAlmostEqual(class_b, -1.0, places=5)

    def test_signal_values_are_state_normalized(self) -> None:
        rows = [
            {"state_id": "s", "query_action_advantage": 1.0, "document_credit": {"positive-sum": 4.0}, "alias_class_size": 2, "full_reward": 1.0},
            {"state_id": "s", "query_action_advantage": 0.0, "document_credit": {"positive-sum": 2.0}, "alias_class_size": 1, "full_reward": 0.0},
        ]
        values = signal_values(rows, "doc-positive-sum")
        self.assertAlmostEqual(sum(values), 0.0, places=6)
        alias = signal_values(rows, "doc-alias-normalized")
        self.assertAlmostEqual(sum(alias), 0.0, places=6)

    def test_ridge_model_round_trip(self) -> None:
        rows = [
            {"query": "alpha", "document_title": "alpha", "document_text": "alpha evidence", "document_rank": 1, "retriever_score": 1.0, "retriever_score_z": 1.0, "backend": "bm25"},
            {"query": "alpha", "document_title": "beta", "document_text": "noise", "document_rank": 2, "retriever_score": 0.0, "retriever_score_z": -1.0, "backend": "e5"},
        ]
        matrix = np.stack([document_features(row) for row in rows])
        model = fit_ridge_ensemble(matrix, np.asarray([1.0, 0.0]), alphas=[0.1, 1.0])
        restored = LinearUtilityModel.from_json(model.to_json())
        self.assertAlmostEqual(model.predict_row(rows[0]), restored.predict_row(rows[0]))

    def test_question_split_is_stable(self) -> None:
        self.assertEqual(question_split("q1"), question_split("q1"))


class RegistryOverlayTest(unittest.TestCase):
    def test_overlay_registers_query_credit_experiments(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        overlay = repository_root / "experiments" / "registry.query_credit.json"
        previous = os.environ.get("STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "registry.json"
            base.write_text(json.dumps({
                "schema": 1,
                "experiments": [{
                    "id": "EXP-034",
                    "slug": "base",
                    "status": "planned",
                    "question": "base",
                    "entrypoint": "base.sh",
                    "parent": None,
                }],
            }), encoding="utf-8")
            os.environ["STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY"] = str(overlay)
            try:
                with patch("stackpilot.experiment_registry.default_registry_path", return_value=base):
                    registry = load_registry()
                self.assertEqual(experiment_by_id(registry, "EXP-056")["parent"], "EXP-055")
            finally:
                if previous is None:
                    os.environ.pop("STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY", None)
                else:
                    os.environ["STACKPILOT_EXPERIMENT_REGISTRY_OVERLAY"] = previous


class QueryCreditPatchIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(Path("upstream/Search-R1/search_r1/llm_agent/generation.py").is_file(), "Search-R1 fixture is unavailable")
    def test_query_credit_patch_applies_after_existing_stack(self) -> None:
        from hard_rq0.patch_searchr1_action_protocol import patch as patch_action
        from hard_rq0.patch_searchr1_observation_geometry import patch as patch_observation
        from hard_rq0.patch_searchr1_reward_protocol import patch as patch_reward
        from hard_rq0.patch_searchr1_behavior_quotient import patch as patch_behavior
        from hard_rq0.patch_searchr1_query_credit import patch as patch_query_credit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Search-R1"
            shutil.copytree("upstream/Search-R1", root)
            patch_action(root)
            patch_observation(root)
            patch_reward(root)
            patch_behavior(root)
            patch_query_credit(root)
            patch_query_credit(root)
            generation = (root / "search_r1" / "llm_agent" / "generation.py").read_text(encoding="utf-8")
            trainer = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")
            self.assertIn("stackpilot_search_span_ids", generation)
            self.assertIn("apply_query_credit_bonus", trainer)


if __name__ == "__main__":
    unittest.main()

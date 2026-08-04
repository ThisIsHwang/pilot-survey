from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hard_rq0.patch_searchr1_response_feedback import patch as patch_feedback
from stackpilot.adaptive_interface_router import FEATURES, fit_logistic, predict, split_mask
from stackpilot.document_action_ctu_closure import state_rows
from stackpilot.paired_retriever_grid import rrf_fuse
from stackpilot.response_feedback_runtime import (
    feedback_text,
    feedback_title_map,
    phase_indices,
    stable_prompt_key,
)


class ResponseFeedbackTests(unittest.TestCase):
    def test_phase_indices_split_each_prompt_group(self) -> None:
        keys = ["a"] * 8 + ["b"] * 8
        first, second = phase_indices(keys, 4)
        self.assertEqual(first, [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(second, [4, 5, 6, 7, 12, 13, 14, 15])

    def test_prompt_key_ignores_left_padding(self) -> None:
        self.assertEqual(stable_prompt_key([0, 0, 3, 4], 0), stable_prompt_key([3, 4], 0))

    def test_feedback_uses_visible_titles_only_and_deduplicates(self) -> None:
        titles = feedback_title_map(
            group_keys=["q", "q"],
            first_indices=[0, 1],
            first_title_batches=[[['Doc A', 'Doc B']], [['doc a', 'Doc C']]],
            maximum_titles=3,
        )
        self.assertEqual(titles["q"], ["Doc A", "Doc B", "Doc C"])
        text = feedback_text(titles["q"], maximum_chars=500)
        self.assertIn("Doc A", text)
        self.assertIn("observations, not gold labels", text)
        self.assertNotIn("support_titles", text)

    def test_rrf_fusion_rewards_cross_retriever_agreement(self) -> None:
        left = [
            {"title": "A", "id": "a"},
            {"title": "B", "id": "b"},
        ]
        right = [
            {"title": "B", "id": "b"},
            {"title": "C", "id": "c"},
        ]
        fused = rrf_fuse(left, right, rrf_k=60, topk=3)
        self.assertEqual(fused[0]["title"], "B")
        self.assertEqual(len(fused), 3)

    def test_router_split_is_question_stable_and_logistic_learns(self) -> None:
        rows = []
        for index in range(80):
            row = {feature: 0.0 for feature in FEATURES}
            row["question_id"] = f"q-{index // 2}"
            row["label_free"] = int(index % 4 >= 2)
            row["menu_size"] = float(row["label_free"])
            rows.append(row)
        frame = pd.DataFrame(rows)
        mask = split_mask(frame, 0.7)
        for _, group in frame.assign(mask=mask).groupby("question_id"):
            self.assertEqual(group["mask"].nunique(), 1)
        cfg = {"router": {"learning_rate": 0.1, "l2": 0.0, "iterations": 600}}
        model = fit_logistic(frame, cfg)
        probability = predict(model, frame)
        accuracy = np.mean((probability >= 0.5) == frame["label_free"].to_numpy())
        self.assertGreater(accuracy, 0.95)

    def test_ctu_closure_joins_document_query_and_class_credit(self) -> None:
        result = {
            "state": {
                "state_id": "s",
                "question_id": "q",
                "question": "question",
                "dataset": "hotpotqa",
                "backend": "bm25",
                "topk": 3,
                "support_titles": ["Gold"],
            },
            "candidates": [
                {
                    "candidate_id": "f",
                    "origin": "factual",
                    "style": "factual",
                    "query": "query one",
                    "intervention_observed_titles": ["A"],
                    "immediate_support_gain": 0.0,
                    "final_support_recall": 0.0,
                    "answer_f1": 0.0,
                    "support_tqe": -0.1,
                    "composite_tqe": -0.1,
                    "total_search_count": 1,
                    "protocol_failure": 0,
                    "branch_turns": [{"observed_titles": ["A"]}],
                },
                {
                    "candidate_id": "a",
                    "style": "semantic",
                    "query": "query two",
                    "intervention_observed_titles": ["A"],
                    "immediate_support_gain": 0.0,
                    "final_support_recall": 0.0,
                    "answer_f1": 0.0,
                    "support_tqe": 0.1,
                    "composite_tqe": 0.1,
                    "total_search_count": 1,
                    "protocol_failure": 0,
                    "branch_turns": [{"observed_titles": ["A"]}],
                },
            ],
        }
        cfg = {
            "ctu_closure": {"minimum_positive_document_ctu": 0.001},
            "reward": {
                "support": 1.0,
                "answer_f1": 0.5,
                "immediate_gain": 0.25,
                "search_cost": 0.02,
                "protocol_cost": 1.0,
            },
        }
        frame = state_rows(
            cfg,
            [result],
            [
                {
                    "state_id": "s",
                    "document_rank": 1,
                    "document_title": "A",
                    "document_ctu": 0.5,
                    "support_ctu": 0.5,
                    "answer_ctu": 0.0,
                    "search_ctu": 0.0,
                }
            ],
        )
        self.assertEqual(int(frame.iloc[0]["document_query_disagreement"]), 1)
        self.assertAlmostEqual(float(frame.iloc[0]["behavior_class_tqe"]), 0.0)

    def test_response_feedback_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            trainer.parent.mkdir(parents=True)
            trainer.write_text(
                "import os\n"
                "from stackpilot.behavior_quotient_runtime import (\n"
                "    append_behavior_telemetry,\n"
                "    compute_behavior_advantages,\n"
                "    select_behavior_rows,\n"
                ")\n"
                "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1\n"
                "                        final_gen_batch_output = generation_manager.run_llm_loop(\n"
                "                            gen_batch=test_gen_batch,\n"
                "                            initial_input_ids=first_input_ids,\n"
                "                        )\n"
                "                        final_gen_batch_output = generation_manager.run_llm_loop(\n"
                "                            gen_batch=gen_batch,\n"
                "                            initial_input_ids=first_input_ids,\n"
                "                        )\n",
                encoding="utf-8",
            )
            patch_feedback(root)
            first = trainer.read_text(encoding="utf-8")
            patch_feedback(root)
            self.assertEqual(first, trainer.read_text(encoding="utf-8"))
            self.assertIn("run_grouped_feedback_rollouts", first)


if __name__ == "__main__":
    unittest.main()

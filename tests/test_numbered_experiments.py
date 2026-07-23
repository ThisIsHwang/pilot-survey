from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from hard_rq0.patch_searchr1_evidence_reward import patch as patch_evidence_reward
from hard_rq0.patch_searchr1_mixed import patch as patch_mixed_routing
from stackpilot.hybrid_rrf_server import fuse
from stackpilot.mixed_retriever_server import create_app
from stackpilot.numbered_experiment_report import (
    evidence_reward_value,
    metadata_value,
    mixed_regret,
)
from stackpilot.prepare_mixed_data import add_marker, duplicate_row


def write_main_ppo(root: Path, *, reward_block: bool) -> Path:
    reward = ""
    if reward_block:
        reward = '''
class RewardManager():
    def score(self, data_item, sequences_str, ground_truth, reward_tensor, i, valid_response_length):
        compute_score_fn = lambda **kwargs: 1.0
        if True:
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
'''
    source = f'''import re
import numpy as np
{reward}
import ray

def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={{'env_vars': {{'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}}})
'''
    target = root / "verl" / "trainer" / "main_ppo.py"
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    return target


class NumberedExperimentTests(unittest.TestCase):
    def test_backend_marker_is_explicit_and_does_not_mutate_source(self) -> None:
        prompt = [{"role": "user", "content": "Question: test"}]
        marked = add_marker(prompt, "bm25")
        self.assertEqual(prompt[0]["content"], "Question: test")
        self.assertTrue(
            marked[0]["content"].startswith(
                "<retrieval_environment>bm25</retrieval_environment>\n"
            )
        )
        row = duplicate_row({"prompt": prompt, "extra_info": {"x": 1}}, "e5")
        self.assertEqual(row["extra_info"]["routing_backend"], "e5")
        with self.assertRaises(ValueError):
            add_marker(prompt, "hybrid")

    def test_mixed_router_requires_ids_and_restores_batch_order(self) -> None:
        def fake_post(url: str, queries: list[str], topk: int, timeout: float):
            backend = "bm25" if "8101" in url else "e5"
            return [
                [
                    {
                        "document": {
                            "id": f"{backend}:{query}",
                            "contents": f'"{backend}"\n{query}',
                        },
                        "score": 1.0,
                    }
                ]
                for query in queries
            ]

        app = create_app(
            bm25_url="http://127.0.0.1:8101/retrieve",
            e5_url="http://127.0.0.1:8102/retrieve",
            default_topk=3,
            timeout=1.0,
            assignment_log=None,
        )
        with patch("stackpilot.mixed_retriever_server.post_batch", side_effect=fake_post):
            client = TestClient(app)
            missing = client.post("/retrieve", json={"queries": ["a"]})
            self.assertEqual(missing.status_code, 422)
            response = client.post(
                "/retrieve",
                json={
                    "queries": ["q0", "q1", "q2"],
                    "backend_ids": ["e5", "bm25", "e5"],
                    "topk": 1,
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["routes"], ["e5", "bm25", "e5"])
        identifiers = [result[0]["document"]["id"] for result in payload["result"]]
        self.assertEqual(identifiers, ["e5:q0", "bm25:q1", "e5:q2"])

    def test_rrf_merges_duplicate_documents(self) -> None:
        bm25 = [
            {"document": {"id": "a", "contents": "A"}, "score": 10.0},
            {"document": {"id": "b", "contents": "B"}, "score": 9.0},
        ]
        e5 = [
            {"document": {"id": "b", "contents": "B"}, "score": 0.9},
            {"document": {"id": "c", "contents": "C"}, "score": 0.8},
        ]
        result = fuse(bm25, e5, topk=2, rrf_constant=60.0)
        self.assertEqual(result[0]["document"]["id"], "b")
        self.assertEqual(result[0]["sources"], ["bm25", "e5"])
        self.assertEqual(len(result), 2)

    def test_mixed_patch_is_idempotent_and_compiles(self) -> None:
        source = '''import os
import re
from typing import List, Dict, Any, Tuple

class Manager:
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        return {}, {}

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
        return next_obs

    def batch_search(self, queries: List[str] = None) -> str:
        """Batchified search for queries."""
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        return str(retrieval_result)
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main_ppo = write_main_ppo(root, reward_block=False)
            target = root / "search_r1" / "llm_agent" / "generation.py"
            target.parent.mkdir(parents=True)
            target.write_text(source, encoding="utf-8")
            patch_mixed_routing(root)
            first = target.read_text(encoding="utf-8")
            first_main = main_ppo.read_text(encoding="utf-8")
            patch_mixed_routing(root)
            second = target.read_text(encoding="utf-8")
            self.assertEqual(first, second)
            self.assertIn("STACKPILOT_MIXED_ROUTING_V1", first)
            self.assertIn('payload["backend_ids"] = backend_ids', first)
            self.assertIn("STACKPILOT_EXPERIMENT_ENV_V1", first_main)
            compile(first, str(target), "exec")
            compile(first_main, str(main_ppo), "exec")

    def test_evidence_reward_patch_is_idempotent_and_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = write_main_ppo(root, reward_block=True)
            patch_evidence_reward(root)
            first = target.read_text(encoding="utf-8")
            patch_evidence_reward(root)
            second = target.read_text(encoding="utf-8")
            self.assertEqual(first, second)
            self.assertIn("STACKPILOT_EVIDENCE_REWARD_V1", first)
            self.assertIn("STACKPILOT_EXPERIMENT_ENV_V1", first)
            self.assertIn("EVIDENCE_REWARD_WEIGHT", first)
            self.assertIn(r"Doc\s+\d+\(Title:", first)
            compile(first, str(target), "exec")

    @staticmethod
    def result_frame(tag: str, seed: int, bm25: float, e5: float) -> pd.DataFrame:
        rows = []
        for backend, value in (("bm25", bm25), ("e5", e5)):
            rows.append(
                {
                    "subset": "all",
                    "policy_tag": tag,
                    "seed": seed,
                    "question_id": "q1",
                    "dataset": "musique",
                    "backend": backend,
                    "topk": 3,
                    "support_recall": value,
                }
            )
        return pd.DataFrame(rows)

    def test_numbered_report_effects_have_expected_sign(self) -> None:
        hard = pd.concat(
            [
                self.result_frame("bm25-specialist", 42, 0.8, 0.4),
                self.result_frame("e5-specialist", 42, 0.5, 0.9),
            ],
            ignore_index=True,
        )
        blind = self.result_frame("mixed-blind", 42, 0.6, 0.7)
        oracle = self.result_frame("mixed-backend-id", 42, 0.75, 0.85)
        evidence = pd.concat(
            [
                self.result_frame("evidence-bm25", 42, 0.9, 0.45),
                self.result_frame("evidence-e5", 42, 0.55, 0.95),
            ],
            ignore_index=True,
        )
        regret = mixed_regret(hard, blind, "support_recall")
        self.assertAlmostEqual(float(regret["mixed_regret"].mean()), 0.2)
        value = metadata_value(blind, oracle, "support_recall")
        self.assertAlmostEqual(float(value["metadata_value"].mean()), 0.15)
        reward = evidence_reward_value(hard, evidence, "support_recall")
        self.assertTrue((reward["evidence_reward_value"] > 0).all())

    def test_run_scripts_bind_numbers_to_variants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        exp003 = (root / "experiments" / "EXP-003" / "run.sh").read_text(encoding="utf-8")
        exp004 = (root / "experiments" / "EXP-004" / "run.sh").read_text(encoding="utf-8")
        exp005 = (root / "experiments" / "EXP-005" / "run.sh").read_text(encoding="utf-8")
        exp006 = (root / "experiments" / "EXP-006" / "run.sh").read_text(encoding="utf-8")
        self.assertIn("EXPERIMENT_ID=EXP-003", exp003)
        self.assertIn("VARIANT=blind", exp003)
        self.assertIn("INJECT_BACKEND_ID=1", exp004)
        self.assertIn("patch_searchr1_evidence_reward.py", exp005)
        self.assertIn('BACKENDS="hybrid"', exp006)


if __name__ == "__main__":
    unittest.main()

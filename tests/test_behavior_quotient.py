from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hard_rq0.patch_searchr1_behavior_quotient import patch
from stackpilot.behavior_quotient_common import ranked_transition
from stackpilot.behavior_quotient_fixed_budget import _quotient_indices
from stackpilot.behavior_signature_audit import partition, state_signature_metrics


def candidate(candidate_id: str, query: str, titles: list[str], reward: float = 0.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "query": query,
        "intervention_observed_titles": titles,
        "immediate_support_gain": reward,
        "final_support_recall": reward,
        "answer_f1": reward,
        "support_tqe": 0.0,
        "composite_tqe": 0.0,
        "total_search_count": 1,
        "protocol_failure": 0,
        "branch_turns": [{"observed_titles": titles}],
    }


class BehaviorQuotientTests(unittest.TestCase):
    def test_ranked_transition_ignores_surface_query(self) -> None:
        left = candidate("a", "alpha wording", ["Doc A", "Doc B"])
        right = candidate("b", "unrelated wording", ["Doc A", "Doc B"])
        reversed_result = candidate("c", "alpha wording", ["Doc B", "Doc A"])
        self.assertEqual(ranked_transition(left), ranked_transition(right))
        self.assertNotEqual(ranked_transition(left), ranked_transition(reversed_result))

    def test_quotient_selection_covers_classes_before_aliases(self) -> None:
        rows = [
            candidate("a1", "a one", ["A"]),
            candidate("a2", "a two", ["A"]),
            candidate("a3", "a three", ["A"]),
            candidate("b1", "b one", ["B"]),
        ]
        selected = _quotient_indices(rows, budget=2, seed=13, state_id="state")
        signatures = {ranked_transition(rows[index]) for index in selected}
        self.assertEqual(len(signatures), 2)

    def test_signature_audit_detects_unordered_false_merge(self) -> None:
        rows = [
            candidate("a", "a", ["A", "B"], 1.0),
            candidate("b", "b", ["B", "A"], 0.0),
            candidate("c", "c", ["C"], 0.0),
        ]
        exact = partition(rows, "exact-ranked")
        unordered = partition(rows, "unordered-set")
        self.assertNotEqual(exact[0], exact[1])
        self.assertEqual(unordered[0], unordered[1])
        metrics = state_signature_metrics(
            rows,
            mode="unordered-set",
            config={
                "reward": {
                    "support": 1.0,
                    "answer_f1": 0.0,
                    "immediate_gain": 0.0,
                    "search_cost": 0.0,
                    "protocol_cost": 0.0,
                }
            },
        )
        self.assertLess(metrics["pair_precision"], 1.0)

    def test_searchr1_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "search_r1" / "llm_agent" / "generation.py"
            trainer = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            generation.parent.mkdir(parents=True)
            trainer.parent.mkdir(parents=True)
            generation.write_text(
                "# STACKPILOT_STRICT_ACTION_PROTOCOL_V2\n"
                "        self._stackpilot_retrieved_titles = [\n"
                "            [] for _ in range(protocol_batch_size)\n"
                "        ]\n\n"
                "                    self._stackpilot_executed_search_counts[i] += 1\n"
                "                    self._stackpilot_retrieved_titles[i].extend(\n"
                "                        search_title_batches.pop(0)\n"
                "                    )\n"
                "            'stackpilot_retrieved_titles': self._stackpilot_retrieved_titles,\n",
                encoding="utf-8",
            )
            trainer.write_text(
                "import os\nimport numpy as np\n"
                "from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig\n"
                "        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,\n"
                "                                                                        eos_mask=response_mask,\n"
                "                                                                        index=index)\n"
                "        data.batch['advantages'] = advantages\n"
                "        data.batch['returns'] = returns\n"
                "                        batch = compute_advantage(batch,\n"
                "                                                  adv_estimator=self.config.algorithm.adv_estimator,\n"
                "                                                  gamma=self.config.algorithm.gamma,\n"
                "                                                  lam=self.config.algorithm.lam,\n"
                "                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)\n\n"
                "                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:\n"
                "                                batch, metrics = self._create_loss_mask(batch, metrics)\n"
                "                            actor_output = self.actor_rollout_wg.update_actor(batch)\n",
                encoding="utf-8",
            )
            patch(root)
            first_generation = generation.read_text(encoding="utf-8")
            first_trainer = trainer.read_text(encoding="utf-8")
            patch(root)
            self.assertEqual(first_generation, generation.read_text(encoding="utf-8"))
            self.assertEqual(first_trainer, trainer.read_text(encoding="utf-8"))
            self.assertIn("stackpilot_search_title_batches", first_generation)
            self.assertIn("compute_behavior_advantages", first_trainer)
            self.assertIn("stackpilot_bq_selected_mask", first_trainer)
            self.assertIn("update_actor(\n                                actor_batch", first_trainer)

    def test_fixed_k_selection_returns_explicit_row_mask(self) -> None:
        try:
            import torch
            from stackpilot.behavior_quotient_runtime import compute_behavior_advantages
        except ImportError:
            self.skipTest("torch is not installed in lightweight CI")
        rewards = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
        mask = torch.ones_like(rewards)
        _advantages, _returns, _metrics, _rows, _signatures, selected = (
            compute_behavior_advantages(
                token_level_rewards=rewards,
                eos_mask=mask,
                index=["q"] * 4,
                query_batches=[[str(i)] for i in range(4)],
                title_batches=[[[chr(65 + i)]] for i in range(4)],
                advantage_mode="surface",
                selection_mode="surface-random",
                update_per_prompt=2,
                seed=13,
            )
        )
        self.assertEqual(int(selected.sum().item()), 2)

    def test_surface_runtime_matches_standard_when_available(self) -> None:
        try:
            import torch
            from stackpilot.behavior_quotient_runtime import compute_behavior_advantages
        except ImportError:
            self.skipTest("torch is not installed in lightweight CI")
        rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float32)
        mask = torch.ones_like(rewards)
        advantages, _, _, _, _, selected = compute_behavior_advantages(
            token_level_rewards=rewards,
            eos_mask=mask,
            index=["q", "q"],
            query_batches=[["one"], ["two"]],
            title_batches=[[["A"]], [["B"]]],
            advantage_mode="surface",
            selection_mode="all",
        )
        expected = torch.tensor([[-0.7071067, -0.7071067], [0.7071067, 0.7071067]])
        self.assertTrue(torch.allclose(advantages, expected, atol=1e-5))
        self.assertTrue(bool(selected.all()))


if __name__ == "__main__":
    unittest.main()

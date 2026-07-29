from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stackpilot.trace_bank import add_group_features, build_transitions
from stackpilot.trace_common import (
    classify_episode,
    counterfactual_recovery_score,
    hierarchical_bootstrap_difference,
)
from stackpilot.trace_curricula import (
    match_by_marginals,
    paired_portable_pool,
    recovered_vs_deep_pools,
)
from stackpilot.trace_plan import _write_examples
from stackpilot.trace_scheduler import completed


def transition(
    transition_id: str,
    *,
    question_id: str,
    backend: str,
    episode_class: str,
    source_turn: int,
    search_count: int,
    evidence_gain: float,
    crs: float,
    portable: float = 0.0,
) -> dict:
    return {
        "transition_id": transition_id,
        "episode_id": f"ep-{transition_id}",
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "dataset": "toy",
        "policy_tag": "teacher",
        "policy_seed": 13,
        "backend": backend,
        "topk": 3,
        "view_id": f"{backend}:k3",
        "split": "train",
        "source_turn": source_turn,
        "prompt": f"prompt {transition_id}",
        "target": f"query {transition_id}",
        "evidence_gain": evidence_gain,
        "positive_gain": int(evidence_gain > 0),
        "episode_class": episode_class,
        "search_count": search_count,
        "turn1_recall": 0.0,
        "final_recall": evidence_gain,
        "total_recovery": evidence_gain,
        "crs": crs,
        "portable_recovery_proxy": portable,
        "other_view_crs": portable,
        "cross_backend_crs": portable,
        "reward_variance": 0.1,
        "question_difficulty": 0.7,
        "paired_view_count": 2,
        "paired_backend_count": 2,
    }


class TraceGoTests(unittest.TestCase):
    def test_recoverable_is_not_merely_hard(self) -> None:
        self.assertEqual(
            classify_episode(
                0.0,
                1.0,
                fail_threshold=0.5,
                solve_threshold=0.5,
                min_recovery=0.25,
            ),
            "recoverable",
        )
        self.assertEqual(
            classify_episode(
                0.0,
                0.0,
                fail_threshold=0.5,
                solve_threshold=0.5,
                min_recovery=0.25,
            ),
            "unrecoverable",
        )
        self.assertEqual(
            classify_episode(
                1.0,
                1.0,
                fail_threshold=0.5,
                solve_threshold=0.5,
                min_recovery=0.25,
            ),
            "easy",
        )
        score = counterfactual_recovery_score(
            0.0,
            1.0,
            fail_threshold=0.5,
            solve_threshold=0.5,
            min_recovery=0.25,
            search_count=2,
            search_cost=0.01,
        )
        self.assertAlmostEqual(score, 0.99)

    def test_paired_portable_proxy_requires_other_view(self) -> None:
        episodes = [
            {
                "question_id": "q1",
                "view_id": "bm25:k3",
                "backend": "bm25",
                "policy_tag": "teacher",
                "final_recall": 1.0,
                "answer_em": 1.0,
                "search_count": 2,
                "crs": 0.8,
            },
            {
                "question_id": "q1",
                "view_id": "bm25:k5",
                "backend": "bm25",
                "policy_tag": "teacher",
                "final_recall": 1.0,
                "answer_em": 1.0,
                "search_count": 2,
                "crs": 1.0,
            },
            {
                "question_id": "q1",
                "view_id": "e5:k3",
                "backend": "e5",
                "policy_tag": "teacher",
                "final_recall": 1.0,
                "answer_em": 1.0,
                "search_count": 2,
                "crs": 0.5,
            },
        ]
        add_group_features(episodes)
        # Same-backend top-k variants do not inflate cross-retriever portability.
        self.assertAlmostEqual(episodes[0]["portable_recovery_proxy"], 0.4)
        self.assertAlmostEqual(episodes[1]["portable_recovery_proxy"], 0.5)
        self.assertAlmostEqual(episodes[2]["portable_recovery_proxy"], 0.45)

    def test_b_pools_separate_recovery_from_depth(self) -> None:
        rows = [
            transition(
                "r1",
                question_id="q1",
                backend="bm25",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=1.0,
                crs=1.0,
            ),
            transition(
                "d1",
                question_id="q2",
                backend="bm25",
                episode_class="unrecoverable",
                source_turn=3,
                search_count=3,
                evidence_gain=0.0,
                crs=0.0,
            ),
        ]
        recovered, deep = recovered_vs_deep_pools(
            rows,
            source_backend="bm25",
            short_turn=2,
            deep_turn=3,
            recovery_epsilon=1e-8,
        )
        self.assertEqual([row["transition_id"] for row in recovered], ["r1"])
        self.assertEqual([row["transition_id"] for row in deep], ["d1"])

    def test_c_paired_pool_requires_cross_view_recovery(self) -> None:
        rows = [
            transition(
                "b1",
                question_id="q1",
                backend="bm25",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=0.5,
                crs=0.5,
                portable=0.25,
            ),
            transition(
                "e1",
                question_id="q1",
                backend="e5",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=0.5,
                crs=0.5,
                portable=0.25,
            ),
            transition(
                "b2",
                question_id="q2",
                backend="bm25",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=1.0,
                crs=1.0,
                portable=0.0,
            ),
        ]
        pool = paired_portable_pool(
            rows,
            source_backend="bm25",
            target_backend="e5",
            recovery_epsilon=1e-8,
        )
        self.assertEqual([row["question_id"] for row in pool], ["q1"])

    def test_matching_preserves_marginals_and_count(self) -> None:
        positive = [
            transition(
                f"p{index}",
                question_id=f"p{index}",
                backend="bm25",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=1.0,
                crs=1.0,
            )
            for index in range(4)
        ]
        negative = [
            transition(
                f"n{index}",
                question_id=f"n{index}",
                backend="bm25",
                episode_class="unrecoverable",
                source_turn=3,
                search_count=3,
                evidence_gain=0.0,
                crs=0.0,
            )
            for index in range(4)
        ]
        left, right = match_by_marginals(positive, negative, count=3, seed=42)
        self.assertEqual(len(left), 3)
        self.assertEqual(len(right), 3)
        self.assertTrue(all(row["dataset"] == "toy" for row in left + right))

    def test_hierarchical_bootstrap_is_paired(self) -> None:
        rows = []
        for seed in (13, 42, 87):
            for example in ("a", "b", "c"):
                rows.append(
                    {
                        "seed": seed,
                        "example": example,
                        "variant": "recovered",
                        "gain": 0.3,
                    }
                )
                rows.append(
                    {
                        "seed": seed,
                        "example": example,
                        "variant": "deep",
                        "gain": 0.1,
                    }
                )
        result = hierarchical_bootstrap_difference(
            rows,
            value_key="gain",
            variant_key="variant",
            positive_variant="recovered",
            negative_variant="deep",
            seed_key="seed",
            example_key="example",
            samples=100,
            random_seed=1,
        )
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertGreater(result["ci_low"], 0.19)

    def test_signed_query_credit_discourages_zero_gain(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "examples.jsonl"
            positive = transition(
                "p",
                question_id="qp",
                backend="bm25",
                episode_class="recoverable",
                source_turn=2,
                search_count=2,
                evidence_gain=0.5,
                crs=0.5,
            )
            negative = transition(
                "n",
                question_id="qn",
                backend="bm25",
                episode_class="unrecoverable",
                source_turn=3,
                search_count=3,
                evidence_gain=0.0,
                crs=0.0,
            )
            _write_examples(path, [positive, negative], negative_query_weight=0.25)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["weight"], 0.5)
            self.assertEqual(rows[1]["weight"], -0.25)

    def test_scheduler_completion_signature(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "metrics.json").write_text(
                '{"job_signature":"abc"}\n', encoding="utf-8"
            )
            self.assertTrue(completed({"output_dir": str(output), "job_signature": "abc"}))
            self.assertFalse(completed({"output_dir": str(output), "job_signature": "def"}))

    def test_build_transition_uses_prior_observations_only(self) -> None:
        episodes = [
            {
                "episode_id": "ep",
                "question_id": "q",
                "question": "Who wrote it?",
                "dataset": "toy",
                "policy_tag": "teacher",
                "policy_seed": 13,
                "backend": "bm25",
                "topk": 3,
                "view_id": "bm25:k3",
                "split": "train",
                "episode_class": "recoverable",
                "search_count": 2,
                "turn1_recall": 0.0,
                "final_recall": 1.0,
                "total_recovery": 1.0,
                "crs": 1.0,
                "portable_recovery_proxy": 0.5,
                "other_view_crs": 0.5,
                "cross_backend_crs": 0.5,
                "reward_variance": 0.0,
                "question_difficulty": 0.5,
                "paired_view_count": 2,
                "paired_backend_count": 2,
                "turns": [
                    {
                        "query": "first",
                        "observed_titles": ["Intermediate"],
                        "support_recall": 0.0,
                        "evidence_gain": 0.0,
                    },
                    {
                        "query": "second",
                        "observed_titles": ["Gold"],
                        "support_recall": 1.0,
                        "evidence_gain": 1.0,
                    },
                ],
            }
        ]
        rows = build_transitions(episodes)
        self.assertEqual(len(rows), 1)
        self.assertIn("Intermediate", rows[0]["prompt"])
        self.assertNotIn("Gold", rows[0]["prompt"])
        self.assertEqual(rows[0]["target"], "second")


if __name__ == "__main__":
    unittest.main()

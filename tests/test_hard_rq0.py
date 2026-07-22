from __future__ import annotations

import unittest

import pandas as pd

from stackpilot.hard_rq0_report import gain_over_base, home_excess, matched_hard_question_ids
from stackpilot.normalize_hard_results import normalize
from stackpilot.retrieval_clients import normalize_document


class HardRQ0Tests(unittest.TestCase):
    def test_gain_over_base_and_home_excess(self) -> None:
        rows = []
        scores = {
            ("base-qwen", 0, "bm25"): 0.40,
            ("base-qwen", 0, "e5"): 0.60,
            ("bm25-specialist", 13, "bm25"): 0.55,
            ("bm25-specialist", 13, "e5"): 0.65,
            ("e5-specialist", 13, "bm25"): 0.45,
            ("e5-specialist", 13, "e5"): 0.75,
        }
        for (tag, seed, backend), score in scores.items():
            rows.append(
                {
                    "subset": "all",
                    "policy_tag": tag,
                    "seed": seed,
                    "question_id": "q1",
                    "dataset": "toy",
                    "backend": backend,
                    "topk": 3,
                    "support_recall": score,
                }
            )
        frame = pd.DataFrame(rows)
        gains = gain_over_base(frame, "support_recall")
        interactions = home_excess(gains).set_index("policy_tag")
        self.assertAlmostEqual(
            float(interactions.loc["bm25-specialist", "home_excess_gain"]),
            0.10,
        )
        self.assertAlmostEqual(
            float(interactions.loc["e5-specialist", "home_excess_gain"]),
            0.10,
        )

    def test_matched_hard_requires_base_difficulty_and_recovery(self) -> None:
        rows = []
        for backend, first in (("bm25", 0.0), ("e5", 0.5)):
            rows.append(
                {
                    "policy_tag": "base-qwen",
                    "seed": 0,
                    "question_id": "q1",
                    "dataset": "toy",
                    "backend": backend,
                    "topk": 3,
                    "turn1_support_recall": first,
                    "turn3_support_recall": first,
                }
            )
        rows.append(
            {
                "policy_tag": "bm25-specialist",
                "seed": 13,
                "question_id": "q1",
                "dataset": "toy",
                "backend": "bm25",
                "topk": 3,
                "turn1_support_recall": 0.0,
                "turn3_support_recall": 1.0,
            }
        )
        matched = matched_hard_question_ids(pd.DataFrame(rows))
        self.assertTrue(bool(matched.loc[0, "base_hard"]))
        self.assertTrue(bool(matched.loc[0, "recoverable"]))
        self.assertTrue(bool(matched.loc[0, "matched_hard"]))

    def test_missing_turn_recall_is_carried_forward(self) -> None:
        row = normalize(
            {
                "turns": [
                    {"turn": 1, "support_recall": 0.5, "evidence_gain": 0.5}
                ]
            }
        )
        self.assertEqual(row["turn2_support_recall"], 0.5)
        self.assertEqual(row["turn3_support_recall"], 0.5)
        self.assertEqual(row["turn2_evidence_gain"], 0.0)
        self.assertEqual(row["turn3_evidence_gain"], 0.0)

    def test_wiki18_title_is_recovered_from_contents(self) -> None:
        title, text = normalize_document(
            {"id": "1", "contents": '"Story of Your Life"\nA novella by Ted Chiang.'}
        )
        self.assertEqual(title, "Story of Your Life")
        self.assertEqual(text, "A novella by Ted Chiang.")


if __name__ == "__main__":
    unittest.main()

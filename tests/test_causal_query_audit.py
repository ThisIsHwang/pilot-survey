from __future__ import annotations

import unittest
from pathlib import Path

from stackpilot.causal_query_common import (
    attach_query_effects,
    parse_alternative_queries,
    spearman,
    transferred_bridge_tokens,
    validate_alternatives,
)
from stackpilot.causal_query_prepare import (
    build_candidate_states,
    select_balanced_states,
)
from stackpilot.causal_query_report import top1_accuracy


class CausalQueryAuditTests(unittest.TestCase):
    def test_alternative_parser_accepts_json_and_lines(self) -> None:
        styles = ["lexical", "semantic", "entity"]
        parsed = parse_alternative_queries(
            '{"lexical":"Ted Chiang birthplace",'
            '"semantic":"Where was the author of Story of Your Life born?",'
            '"entity":"Story of Your Life writer birth city"}',
            styles,
        )
        self.assertEqual(set(parsed), set(styles))
        line_parsed = parse_alternative_queries(
            "LEXICAL: Ted Chiang birthplace\nSEMANTIC: writer birth location\n",
            styles[:2],
        )
        self.assertEqual(line_parsed["lexical"], "Ted Chiang birthplace")

    def test_alternative_validation_rejects_duplicates_and_unknown_queries(self) -> None:
        valid = validate_alternatives(
            {
                "lexical": "Ted Chiang birthplace",
                "semantic": "Arrival director box office",  # known token but wrong intent is a later audit limitation
                "entity": "completely unrelated bananas",
            },
            styles=["lexical", "semantic", "entity"],
            factual_query="Story of Your Life author birthplace",
            question="Where was the author of Story of Your Life born?",
            observed_titles=["Ted Chiang"],
            minimum_tokens=2,
            length_ratio_low=0.4,
            length_ratio_high=2.0,
        )
        self.assertIn("lexical", valid)
        self.assertIn("semantic", valid)
        self.assertNotIn("entity", valid)

    def test_query_effect_decomposition_and_bridge_label(self) -> None:
        branches = [
            {
                "candidate_id": "a",
                "immediate_support_gain": 0.0,
                "recall_after_intervention": 0.0,
                "final_support_recall": 1.0,
                "answer_f1": 1.0,
                "suffix_search_count": 1,
                "next_query": "Ted Chiang birthplace",
                "next_query_evidence_gain": 1.0,
                "transferred_bridge_token_count": 1,
            },
            {
                "candidate_id": "b",
                "immediate_support_gain": 0.5,
                "recall_after_intervention": 0.5,
                "final_support_recall": 0.5,
                "answer_f1": 0.0,
                "suffix_search_count": 0,
                "next_query": "",
                "next_query_evidence_gain": 0.0,
                "transferred_bridge_token_count": 0,
            },
            {
                "candidate_id": "c",
                "immediate_support_gain": 0.0,
                "recall_after_intervention": 0.0,
                "final_support_recall": 0.0,
                "answer_f1": 0.0,
                "suffix_search_count": 1,
                "next_query": "repeat",
                "next_query_evidence_gain": 0.0,
                "transferred_bridge_token_count": 0,
            },
        ]
        output = attach_query_effects(
            branches,
            answer_weight=0.25,
            search_cost=0.01,
            epsilon=1e-8,
            bridge_min_support_tqe=0.05,
        )
        self.assertEqual(output[0]["mediated_bridge"], 1)
        self.assertEqual(output[0]["positive_causal_bridge"], 1)
        self.assertEqual(output[1]["redundant_direct"], 1)
        for row in output:
            self.assertAlmostEqual(
                row["support_tqe"], row["direct_effect"] + row["downstream_effect"]
            )

    def test_transferred_bridge_tokens_excludes_prior_knowledge(self) -> None:
        tokens = transferred_bridge_tokens(
            next_query="Ted Chiang birthplace Bellevue",
            intervention_titles=["Ted Chiang", "Bellevue, Washington"],
            prior_text="Who wrote the story? Ted",
        )
        self.assertIn("chiang", tokens)
        self.assertIn("bellevue", tokens)
        self.assertNotIn("ted", tokens)

    def test_prepare_builds_intervention_states(self) -> None:
        cfg = {
            "source": {
                "datasets": ["toy"],
                "policy_tags": [],
                "topks": [3],
                "intervention_turns": [2, 3],
                "require_protocol_success": True,
            },
            "agent": {"max_search_turns": 4},
        }
        raw = {
            "question_id": "q1",
            "question": "Who wrote it?",
            "answers": ["Author"],
            "support_titles": ["Book", "Author"],
            "dataset": "toy",
            "backend": "bm25",
            "topk": 3,
            "policy_tag": "teacher",
            "seed": 13,
            "protocol_failure": 0,
            "turns": [
                {
                    "query": "book title",
                    "observed_titles": ["Book"],
                    "support_recall": 0.5,
                    "evidence_gain": 0.5,
                },
                {
                    "query": "Book author",
                    "observed_titles": ["Author"],
                    "support_recall": 1.0,
                    "evidence_gain": 0.5,
                },
            ],
        }
        states = build_candidate_states([(raw, Path("raw.jsonl"))], cfg=cfg)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["source_turn"], 2)
        self.assertEqual(states[0]["factual_query"], "Book author")

    def test_balanced_selection_uses_one_state_per_question_backend(self) -> None:
        rows = []
        for backend in ("bm25", "e5"):
            for index in range(8):
                rows.append(
                    {
                        "state_id": f"{backend}-{index}",
                        "question_id": f"q{index}",
                        "backend": backend,
                        "dataset": "toy-a" if index % 2 else "toy-b",
                        "source_turn": 2 if index % 3 else 3,
                        "policy_tag": "teacher",
                        "policy_seed": 13,
                    }
                )
        selected = select_balanced_states(
            rows,
            count_per_backend=4,
            seed=1,
            one_state_per_question_backend=True,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            len({(row["question_id"], row["backend"]) for row in selected}), 8
        )

    def test_spearman_and_top1_detect_proxy_mismatch(self) -> None:
        self.assertLess(spearman([0.0, 1.0, 0.5], [1.0, 0.0, 0.5]), 0.0)
        rows = [
            {
                "state_id": "s",
                "candidate_id": "a",
                "immediate_support_gain": 1.0,
                "final_support_recall": 0.5,
            },
            {
                "state_id": "s",
                "candidate_id": "b",
                "immediate_support_gain": 0.0,
                "final_support_recall": 1.0,
            },
        ]
        self.assertEqual(top1_accuracy(rows), 0.0)


if __name__ == "__main__":
    unittest.main()

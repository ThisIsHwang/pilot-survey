from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from stackpilot.query_credit_qwen35_runtime import (
    ThinkingLeakError,
    assert_non_thinking_message,
    build_request_options,
)
from stackpilot.query_credit_weekend_collect_support import _candidate_bank


class Qwen35NoThinkRuntimeTest(unittest.TestCase):
    def test_request_forces_non_thinking_and_recommended_sampling(self) -> None:
        cfg = {
            "model": {
                "chat_template_kwargs": {"enable_thinking": True},
                "sampling": {
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                },
            }
        }
        options = build_request_options(cfg, temperature=0.7)
        self.assertIs(
            options["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            False,
        )
        self.assertEqual(options["extra_body"]["top_k"], 20)
        self.assertEqual(options["top_p"], 0.8)
        self.assertEqual(options["presence_penalty"], 1.5)

    def test_thinking_leak_is_logged_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaks.jsonl"
            previous = os.environ.get("STACKPILOT_THINKING_LEAK_LOG")
            os.environ["STACKPILOT_THINKING_LEAK_LOG"] = str(path)
            try:
                with self.assertRaises(ThinkingLeakError):
                    assert_non_thinking_message(
                        SimpleNamespace(
                            reasoning_content="hidden reasoning",
                            reasoning=None,
                            model_extra={},
                        ),
                        "<answer>x</answer>",
                        seed=13,
                    )
            finally:
                if previous is None:
                    os.environ.pop("STACKPILOT_THINKING_LEAK_LOG", None)
                else:
                    os.environ["STACKPILOT_THINKING_LEAK_LOG"] = previous
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["reason"], "non-empty-reasoning-field")
            self.assertEqual(rows[0]["seed"], 13)

    def test_plain_answer_passes(self) -> None:
        assert_non_thinking_message(
            SimpleNamespace(
                reasoning_content=None,
                reasoning=None,
                model_extra={},
            ),
            "<answer>OK</answer>",
        )


class Qwen35WeekendContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_config_targets_qwen35_for_exactly_five_days(self) -> None:
        cfg = yaml.safe_load(
            (self.root / "configs/query_credit_weekend.yaml").read_text()
        )
        self.assertEqual(cfg["model"]["base_model"], "Qwen/Qwen3.5-9B")
        self.assertIs(cfg["model"]["enable_thinking"], False)
        self.assertIs(cfg["model"]["require_non_thinking"], True)
        self.assertIs(
            cfg["model"]["chat_template_kwargs"]["enable_thinking"], False
        )
        self.assertEqual(sum(cfg["budget"].values()), 120)
        targets = set(cfg["gradient"]["target_modules"])
        self.assertTrue({"q_proj", "in_proj_qkv", "in_proj_a", "out_proj"} <= targets)
        node8 = cfg["profiles"]["node8"]
        self.assertEqual(len(node8["continuation_seeds"]), 6)
        self.assertIs(node8["include_source_factual"], False)
        self.assertIs(node8["allow_controlled_fallback"], False)
        self.assertEqual(
            cfg["gates"]["audit"]["minimum_direct_policy_candidate_fraction"],
            1.0,
        )

    def test_qwen35_results_cannot_share_qwen25_output_namespace(self) -> None:
        cfg = yaml.safe_load(
            (self.root / "configs/query_credit_weekend.yaml").read_text()
        )
        self.assertEqual(cfg["work_dir"], "work/query_credit_weekend_qwen35")
        self.assertEqual(
            cfg["source"]["cross_model_artifact_policy"],
            "reject",
        )
        self.assertIn("qwen35", cfg["collection"]["selection_salt"])
        self.assertIn("qwen35", cfg["micro_update"]["split_salt"])

    def test_candidate_bank_excludes_legacy_factual_query(self) -> None:
        state = {"state_id": "s", "factual_query": "legacy qwen2.5 query"}
        prefix = {"messages": [{"role": "user", "content": "Question: x"}]}
        profile = {
            "candidates_per_state": 2,
            "minimum_candidates_per_state": 2,
            "include_source_factual": False,
            "allow_controlled_fallback": False,
            "sibling_generation": {
                "attempts": 2,
                "temperature": 0.7,
                "max_tokens": 96,
            },
        }
        with patch(
            "stackpilot.query_credit_weekend_collect_support._complete",
            side_effect=[
                "<search>qwen35 direct query one</search>",
                "<search>qwen35 direct query two</search>",
            ],
        ):
            rows = _candidate_bank(
                {"candidates": []},
                state=state,
                prefix=prefix,
                causal_cfg={},
                profile=profile,
            )
        self.assertEqual(len(rows), 2)
        self.assertNotIn(state["factual_query"], [row["query"] for row in rows])
        self.assertEqual({row["origin"] for row in rows}, {"direct-policy-sibling"})

    def test_vllm_launcher_is_text_only_nonthinking_and_unbatched(self) -> None:
        text = (self.root / "query_credit/launch_qwen35_vllm.sh").read_text()
        self.assertIn("--language-model-only", text)
        self.assertIn("--reasoning-parser qwen3", text)
        self.assertIn("--default-chat-template-kwargs", text)
        self.assertIn("enable_thinking", text)
        self.assertIn("VLLM_BATCH_INVARIANT=0", text)
        self.assertIn("--max-num-seqs", text)

    def test_local_template_guard_forces_false_and_rejects_true(self) -> None:
        text = (
            self.root / "query_credit/qwen35_site/sitecustomize.py"
        ).read_text()
        self.assertIn('kwargs["enable_thinking"] = False', text)
        self.assertIn("forbids enable_thinking=True", text)
        self.assertNotIn("/nothink", text.lower())

    def test_isolated_qwen35_entrypoint_checks_fresh_namespace(self) -> None:
        text = (self.root / "query_credit/run_qwen35_five_day.sh").read_text()
        self.assertIn("work/query_credit_weekend_qwen35", text)
        self.assertIn("cross_model_artifact_policy", text)
        self.assertIn("require_non_thinking", text)


if __name__ == "__main__":
    unittest.main()

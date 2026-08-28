from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from stackpilot.query_credit_qwen35_runtime import (
    ThinkingLeakError,
    assert_non_thinking_message,
    build_request_options,
)


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
        self.assertIs(
            cfg["model"]["chat_template_kwargs"]["enable_thinking"], False
        )
        self.assertEqual(sum(cfg["budget"].values()), 120)
        targets = set(cfg["gradient"]["target_modules"])
        self.assertTrue({"q_proj", "in_proj_qkv", "in_proj_a", "out_proj"} <= targets)
        self.assertEqual(len(cfg["profiles"]["node8"]["continuation_seeds"]), 6)

    def test_vllm_launcher_is_text_only_nonthinking_and_unbatched(self) -> None:
        text = (self.root / "query_credit/launch_qwen35_vllm.sh").read_text()
        self.assertIn("--language-model-only", text)
        self.assertIn("--reasoning-parser qwen3", text)
        self.assertIn("--default-chat-template-kwargs", text)
        self.assertIn("enable_thinking", text)
        self.assertIn("VLLM_BATCH_INVARIANT=0", text)
        self.assertIn("--max-num-seqs", text)

    def test_local_template_guard_does_not_use_nothink_token(self) -> None:
        text = (
            self.root / "query_credit/qwen35_site/sitecustomize.py"
        ).read_text()
        self.assertIn('kwargs.setdefault("enable_thinking", False)', text)
        self.assertNotIn("/nothink", text.lower())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from stackpilot.trace_model_contract import planned_model, validate_parameter_count


ROOT = Path(__file__).resolve().parents[1]


class Trace7BContractTests(unittest.TestCase):
    def test_default_config_targets_7b_with_same_effective_batch(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "trace_go.yaml").read_text(encoding="utf-8")
        )
        model = config["model"]
        lora = config["lora"]
        self.assertEqual(model["base_model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertLessEqual(model["minimum_parameters"], 7_000_000_000)
        self.assertGreaterEqual(model["maximum_parameters"], 7_000_000_000)
        self.assertEqual(lora["batch_size"], 2)
        self.assertEqual(lora["batch_size"] * lora["gradient_accumulation"], 16)
        self.assertEqual(lora["eval_batch_size"], 4)
        self.assertTrue(lora["gradient_checkpointing"])

    def test_plan_script_uses_7b_default(self) -> None:
        script = (ROOT / "trace_go" / "plan.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", script)
        self.assertNotIn("Qwen/Qwen2.5-3B-Instruct", script)

    def test_parameter_contract_rejects_stale_3b(self) -> None:
        validate_parameter_count(
            7_615_000_000,
            minimum_parameters=6_000_000_000,
            maximum_parameters=9_000_000_000,
        )
        with self.assertRaises(RuntimeError):
            validate_parameter_count(
                3_090_000_000,
                minimum_parameters=6_000_000_000,
                maximum_parameters=9_000_000_000,
            )

    def test_all_jobs_must_share_one_model_contract(self) -> None:
        model_ref, trust_remote_code = planned_model(
            [
                {
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "trust_remote_code": False,
                },
                {
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "trust_remote_code": False,
                },
            ]
        )
        self.assertEqual(model_ref, "Qwen/Qwen2.5-7B-Instruct")
        self.assertFalse(trust_remote_code)
        with self.assertRaises(RuntimeError):
            planned_model(
                [
                    {"base_model": "Qwen/Qwen2.5-7B-Instruct"},
                    {"base_model": "Qwen/Qwen2.5-3B-Instruct"},
                ]
            )


if __name__ == "__main__":
    unittest.main()

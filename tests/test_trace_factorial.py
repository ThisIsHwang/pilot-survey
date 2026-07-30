from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stackpilot.trace_factorial import (
    VARIANTS,
    _require_finite,
    _write_positive_examples,
    build_factorial_effect_rows,
    factorial_pools,
    match_factorial_pools,
)
from stackpilot.trace_factorial_lora_job import _validate_positive_job


def row(
    transition_id: str,
    *,
    episode_id: str,
    question_id: str,
    source_turn: int,
    search_count: int,
    episode_class: str,
    evidence_gain: float,
    total_recovery: float,
    dataset: str = "toy",
) -> dict:
    return {
        "transition_id": transition_id,
        "episode_id": episode_id,
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "dataset": dataset,
        "policy_tag": "teacher",
        "policy_seed": 13,
        "backend": "bm25",
        "topk": 3,
        "split": "train",
        "source_turn": source_turn,
        "search_count": search_count,
        "episode_class": episode_class,
        "evidence_gain": evidence_gain,
        "total_recovery": total_recovery,
        "crs": total_recovery,
        "question_difficulty": 0.5,
        "prompt": f"prompt {transition_id}",
        "target": f"query {transition_id}",
    }


class TraceFactorialTests(unittest.TestCase):
    def test_factorial_pools_use_first_recovery_and_final_failure(self) -> None:
        rows = [
            row(
                "sr1",
                episode_id="sr",
                question_id="q-sr",
                source_turn=2,
                search_count=2,
                episode_class="recoverable",
                evidence_gain=1.0,
                total_recovery=1.0,
            ),
            row(
                "dr1",
                episode_id="dr",
                question_id="q-dr",
                source_turn=2,
                search_count=4,
                episode_class="recoverable",
                evidence_gain=0.0,
                total_recovery=1.0,
            ),
            row(
                "dr2",
                episode_id="dr",
                question_id="q-dr",
                source_turn=3,
                search_count=4,
                episode_class="recoverable",
                evidence_gain=0.5,
                total_recovery=1.0,
            ),
            row(
                "dr3",
                episode_id="dr",
                question_id="q-dr",
                source_turn=4,
                search_count=4,
                episode_class="recoverable",
                evidence_gain=0.5,
                total_recovery=1.0,
            ),
            row(
                "su",
                episode_id="su",
                question_id="q-su",
                source_turn=2,
                search_count=2,
                episode_class="unrecoverable",
                evidence_gain=0.0,
                total_recovery=0.0,
            ),
            row(
                "du2",
                episode_id="du",
                question_id="q-du",
                source_turn=2,
                search_count=4,
                episode_class="unrecoverable",
                evidence_gain=0.0,
                total_recovery=0.0,
            ),
            row(
                "du4",
                episode_id="du",
                question_id="q-du",
                source_turn=4,
                search_count=4,
                episode_class="unrecoverable",
                evidence_gain=0.0,
                total_recovery=0.0,
            ),
        ]
        pools = factorial_pools(
            rows,
            source_backend="bm25",
            short_max_turn=2,
            deep_min_turn=3,
            recovery_epsilon=1e-8,
        )
        self.assertEqual(set(pools), set(VARIANTS))
        self.assertEqual(pools["short-recovered"][0]["transition_id"], "sr1")
        self.assertEqual(pools["deep-recovered"][0]["transition_id"], "dr2")
        self.assertEqual(pools["short-unrecovered"][0]["transition_id"], "su")
        self.assertEqual(pools["deep-unrecovered"][0]["transition_id"], "du4")

    def test_matching_returns_equal_complete_quartets(self) -> None:
        pools = {}
        recovered_variants = {"short-recovered", "deep-recovered"}
        for variant in VARIANTS:
            pools[variant] = [
                row(
                    f"{variant}-{index}",
                    episode_id=f"{variant}-{index}",
                    question_id=f"{variant}-q{index}",
                    source_turn=2 if variant.startswith("short") else 3,
                    search_count=2 if variant.startswith("short") else 3,
                    episode_class=(
                        "recoverable"
                        if variant in recovered_variants
                        else "unrecoverable"
                    ),
                    evidence_gain=1.0 if variant in recovered_variants else 0.0,
                    total_recovery=1.0 if variant in recovered_variants else 0.0,
                )
                for index in range(4)
            ]
        matched, diagnostics = match_factorial_pools(
            pools,
            count=3,
            seed=1,
            group_keys=("dataset", "backend", "topk"),
        )
        self.assertEqual(
            {name: len(values) for name, values in matched.items()},
            {name: 3 for name in VARIANTS},
        )
        self.assertEqual(diagnostics["matched_count"], 3)

    def test_positive_writer_and_job_guard_reject_negative_weights(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            training_path = root / "train.jsonl"
            example = row(
                "x",
                episode_id="x",
                question_id="qx",
                source_turn=2,
                search_count=2,
                episode_class="unrecoverable",
                evidence_gain=0.0,
                total_recovery=0.0,
            )
            _write_positive_examples(training_path, [example])
            payload = json.loads(training_path.read_text().strip())
            self.assertEqual(payload["weight"], 1.0)
            job_path = root / "job.json"
            job = {
                "experiment_id": "EXP-012",
                "weight_mode": "positive-only",
                "train_file": str(training_path),
            }
            job_path.write_text(json.dumps(job), encoding="utf-8")
            _validate_positive_job(job_path)
            payload["weight"] = -0.25
            training_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _validate_positive_job(job_path)

    def test_factorial_effects_recover_known_main_effect(self) -> None:
        gains = {
            "short-recovered": 0.30,
            "short-unrecovered": 0.10,
            "deep-recovered": 0.40,
            "deep-unrecovered": 0.20,
        }
        rows = []
        for variant, gain in gains.items():
            rows.append(
                {
                    "direction": "bm25-to-e5",
                    "variant": variant,
                    "seed": 13,
                    "example_id": "q1",
                    "baseline_nll": 2.0,
                    "adapted_nll": 2.0 - gain,
                    "heldout_gain": gain,
                }
            )
        effects = build_factorial_effect_rows(
            pd.DataFrame(rows), baseline_tolerance=1e-8
        )
        self.assertAlmostEqual(float(effects.iloc[0]["recovery_effect"]), 0.20)
        self.assertAlmostEqual(float(effects.iloc[0]["depth_effect"]), 0.10)
        self.assertAlmostEqual(float(effects.iloc[0]["interaction"]), 0.0)

    def test_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            _require_finite(float("nan"), label="test")
        with self.assertRaises(RuntimeError):
            _require_finite(float("inf"), label="test")


if __name__ == "__main__":
    unittest.main()

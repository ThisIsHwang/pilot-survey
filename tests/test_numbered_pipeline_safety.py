from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from stackpilot.numbered_experiment_report import load_completed_numbered_results


def write_completed_run(root: Path, *, profile: str = "pilot") -> Path:
    run = root / "EXP-003__seed-013__profile-pilot__variant-blind"
    run.mkdir(parents=True)
    episode = {
        "experiment_id": "EXP-003",
        "run_id": run.name,
        "run_signature": "signature",
        "profile": profile,
        "policy_tag": "mixed-blind",
        "seed": 13,
        "question_id": "q1",
        "dataset": "musique",
        "backend": "bm25",
        "topk": 3,
        "em": 0.0,
        "f1": 0.0,
        "support_recall": 0.5,
        "turn2_evidence_gain": 0.1,
        "turn3_evidence_gain": 0.2,
        "recovery_at_2": 0.0,
        "recovery_at_3": 1.0,
    }
    episodes = run / "episodes.jsonl"
    episodes.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    digest = hashlib.sha256(episodes.read_bytes()).hexdigest()
    manifest = {
        "schema": 2,
        "status": "complete",
        "experiment_id": "EXP-003",
        "run_id": run.name,
        "run_signature": "signature",
        "profile": profile,
        "questions": 1,
        "episodes": 1,
        "backends": ["bm25"],
        "topks": [3],
        "episodes_sha256": digest,
    }
    (run / "evaluation_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return run


class NumberedPipelineSafetyTests(unittest.TestCase):
    def test_report_uses_only_completed_profile_matched_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = write_completed_run(root)
            partial = root / "partial"
            partial.mkdir()
            (partial / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            frame = load_completed_numbered_results(
                root, profile="pilot", experiment_id="EXP-003"
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["question_id"], "q1")

            with (completed / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                load_completed_numbered_results(
                    root, profile="pilot", experiment_id="EXP-003"
                )

    def test_report_rejects_results_from_another_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_completed_run(root, profile="smoke")
            with self.assertRaisesRegex(RuntimeError, "No completed EXP-003"):
                load_completed_numbered_results(
                    root, profile="pilot", experiment_id="EXP-003"
                )

    def test_scripts_validate_artifacts_and_external_exp002(self) -> None:
        root = Path(__file__).resolve().parents[1]
        training = (root / "experiments" / "train_mixed_policy.sh").read_text(
            encoding="utf-8"
        )
        merger = (
            root / "experiments" / "merge_numbered_checkpoint.sh"
        ).read_text(encoding="utf-8")
        exp006 = (root / "experiments" / "EXP-006" / "run.sh").read_text(
            encoding="utf-8"
        )
        report = (root / "experiments" / "make_report.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("validate_checkpoint_artifact", training)
        self.assertIn("Reusing validated completed run", training)
        self.assertIn('validate_checkpoint_artifact "$FINAL_CHECKPOINT"', training)
        self.assertIn('"${OUTPUT_DIR}.complete.json"', merger)
        self.assertIn("EXP002_ROOT=", exp006)
        self.assertIn("EXP002_COMPLETE_MARKER=", exp006)
        self.assertIn("--profile", report)


if __name__ == "__main__":
    unittest.main()

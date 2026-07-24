from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from stackpilot.hard_rq0_contract import (
    NUMBERED_EVALUATION_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
)
from stackpilot.numbered_experiment_report import (
    load_completed_numbered_results,
    stable_signature,
)


def write_completed_run(root: Path, *, profile: str = "pilot") -> Path:
    run = root / "EXP-003__seed-013__profile-pilot__variant-blind"
    run.mkdir(parents=True)
    evaluation_context = {
        "schema": RESULT_SCHEMA,
        "question_ids": ["q1"],
        "backends": ["bm25"],
        "topks": [3],
        "protocol": {
            "agent": {"max_search_turns": 4},
            "inject_backend_id": False,
        },
    }
    evaluation_signature = stable_signature(evaluation_context)
    episode = {
        "schema": RESULT_SCHEMA,
        "experiment_id": "EXP-003",
        "run_id": run.name,
        "run_signature": "signature",
        "evaluation_signature": evaluation_signature,
        "profile": profile,
        "variant": "blind",
        "policy_tag": "mixed-blind",
        "seed": 13,
        "backend_id_injected": False,
        "question_id": "q1",
        "question": "Question?",
        "answers": ["answer"],
        "support_titles": ["Evidence"],
        "dataset": "musique",
        "backend": "bm25",
        "topk": 3,
        "prediction": "answer",
        "raw_text_prediction": "answer",
        "em": 1.0,
        "f1": 1.0,
        "raw_text_em": 1.0,
        "raw_text_f1": 1.0,
        "protocol_failure": 0,
        "invalid_action_count": 0,
        "support_recall": 0.0,
        "turn1_support_recall": 0.0,
        "turn2_support_recall": 0.0,
        "turn3_support_recall": 0.0,
        "turn2_evidence_gain": 0.0,
        "turn3_evidence_gain": 0.0,
        "recovery_at_2": 0.0,
        "recovery_at_3": 0.0,
        "full_recovery_at_2": 0.0,
        "full_recovery_at_3": 0.0,
        "search_count": 0,
        "queries": [],
        "turns": [],
    }
    episodes = run / "episodes.jsonl"
    episodes.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    digest = hashlib.sha256(episodes.read_bytes()).hexdigest()
    manifest = {
        "schema": NUMBERED_EVALUATION_MANIFEST_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "status": "complete",
        "experiment_id": "EXP-003",
        "run_id": run.name,
        "run_signature": "signature",
        "evaluation_signature": evaluation_signature,
        "profile": profile,
        "variant": "blind",
        "policy_tag": "mixed-blind",
        "seed": 13,
        "questions": 1,
        "episodes": 1,
        "backends": ["bm25"],
        "topks": [3],
        "backend_id_injected": False,
        "episodes_sha256": digest,
        "evaluation_context": evaluation_context,
    }
    (run / "evaluation_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return run


def rewrite_episode(run: Path, update: dict[str, object]) -> None:
    episodes = run / "episodes.jsonl"
    episode = json.loads(episodes.read_text(encoding="utf-8"))
    episode.update(update)
    episodes.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    manifest_path = run / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes_sha256"] = hashlib.sha256(episodes.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


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

    def test_report_rejects_schema_labeled_episode_contract_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = write_completed_run(root)
            rewrite_episode(completed, {"raw_text_f1": 0.5})
            with self.assertRaisesRegex(RuntimeError, "Episode contract"):
                load_completed_numbered_results(
                    root, profile="pilot", experiment_id="EXP-003"
                )

    def test_report_rejects_manifest_episode_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = write_completed_run(root)
            rewrite_episode(completed, {"evaluation_signature": "wrong"})
            with self.assertRaisesRegex(RuntimeError, "Episode provenance"):
                load_completed_numbered_results(
                    root, profile="pilot", experiment_id="EXP-003"
                )

    def test_report_rejects_backend_topk_key_coverage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = write_completed_run(root)
            rewrite_episode(completed, {"backend": "e5"})
            with self.assertRaisesRegex(RuntimeError, "Episode key coverage"):
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
        merger = (root / "experiments" / "merge_numbered_checkpoint.sh").read_text(
            encoding="utf-8"
        )
        exp006 = (root / "experiments" / "EXP-006" / "run.sh").read_text(
            encoding="utf-8"
        )
        report = (root / "experiments" / "make_report.sh").read_text(encoding="utf-8")

        self.assertIn("validate_checkpoint_artifact", training)
        self.assertIn("Reusing validated completed run", training)
        self.assertIn('validate_checkpoint_artifact "$FINAL_CHECKPOINT"', training)
        self.assertIn("patch_searchr1_worker_cuda.py", training)
        self.assertIn("patch_searchr1_validation.py", training)
        self.assertIn("stackpilot.prepare_hard_rq0", training)
        self.assertIn("--mode backend-id", training)
        self.assertIn("ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}", training)
        self.assertNotIn("fsdp_config.param_offload=true", training)
        self.assertIn("Archived protocol-incompatible completed run", training)
        self.assertIn('"${OUTPUT_DIR}.complete.json"', merger)
        self.assertIn("EXP002_ROOT=", exp006)
        self.assertIn("EXP002_COMPLETE_MARKER=", exp006)
        self.assertIn("--profile", report)

    def test_training_and_evaluation_wrappers_keep_split_roles_separate(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        specialist = (root / "hard_rq0" / "train_specialist.sh").read_text(
            encoding="utf-8"
        )
        mixed = (root / "experiments" / "train_mixed_policy.sh").read_text(
            encoding="utf-8"
        )
        hard_eval = (root / "hard_rq0" / "eval_policy.sh").read_text(encoding="utf-8")
        numbered_eval = (root / "experiments" / "eval_numbered_policy.sh").read_text(
            encoding="utf-8"
        )
        config = (root / "configs" / "hard_rq0.yaml").read_text(encoding="utf-8")

        self.assertIn("searchr1/dev.parquet", specialist)
        self.assertIn("--validate-training-inputs", specialist)
        self.assertIn('--train-file "$TRAIN_DATA"', specialist)
        self.assertIn('--val-file "$VAL_DATA"', specialist)
        self.assertNotIn("searchr1/test.parquet", specialist)

        self.assertIn("searchr1/dev.parquet", mixed)
        self.assertIn("--validate-training-inputs", mixed)
        self.assertIn('--train-file "$TRAIN_DATA"', mixed)
        self.assertIn('--val-file "$VAL_DATA"', mixed)
        self.assertNotIn("searchr1/test.parquet", mixed)

        self.assertIn("data/final_eval.jsonl", hard_eval)
        self.assertIn("data/final_eval.jsonl", numbered_eval)
        self.assertNotIn("data/eval_all.jsonl", hard_eval)
        self.assertNotIn("data/eval_all.jsonl", numbered_eval)

        self.assertIn("trainer_dev_examples_per_dataset:", config)
        self.assertNotIn("validation_examples_per_dataset:", config)

    def test_all_searchr1_training_paths_apply_shared_action_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scripts = {
            name: (root / path).read_text(encoding="utf-8")
            for name, path in {
                "bootstrap": Path("scripts/bootstrap_searchr1.sh"),
                "stage2_all": Path("searchr1_stage2/run_all.sh"),
                "stage2_train": Path("searchr1_stage2/run_single_stack_grpo.sh"),
                "specialist": Path("hard_rq0/train_specialist.sh"),
                "mixed": Path("experiments/train_mixed_policy.sh"),
            }.items()
        }
        for name, text in scripts.items():
            self.assertIn(
                "patch_searchr1_action_protocol.py",
                text,
                f"{name} does not install the shared action parser",
            )
            self.assertIn(
                "patch_searchr1_reward_protocol.py",
                text,
                f"{name} does not install terminal-only reward scoring",
            )
        for name in ("stage2_train", "specialist", "mixed"):
            self.assertIn(
                "$ROOT:$ROOT/hard_rq0:$SEARCH_R1",
                scripts[name],
                f"{name} does not expose stackpilot to Search-R1 workers",
            )
        self.assertIn(
            "SEARCH_R1_REWARD_MODE=${SEARCH_R1_REWARD_MODE:-answer}",
            scripts["specialist"],
        )
        self.assertIn(
            "patch_searchr1_evidence_reward.py",
            scripts["specialist"],
        )
        self.assertIn(
            '"evidence_reward_patch_sha256"',
            scripts["specialist"],
        )
        baseline_driver = (
            root / "hard_rq0" / "run_three_seed_specialists.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "SEARCH_R1_REWARD_MODE=answer",
            baseline_driver,
        )

        preflight = (root / "scripts" / "preflight_searchr1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("STACKPILOT_STRICT_ACTION_PROTOCOL_V2", preflight)
        self.assertIn("STACKPILOT_TERMINAL_REWARD_V2", preflight)
        self.assertIn("resp.split('</search>')", preflight)
        self.assertIn("generation.parse_action is not parse_action", preflight)


if __name__ == "__main__":
    unittest.main()

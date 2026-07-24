from __future__ import annotations

import unittest
from pathlib import Path


class Node2QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.queue = (root / "experiments" / "run_node2_queue.sh").read_text(
            encoding="utf-8"
        )
        cls.watcher = (root / "experiments" / "watch_exp002.sh").read_text(
            encoding="utf-8"
        )

    def test_all_post_exp002_stages_default_on_without_launching_exp002(self) -> None:
        for experiment in ("003", "004", "005", "006"):
            self.assertIn(
                f"RUN_EXP{experiment}=${{RUN_EXP{experiment}:-1}}", self.queue
            )
            self.assertIn(f"experiments/EXP-{experiment}/run.sh", self.queue)
        self.assertIn("RUN_REPORT=${RUN_REPORT:-1}", self.queue)
        self.assertNotIn("hard_rq0/run_all.sh", self.queue)
        self.assertNotIn('bash "$ROOT/scripts/run_full_pipeline.sh"', self.queue)

    def test_queue_owns_lifetime_locks_runtime_and_cleanup(self) -> None:
        for text in (
            '>"$ROOT/work/locks/full-pipeline.lock"',
            "stackpilot-node2-${USER:-unknown}-${HOST_ID}.lock",
            "STACKPILOT_RUNTIME_ROOT",
            "STACKPILOT_LOG_ROOT",
            "SEARCH_R1_ROOT must be private",
            "stop_background_groups",
            "experiments/stop_aux_services.sh",
            "hard_rq0/stop_retrievers.sh",
            "scripts/stop_servers.sh",
            "ray",
            "trap cleanup EXIT",
            "exit 0",
        ):
            self.assertIn(text, self.queue)

    def test_common_setup_and_exp002_read_only_gate_are_explicit(self) -> None:
        self.assertEqual(self.queue.count('bash "$ROOT/scripts/bootstrap.sh"'), 1)
        self.assertEqual(
            self.queue.count('bash "$ROOT/scripts/bootstrap_searchr1.sh"'), 1
        )
        self.assertIn("NUMBERED_SETUP_READY=1", self.queue)
        self.assertIn("EXP002_ARTIFACT_ROOT", self.queue)
        self.assertIn("wait_for_exp002_completion", self.queue)
        self.assertIn('EXP002_ROOT="$EXP002_ROOT"', self.queue)
        self.assertIn("setsid env CUDA_VISIBLE_DEVICES=", self.queue)
        self.assertIn("LOW_PRIORITY=(nice -n 10)", self.queue)
        self.assertIn("LOW_PRIORITY=(ionice -c 3 nice -n 10)", self.queue)
        self.assertIn("DEFAULT_E5_MODEL=intfloat/e5-base-v2", self.queue)
        self.assertIn("scripts/resolve_hf_model.sh", self.queue)
        self.assertIn("No experiment has started.", self.queue)
        self.assertIn("RUN_EXP006=0 RUN_REPORT=0", self.queue)
        self.assertIn("Refusing to mix node-local and external", self.queue)
        self.assertIn("KEEP_VLLM=1 is unsafe", self.queue)
        self.assertIn("validate_run_completion", self.watcher)
        self.assertIn(
            "hard-rq0-{backend}-seed{seed}-{profile}", self.watcher
        )

    def test_vllm_setup_overlaps_only_the_first_mixed_training(self) -> None:
        start = self.queue.index("  start_vllm_bootstrap")
        pretrain = self.queue.index("  pretrain_first_mixed_policy", start + 1)
        wait = self.queue.index("  wait_for_vllm_bootstrap", pretrain + 1)
        probe = self.queue.index("  validate_vllm_hardware", wait + 1)
        exp003 = self.queue.index(
            'bash "$ROOT/experiments/EXP-003/run.sh"', probe + 1
        )
        self.assertLess(start, pretrain)
        self.assertLess(pretrain, wait)
        self.assertLess(wait, probe)
        self.assertLess(probe, exp003)
        self.assertIn("VLLM_DEFER_GPU_PROBE=1", self.queue)
        self.assertIn('OVERLAP_VLLM_SETUP" == 1', self.queue)
        self.assertIn('bash "$ROOT/scripts/bootstrap_vllm.sh"', self.queue)
        self.assertIn('bash "$ROOT/experiments/train_mixed_policy.sh"', self.queue)
        self.assertIn(
            'OVERLAP_VLLM_SETUP" == 1 && "$FORCE_TRAIN" == 1', self.queue
        )
        self.assertIn("OVERLAP_VLLM_SETUP=0", self.queue)

    def test_exp005_restores_the_runtime_timeout_after_each_reset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        exp005 = (root / "experiments" / "EXP-005" / "run.sh").read_text(
            encoding="utf-8"
        )
        reset = exp005.index("reset_searchr1_experiment_files.sh")
        runtime = exp005.index("apply_searchr1_runtime_patch.sh", reset)
        evidence = exp005.index("patch_searchr1_evidence_reward.py", runtime)
        self.assertLess(reset, runtime)
        self.assertLess(runtime, evidence)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import Mock, patch

from hard_rq0 import sitecustomize as seed_hooks
from hard_rq0.patch_searchr1_seed import RAY_IDENTITY_NEW, ROLLOUT_V2


class SeedIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_hooks._SEEDED_IDENTITY = None
        seed_hooks._CUDA_FINALIZED = False

    def tearDown(self) -> None:
        seed_hooks._SEEDED_IDENTITY = None
        seed_hooks._CUDA_FINALIZED = False

    def test_seed_derivation_is_stable_and_role_rank_specific(self) -> None:
        baseline = seed_hooks.derive_seed(13, "actor-rollout", 0)
        self.assertEqual(
            baseline,
            seed_hooks.derive_seed(13, "actor-rollout", 0),
        )
        self.assertNotEqual(
            baseline,
            seed_hooks.derive_seed(13, "reference", 0),
        )
        self.assertNotEqual(
            baseline,
            seed_hooks.derive_seed(13, "actor-rollout", 1),
        )

    def test_experiment_mode_rejects_missing_seed_or_identity(self) -> None:
        clean = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "RQ0_SEED",
                "STACKPILOT_WORKER_ROLE",
                "STACKPILOT_GLOBAL_RANK",
                "RANK",
                "WG_PREFIX",
            }
        }
        clean["STACKPILOT_EXPERIMENT_MODE"] = "1"
        with patch.dict(os.environ, clean, clear=True):
            with self.assertRaisesRegex(RuntimeError, "RQ0_SEED"):
                seed_hooks.seed_process(log=False)
            os.environ["RQ0_SEED"] = "13"
            with self.assertRaisesRegex(RuntimeError, "explicit worker role"):
                seed_hooks.seed_process(log=False)

    def test_cuda_finalization_is_logged_and_applied_once_per_process(self) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(manual_seed_all=Mock())
        )
        environment = {
            "STACKPILOT_EXPERIMENT_MODE": "1",
            "RQ0_SEED": "13",
            "STACKPILOT_WORKER_ROLE": "global_pool",
            "STACKPILOT_GLOBAL_RANK": "2",
        }
        expected = seed_hooks.derive_seed(13, "global_pool", 2)
        seed_hooks._SEEDED_IDENTITY = (13, "global_pool", 2)
        with patch.dict(os.environ, environment, clear=False):
            first = seed_hooks.finalize_worker_cuda_seed(fake_torch)
            second = seed_hooks.finalize_worker_cuda_seed(fake_torch)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        fake_torch.cuda.manual_seed_all.assert_called_once_with(expected)

    def test_searchr1_patch_carries_explicit_identity(self) -> None:
        self.assertIn("STACKPILOT_WORKER_ROLE", RAY_IDENTITY_NEW)
        self.assertIn("STACKPILOT_GLOBAL_RANK", RAY_IDENTITY_NEW)
        self.assertIn("rollout_generation", ROLLOUT_V2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from hard_rq0.patch_searchr1_mixed import (
    MARKER,
    TRAINER_MARKER,
    assign_training_backend_ids,
    patch,
    validate_validation_backend_ids,
)


class MixedGrpoGroupTests(unittest.TestCase):
    def test_training_routes_are_uid_homogeneous_and_group_balanced(self) -> None:
        n_agent = 3
        uids = [
            uid
            for uid in ("q0", "q1", "q2", "q3")
            for _ in range(n_agent)
        ]

        routes = assign_training_backend_ids(
            uids,
            n_agent=n_agent,
            seed=13,
            mixed_step=7,
        )

        self.assertEqual(len(routes), len(uids))
        self.assertEqual(routes.count("bm25"), len(routes) // 2)
        self.assertEqual(routes.count("e5"), len(routes) // 2)
        for start in range(0, len(uids), n_agent):
            self.assertEqual(len(set(uids[start : start + n_agent])), 1)
            self.assertEqual(len(set(routes[start : start + n_agent])), 1)
        self.assertEqual(
            routes,
            assign_training_backend_ids(
                uids,
                n_agent=n_agent,
                seed=13,
                mixed_step=7,
            ),
        )

    def test_training_routes_reject_mixed_uid_group(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not UID-homogeneous"):
            assign_training_backend_ids(
                ["q0", "q1", "q0", "q1"],
                n_agent=2,
                seed=13,
                mixed_step=0,
            )

    def test_training_routes_require_even_source_group_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "even positive number"):
            assign_training_backend_ids(
                ["q0", "q0", "q1", "q1", "q2", "q2"],
                n_agent=2,
                seed=13,
                mixed_step=0,
            )

    def test_validation_routes_are_row_level_not_n_agent_grouped(self) -> None:
        self.assertEqual(
            validate_validation_backend_ids(
                ["q0", "q0", "q1"],
                ["BM25", "e5", "bm25"],
            ),
            ["bm25", "e5", "bm25"],
        )
        with self.assertRaisesRegex(ValueError, "invalid validation"):
            validate_validation_backend_ids(["q0"], ["hybrid"])

    def test_pinned_searchr1_patch_carries_uids_and_hidden_validation_routes(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        source_root = repository / "upstream" / "Search-R1"
        relative_files = (
            "search_r1/llm_agent/generation.py",
            "verl/trainer/main_ppo.py",
            "verl/trainer/ppo/ray_trainer.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in relative_files:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)

            patch(root)
            generation_path = root / relative_files[0]
            trainer_path = root / relative_files[2]
            generation = generation_path.read_text(encoding="utf-8")
            trainer = trainer_path.read_text(encoding="utf-8")

            self.assertIn(MARKER, generation)
            self.assertIn(TRAINER_MARKER, trainer)
            self.assertIn(
                "_stackpilot_assign_training_backend_ids(",
                generation,
            )
            self.assertIn("if self.is_validation:", generation)
            self.assertIn(
                "gen_batch.non_tensor_batch['routing_uid']",
                trainer,
            )
            self.assertIn(
                "test_gen_batch.non_tensor_batch['routing_backend']",
                trainer,
            )
            self.assertNotIn(
                "n_agent % 2",
                generation,
            )
            compile(generation, str(generation_path), "exec")
            compile(trainer, str(trainer_path), "exec")

            before = (generation, trainer)
            patch(root)
            after = (
                generation_path.read_text(encoding="utf-8"),
                trainer_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

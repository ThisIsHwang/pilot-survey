from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelPrefetchTests(unittest.TestCase):
    def test_full_pipeline_excludes_stage2_models_from_hard_prefetch(self) -> None:
        pipeline = (ROOT / "scripts" / "run_full_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'prefetch_future_models.sh" --hard-excluding-stage2', pipeline
        )

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_hard_prefetch_skips_default_model_owned_by_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            pilot_bin = root / ".venv-pilot" / "bin"
            scripts.mkdir(parents=True)
            pilot_bin.mkdir(parents=True)
            shutil.copy2(
                ROOT / "scripts" / "prefetch_future_models.sh",
                scripts / "prefetch_future_models.sh",
            )
            (pilot_bin / "python").write_text("#!/usr/bin/env bash\n", "utf-8")
            (scripts / "resolve_hf_model.sh").write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
printf '%s@%s\\n' "$1" "$2" >> "$ROOT/resolved-models.txt"
""",
                encoding="utf-8",
            )
            os.chmod(pilot_bin / "python", 0o755)
            os.chmod(scripts / "resolve_hf_model.sh", 0o755)

            subprocess.run(
                [
                    shutil.which("bash") or "bash",
                    (scripts / "prefetch_future_models.sh").as_posix(),
                    "--hard-excluding-stage2",
                ],
                check=True,
                cwd=root,
                capture_output=True,
                text=True,
            )

            resolved = (root / "resolved-models.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                resolved,
                [
                    "sentence-transformers/all-MiniLM-L6-v2"
                    "@1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
                ],
            )

            (root / "resolved-models.txt").unlink()
            environment = os.environ.copy()
            environment.update(
                {
                    "HARD_BASE_MODEL_REF": "example/hard-only",
                    "HARD_BASE_MODEL_REVISION": "hard-revision",
                }
            )
            subprocess.run(
                [
                    shutil.which("bash") or "bash",
                    (scripts / "prefetch_future_models.sh").as_posix(),
                    "--hard-excluding-stage2",
                ],
                check=True,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
            )
            resolved = (root / "resolved-models.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                resolved,
                [
                    "example/hard-only@hard-revision",
                    "sentence-transformers/all-MiniLM-L6-v2"
                    "@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                ],
            )

    def test_resolver_serializes_identical_hub_snapshots(self) -> None:
        resolver = (ROOT / "scripts" / "resolve_hf_model.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('exec {MODEL_LOCK_FD}>"$MODEL_LOCK_ROOT/', resolver)
        self.assertIn('flock -n "$MODEL_LOCK_FD"', resolver)
        self.assertIn('flock "$MODEL_LOCK_FD"', resolver)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stackpilot.bootstrap_cache import (
    atomic_write_marker,
    interpreter_compatible,
    interpreter_identity,
    marker_matches,
    parse_pins,
    request_signature,
    signature_request,
    verify_requirements,
)


class BootstrapCacheTests(unittest.TestCase):
    def test_signature_tracks_contents_but_not_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requirements = root / "requirements.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")

            def signature() -> str:
                return request_signature(
                    signature_request(
                        root=root,
                        environment="test",
                        python=Path(sys.executable),
                        inputs=[requirements],
                        values=["backend=cu129"],
                    )
                )

            first = signature()
            stat = requirements.stat()
            os.utime(requirements, (stat.st_atime + 10, stat.st_mtime + 10))
            self.assertEqual(first, signature())

            requirements.write_text("example==2.0\n", encoding="utf-8")
            self.assertNotEqual(first, signature())

    def test_marker_is_atomic_and_rejects_stale_or_corrupt_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "venv" / ".bootstrap.json"
            atomic_write_marker(marker, "pilot", "signature-a")
            self.assertTrue(marker_matches(marker, "pilot", "signature-a"))
            self.assertFalse(marker_matches(marker, "vllm", "signature-a"))
            self.assertFalse(marker_matches(marker, "pilot", "signature-b"))
            self.assertEqual(list(marker.parent.glob("*.tmp")), [])

            marker.write_text("not json", encoding="utf-8")
            self.assertFalse(marker_matches(marker, "pilot", "signature-a"))

    def test_exact_pin_parser_supports_extras_and_rejects_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text(
                "# comment\nuvicorn[standard]==0.51.0\ntorch==2.11.0\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_pins([requirements], ["demo_pkg==1.2.3"]),
                {
                    "demo-pkg": "1.2.3",
                    "torch": "2.11.0",
                    "uvicorn": "0.51.0",
                },
            )
            requirements.write_text("example>=1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact NAME==VERSION"):
                parse_pins([requirements], [])

    def test_requirement_verification_accepts_cuda_local_version(self) -> None:
        with patch.object(
            importlib.metadata,
            "version",
            side_effect=lambda name: {
                "torch": "2.11.0+cu129",
                "vllm": "0.19.0",
            }[name],
        ):
            self.assertEqual(
                verify_requirements([], ["torch==2.11.0", "vllm==0.19.0"]),
                {"torch": "2.11.0+cu129", "vllm": "0.19.0"},
            )

    def test_interpreter_compatibility_rejects_a_missing_environment(self) -> None:
        current = Path(sys.executable)
        base = Path(interpreter_identity(current)["base_executable"])
        self.assertTrue(interpreter_compatible(current, base))
        self.assertFalse(
            interpreter_compatible(current.with_name("missing-python"), base)
        )

    def test_interpreter_compatibility_uses_abi_not_base_path(self) -> None:
        environment = {
            "version": [3, 12, 8],
            "implementation": "cpython",
            "soabi": "cpython-312-x86_64-linux-gnu",
            "system": "Linux",
            "machine": "x86_64",
            "base_executable": "/node-a/python3.12",
        }
        base = {
            **environment,
            "version": [3, 12, 12],
            "executable": "/node-b/python3.12",
        }
        with patch(
            "stackpilot.bootstrap_cache.interpreter_identity",
            side_effect=[environment, base],
        ):
            self.assertTrue(
                interpreter_compatible(Path("venv-python"), Path("base-python"))
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hard_rq0.patch_searchr1_worker_cuda import LEGACY, LEGACY_MARKER, MARKER, NEW, OLD
from hard_rq0.patch_searchr1_worker_cuda import patch as patch_worker_cuda
from hard_rq0.sitecustomize import derive_seed

ROOT = Path(__file__).resolve().parents[1]
SITECUSTOMIZE = ROOT / "hard_rq0" / "sitecustomize.py"


class _SeedRecorder:
    def __init__(self) -> None:
        self.values: list[int] = []

    def seed(self, value: int) -> None:
        self.values.append(value)

    def manual_seed(self, value: int) -> None:
        self.values.append(value)


class _PoisonCuda:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"sitecustomize touched torch.cuda.{name}")


class SiteCustomizeTests(unittest.TestCase):
    def test_seeds_cpu_rngs_without_touching_cuda(self) -> None:
        numpy_seed = _SeedRecorder()
        torch_seed = _SeedRecorder()
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.random = numpy_seed
        fake_torch = types.ModuleType("torch")
        fake_torch.default_generator = torch_seed
        fake_torch.cuda = _PoisonCuda()

        with (
            patch.dict(
                os.environ,
                {
                    "RQ0_SEED": "41",
                    "STACKPILOT_WORKER_ROLE": "test-role",
                    "STACKPILOT_GLOBAL_RANK": "3",
                },
                clear=False,
            ),
            patch.dict(
                sys.modules,
                {"numpy": fake_numpy, "torch": fake_torch},
            ),
            patch("random.seed") as python_seed,
        ):
            runpy.run_path(
                str(SITECUSTOMIZE), run_name="_stackpilot_sitecustomize_test"
            )

        expected = derive_seed(41, "test-role", 3)
        python_seed.assert_called_once_with(expected)
        self.assertEqual(numpy_seed.values, [expected])
        self.assertEqual(torch_seed.values, [expected])


class _FakeCuda:
    def __init__(self, events: list[object], *, initialized: bool, count: int) -> None:
        self.events = events
        self.initialized = initialized
        self.count = count

    def is_initialized(self) -> bool:
        self.events.append("cuda.is_initialized")
        return self.initialized

    def device_count(self) -> int:
        self.events.append("cuda.device_count")
        return self.count

    def set_device(self, device: int) -> None:
        self.events.append(("cuda.set_device", device))

    def manual_seed_all(self, seed: int) -> None:
        self.events.append(("cuda.manual_seed_all", seed))


class _FakeTorchC:
    def __init__(self, events: list[object], runtime_count: int) -> None:
        self.events = events
        self.runtime_count = runtime_count

    def _cuda_getDeviceCount(self) -> int:
        self.events.append("torch._C._cuda_getDeviceCount")
        return self.runtime_count


class _FakeDistributed(types.ModuleType):
    def __init__(self, events: list[object]) -> None:
        super().__init__("torch.distributed")
        self.events = events

    def is_initialized(self) -> bool:
        self.events.append("dist.is_initialized")
        return False

    def init_process_group(self, *, backend: str) -> None:
        self.events.append(("dist.init_process_group", backend))


class WorkerCudaPatchTests(unittest.TestCase):
    def make_checkout(
        self, source: str = OLD
    ) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        target = root / "verl" / "workers" / "fsdp_workers.py"
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
        return temporary, root

    def test_patch_is_idempotent_and_fail_closed(self) -> None:
        temporary, root = self.make_checkout()
        self.addCleanup(temporary.cleanup)

        target = patch_worker_cuda(root)
        first = target.read_text(encoding="utf-8")
        self.assertEqual(first, NEW)
        self.assertEqual(first.count(MARKER), 1)
        self.assertEqual(patch_worker_cuda(root), target)
        self.assertEqual(target.read_text(encoding="utf-8"), first)

        broken_temporary, broken_root = self.make_checkout(
            OLD.replace("self.config = config", "self.settings = config")
        )
        self.addCleanup(broken_temporary.cleanup)
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            patch_worker_cuda(broken_root)

    def test_upgrades_legacy_strict_initialization_guard(self) -> None:
        temporary, root = self.make_checkout(LEGACY)
        self.addCleanup(temporary.cleanup)

        target = patch_worker_cuda(root)
        patched = target.read_text(encoding="utf-8")
        self.assertEqual(patched, NEW)
        self.assertNotIn(LEGACY_MARKER, patched)
        self.assertEqual(patched.count(MARKER), 1)

    def test_patch_applies_to_the_pinned_searchr1_worker(self) -> None:
        source = (
            ROOT / "upstream" / "Search-R1" / "verl" / "workers" / "fsdp_workers.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            target = checkout / "verl" / "workers" / "fsdp_workers.py"
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)

            self.assertEqual(patch_worker_cuda(checkout), target)
            patched = target.read_text(encoding="utf-8")
            self.assertEqual(patched.count(MARKER), 1)
            compile(patched, str(target), "exec")
            self.assertEqual(patch_worker_cuda(checkout), target)

    def execute_patched_worker(
        self,
        *,
        initialized: bool = False,
        count: int = 1,
        runtime_count: int = 1,
        visible_devices: str = "4",
    ) -> tuple[list[object], type]:
        temporary, root = self.make_checkout(
            """\
import os
import torch

class Worker:
    def __init__(self):
        self.rank = 2

class ActorRolloutRefWorker(Worker):
"""
            + OLD
            + """\
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
"""
        )
        self.addCleanup(temporary.cleanup)
        target = patch_worker_cuda(root)

        events: list[object] = []
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = _FakeCuda(events, initialized=initialized, count=count)
        fake_torch._C = _FakeTorchC(events, runtime_count)
        fake_distributed = _FakeDistributed(events)
        fake_torch.distributed = fake_distributed
        fake_sitecustomize = types.ModuleType("sitecustomize")
        effective_seed = derive_seed(13, "test-worker", 2)

        def finalize_worker_cuda_seed(torch_module: object) -> int:
            torch_module.cuda.manual_seed_all(effective_seed)
            return effective_seed

        fake_sitecustomize.finalize_worker_cuda_seed = finalize_worker_cuda_seed
        namespace: dict[str, object] = {
            "DictConfig": object,
        }
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                    "RQ0_SEED": "13",
                    "STACKPILOT_EXPERIMENT_MODE": "1",
                    "STACKPILOT_WORKER_ROLE": "test-worker",
                    "STACKPILOT_GLOBAL_RANK": "2",
                },
                clear=False,
            ),
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.distributed": fake_distributed,
                    "sitecustomize": fake_sitecustomize,
                },
            ),
        ):
            exec(  # noqa: S102 - execute the generated patch fixture
                compile(target.read_text(encoding="utf-8"), str(target), "exec"),
                namespace,
            )
            worker_type = namespace["ActorRolloutRefWorker"]
            try:
                worker_type(config=object(), role="ref")
            except RuntimeError as error:
                return events, type(error)
        return events, type(None)

    def test_configures_one_gpu_and_seed_before_distributed_init(self) -> None:
        events, error_type = self.execute_patched_worker()
        self.assertIs(error_type, type(None))
        self.assertEqual(
            events,
            [
                "torch._C._cuda_getDeviceCount",
                "cuda.device_count",
                ("cuda.set_device", 0),
                ("cuda.manual_seed_all", derive_seed(13, "test-worker", 2)),
                "dist.is_initialized",
                ("dist.init_process_group", "nccl"),
            ],
        )

    def test_accepts_cuda_initialized_after_ray_applied_single_gpu_mask(self) -> None:
        events, error_type = self.execute_patched_worker(initialized=True)
        self.assertIs(error_type, type(None))
        self.assertEqual(
            events,
            [
                "torch._C._cuda_getDeviceCount",
                "cuda.device_count",
                ("cuda.set_device", 0),
                ("cuda.manual_seed_all", derive_seed(13, "test-worker", 2)),
                "dist.is_initialized",
                ("dist.init_process_group", "nccl"),
            ],
        )

    def test_rejects_more_than_one_ray_visible_gpu(self) -> None:
        events, error_type = self.execute_patched_worker(visible_devices="0,1")
        self.assertIs(error_type, RuntimeError)
        self.assertEqual(events, [])

    def test_rejects_pre_ray_cuda_context_that_ignores_ray_mask(self) -> None:
        events, error_type = self.execute_patched_worker(
            initialized=True,
            runtime_count=7,
        )
        self.assertIs(error_type, RuntimeError)
        self.assertEqual(
            events,
            [
                "torch._C._cuda_getDeviceCount",
                "cuda.device_count",
            ],
        )


if __name__ == "__main__":
    unittest.main()

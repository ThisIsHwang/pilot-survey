from __future__ import annotations

import argparse
from pathlib import Path

LEGACY_MARKER = "# STACKPILOT_RAY_WORKER_CUDA_SCOPE_V1"
MARKER = "# STACKPILOT_RAY_WORKER_CUDA_SCOPE_V2"
OLD = """\
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
"""
LEGACY = """\
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        # STACKPILOT_RAY_WORKER_CUDA_SCOPE_V1
        # sitecustomize runs before Ray applies this actor's GPU mask.  CUDA
        # must remain lazy until here or every FSDP rank can retain the
        # launcher's full device visibility and enter collectives on GPU 0.
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was initialized before Ray configured this FSDP worker; "
                "startup hooks must not call CUDA APIs"
            )
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_device_ids = (
            []
            if visible_devices is None
            else [item.strip() for item in visible_devices.split(",") if item.strip()]
        )
        if len(visible_device_ids) != 1:
            raise RuntimeError(
                "Each FSDP worker must receive exactly one Ray-assigned GPU; "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
            )
        # torch.cuda.is_available() can initialize the CUDA runtime without
        # setting torch.cuda.is_initialized().  Query the runtime directly so
        # a pre-Ray call that cached the launcher's full GPU mask fails here.
        runtime_device_count = torch._C._cuda_getDeviceCount()
        device_count = torch.cuda.device_count()
        if runtime_device_count != 1 or device_count != 1:
            raise RuntimeError(
                "Each FSDP worker must see exactly one CUDA device after Ray "
                f"assignment; CUDA_VISIBLE_DEVICES={visible_devices!r}, "
                f"CUDA runtime device count={runtime_device_count}, "
                f"torch.cuda.device_count()={device_count}"
            )
        torch.cuda.set_device(0)
        seed_text = os.environ.get("RQ0_SEED")
        if seed_text is not None:
            torch.cuda.manual_seed_all(int(seed_text) + self.rank)
"""
NEW = """\
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        # STACKPILOT_RAY_WORKER_CUDA_SCOPE_V2
        # Ray installs the actor's CUDA_VISIBLE_DEVICES before deserializing
        # its class.  Importing veRL's FSDP module can then import vLLM and
        # initialize CUDA before this constructor runs.  That is safe when the
        # initialized runtime sees exactly the one GPU assigned by Ray.  A
        # CUDA context created too early (for example, by sitecustomize before
        # Ray's mask) retains the launcher's wider device count and is rejected
        # by the runtime-count check below.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_device_ids = (
            []
            if visible_devices is None
            else [item.strip() for item in visible_devices.split(",") if item.strip()]
        )
        if len(visible_device_ids) != 1:
            raise RuntimeError(
                "Each FSDP worker must receive exactly one Ray-assigned GPU; "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
            )
        # Query the CUDA runtime directly as well as PyTorch's public count.
        # This remains effective whether a dependency initialized CUDA during
        # actor deserialization or CUDA is still lazy at this point.
        runtime_device_count = torch._C._cuda_getDeviceCount()
        device_count = torch.cuda.device_count()
        if runtime_device_count != 1 or device_count != 1:
            raise RuntimeError(
                "Each FSDP worker must see exactly one CUDA device after Ray "
                f"assignment; CUDA_VISIBLE_DEVICES={visible_devices!r}, "
                f"CUDA runtime device count={runtime_device_count}, "
                f"torch.cuda.device_count()={device_count}"
            )
        torch.cuda.set_device(0)
        seed_text = os.environ.get("RQ0_SEED")
        if seed_text is not None:
            torch.cuda.manual_seed_all(int(seed_text) + self.rank)
"""


def patch(search_r1_root: Path) -> Path:
    target = search_r1_root / "verl" / "workers" / "fsdp_workers.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        if NEW not in text:
            raise RuntimeError(
                f"Search-R1 worker CUDA marker is present but its verified block "
                f"is incomplete in {target}"
            )
        return target
    old_block = LEGACY if LEGACY_MARKER in text else OLD
    occurrences = text.count(old_block)
    if occurrences != 1:
        raise RuntimeError(
            "Pinned Search-R1 ActorRolloutRefWorker constructor was not found "
            f"exactly once in {target}; found {occurrences}. Refusing an "
            "unverified patch."
        )
    target.write_text(text.replace(old_block, NEW, 1), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    target = patch(Path(args.search_r1_root).resolve())
    print(f"Search-R1 Ray-worker CUDA scoping patch ready: {target}")


if __name__ == "__main__":
    main()

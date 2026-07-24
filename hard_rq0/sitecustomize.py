"""Install deterministic, role-aware process seeding for Search-R1."""

from __future__ import annotations

import hashlib
import os
import random
import sys

_SEEDED_IDENTITY: tuple[int, str, int] | None = None
_CUDA_FINALIZED = False


def derive_seed(base_seed: int, role: str, global_rank: int) -> int:
    identity = f"{base_seed}\0{role}\0{global_rank}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def seed_process(*, log: bool = True) -> int | None:
    global _SEEDED_IDENTITY
    strict = os.environ.get("STACKPILOT_EXPERIMENT_MODE") == "1"
    seed_text = os.environ.get("RQ0_SEED")
    if seed_text is None:
        if strict:
            raise RuntimeError("RQ0_SEED is required in experiment mode")
        return None
    role = os.environ.get("STACKPILOT_WORKER_ROLE") or os.environ.get("WG_PREFIX")
    rank_text = os.environ.get("STACKPILOT_GLOBAL_RANK") or os.environ.get("RANK")
    if role is None or rank_text is None:
        message = "Experiment seeding requires an explicit worker role and global rank"
        if strict:
            raise RuntimeError(message)
        role = role or "legacy-process"
        rank_text = rank_text or os.environ.get("LOCAL_RANK", "0")
    try:
        base_seed = int(seed_text)
        global_rank = int(rank_text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid seed identity: RQ0_SEED={seed_text!r}, "
            f"role={role!r}, global_rank={rank_text!r}"
        ) from error
    identity = (base_seed, str(role), global_rank)
    seed = derive_seed(base_seed, str(role), global_rank)
    if _SEEDED_IDENTITY is not None:
        if _SEEDED_IDENTITY != identity:
            raise RuntimeError(
                "Process seed identity changed after initialization: "
                f"{_SEEDED_IDENTITY!r} != {identity!r}"
            )
        return seed
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        if strict:
            raise
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        if strict:
            raise
    else:
        # Keep startup strictly CPU-only. Ray applies the actor GPU mask before
        # ActorRolloutRefWorker finalizes this same seed on CUDA.
        torch.default_generator.manual_seed(seed)
    os.environ["STACKPILOT_WORKER_ROLE"] = str(role)
    os.environ["STACKPILOT_GLOBAL_RANK"] = str(global_rank)
    os.environ["STACKPILOT_EFFECTIVE_SEED"] = str(seed)
    _SEEDED_IDENTITY = identity
    if log:
        print(
            "StackPilot seed initialized: "
            f"role={role} global_rank={global_rank} seed={seed} cuda=0",
            file=sys.stderr,
            flush=True,
        )
    return seed


def finalize_worker_cuda_seed(torch_module: object) -> int:
    """Seed one Ray worker CUDA context and emit its single final seed log."""
    global _CUDA_FINALIZED
    seed = seed_process(log=False)
    if seed is None:
        raise RuntimeError("RQ0_SEED is required for experiment worker seeding")
    if _CUDA_FINALIZED:
        return seed
    torch_module.cuda.manual_seed_all(seed)
    _CUDA_FINALIZED = True
    print(
        "StackPilot seed initialized: "
        f"role={os.environ['STACKPILOT_WORKER_ROLE']} "
        f"global_rank={os.environ['STACKPILOT_GLOBAL_RANK']} "
        f"seed={seed} cuda=1",
        file=sys.stderr,
        flush=True,
    )
    return seed


_is_ray_worker = "WG_PREFIX" in os.environ and "RANK" in os.environ
seed_process(log=not _is_ray_worker)

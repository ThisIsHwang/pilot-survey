from __future__ import annotations

import argparse
from pathlib import Path

ROLLOUT_OLD = (
    "torch.cuda.manual_seed(gen_dp_rank + 1000)  "
    "# make sure all tp ranks have the same random states"
)
ROLLOUT_V1 = (
    "seed_base = int(os.getenv('RQ0_SEED', '0'))\n"
    "            torch.cuda.manual_seed(seed_base + gen_dp_rank + 1000)  "
    "# experiment-aware rollout seed"
)
ROLLOUT_V2 = (
    "seed_text = os.getenv('RQ0_SEED')\n"
    "            if seed_text is None:\n"
    "                generation_seed = gen_dp_rank + 1000\n"
    "            else:\n"
    "                from sitecustomize import derive_seed\n"
    "                generation_seed = derive_seed(\n"
    "                    int(seed_text), 'rollout_generation', gen_dp_rank\n"
    "                )\n"
    "            torch.cuda.manual_seed(generation_seed)  "
    "# role-aware rollout seed"
)

RAY_IDENTITY_MARKER = "# STACKPILOT_RAY_SEED_IDENTITY_V1"
RAY_IDENTITY_OLD = "                    'RAY_LOCAL_RANK': str(local_rank),\n"
RAY_IDENTITY_NEW = (
    RAY_IDENTITY_OLD
    + "                    # STACKPILOT_RAY_SEED_IDENTITY_V1\n"
    + "                    'STACKPILOT_WORKER_ROLE': str(self.name_prefix),\n"
    + "                    'STACKPILOT_GLOBAL_RANK': str(rank),\n"
)


def patch_rollout_seed(search_r1_root: Path) -> Path:
    path = search_r1_root / "verl" / "workers" / "sharding_manager" / "fsdp_vllm.py"
    text = path.read_text(encoding="utf-8")
    if ROLLOUT_V2 in text:
        return path
    old = ROLLOUT_V1 if ROLLOUT_V1 in text else ROLLOUT_OLD
    if text.count(old) != 1:
        raise RuntimeError(
            f"Pinned Search-R1 rollout seed line was not found exactly once in {path}"
        )
    path.write_text(text.replace(old, ROLLOUT_V2, 1), encoding="utf-8")
    return path


def patch_ray_worker_identity(search_r1_root: Path) -> Path:
    path = search_r1_root / "verl" / "single_controller" / "ray" / "base.py"
    text = path.read_text(encoding="utf-8")
    if RAY_IDENTITY_MARKER in text:
        if RAY_IDENTITY_NEW not in text:
            raise RuntimeError(f"Ray seed identity marker is incomplete in {path}")
        return path
    if text.count(RAY_IDENTITY_OLD) != 1:
        raise RuntimeError(
            f"Pinned Ray worker runtime_env block was not found exactly once in {path}"
        )
    path.write_text(
        text.replace(RAY_IDENTITY_OLD, RAY_IDENTITY_NEW, 1),
        encoding="utf-8",
    )
    return path


def patch(search_r1_root: Path) -> tuple[Path, Path]:
    return (
        patch_rollout_seed(search_r1_root),
        patch_ray_worker_identity(search_r1_root),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    rollout, ray_base = patch(Path(args.search_r1_root).resolve())
    print(f"Search-R1 role-aware rollout seed patch ready: {rollout}")
    print(f"Search-R1 explicit Ray worker identity patch ready: {ray_base}")


if __name__ == "__main__":
    main()

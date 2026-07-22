from __future__ import annotations

import argparse
from pathlib import Path

OLD = "torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states"
NEW = (
    "seed_base = int(os.getenv('RQ0_SEED', '0'))\n"
    "            torch.cuda.manual_seed(seed_base + gen_dp_rank + 1000)  # experiment-aware rollout seed"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    path = (
        Path(args.search_r1_root)
        / "verl"
        / "workers"
        / "sharding_manager"
        / "fsdp_vllm.py"
    )
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"Search-R1 seed patch already applied: {path}")
        return
    if OLD not in text:
        raise RuntimeError(
            f"Pinned Search-R1 seed line was not found in {path}; refusing an unverified patch"
        )
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Applied experiment-aware Search-R1 rollout seed patch: {path}")


if __name__ == "__main__":
    main()

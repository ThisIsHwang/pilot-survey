from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stackpilot.common import read_jsonl_tolerant

SPECIALIST_TAGS = ("bm25-specialist", "e5-specialist")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/hard_rq0/runs/pilot/results/policies")
    parser.add_argument("--seeds", nargs="+", type=int, default=(13, 42, 87))
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.jsonl")):
        rows.extend(read_jsonl_tolerant(path))
    if not rows:
        raise RuntimeError(f"No hard-RQ0 results found under {args.results_dir}")
    frame = pd.DataFrame(rows)
    required = {
        "policy_tag",
        "seed",
        "run_signature",
        "question_id",
        "dataset",
        "backend",
        "topk",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing result columns: {sorted(missing)}")

    duplicates = frame.duplicated(
        ["policy_tag", "seed", "question_id", "dataset", "backend", "topk"],
        keep=False,
    )
    if duplicates.any():
        sample = frame.loc[duplicates, ["policy_tag", "seed", "question_id", "backend", "topk"]].head(20)
        raise RuntimeError(f"Duplicate episode rows detected:\n{sample.to_string(index=False)}")

    signature_counts = frame.groupby(["policy_tag", "seed"])["run_signature"].nunique()
    mixed = signature_counts[signature_counts != 1]
    if not mixed.empty:
        raise RuntimeError(
            "A policy/seed mixes multiple model or configuration signatures. "
            f"Use a fresh RESULT_SET. Mixed groups: {mixed.to_dict()}"
        )

    base_seeds = set(
        int(value)
        for value in frame.loc[frame["policy_tag"] == "base-qwen", "seed"].unique()
    )
    if base_seeds != {0}:
        raise RuntimeError(f"Expected deterministic base-qwen seed 0 only, found {sorted(base_seeds)}")
    expected_seeds = set(args.seeds)
    for tag in SPECIALIST_TAGS:
        observed = set(int(value) for value in frame.loc[frame["policy_tag"] == tag, "seed"].unique())
        if observed != expected_seeds:
            raise RuntimeError(
                f"{tag} must contain exactly seeds {sorted(expected_seeds)}, found {sorted(observed)}"
            )

    key_columns = ["question_id", "dataset", "backend", "topk"]
    reference = set(
        map(tuple, frame[frame["policy_tag"] == "base-qwen"][key_columns].itertuples(index=False, name=None))
    )
    if not reference:
        raise RuntimeError("Base Qwen result set is empty")
    for (tag, seed), group in frame.groupby(["policy_tag", "seed"]):
        observed = set(map(tuple, group[key_columns].itertuples(index=False, name=None)))
        if observed != reference:
            missing_keys = list(reference - observed)[:10]
            extra_keys = list(observed - reference)[:10]
            raise RuntimeError(
                f"{tag}/seed{seed} uses a different evaluation set; "
                f"missing={missing_keys}, extra={extra_keys}"
            )

    print(
        f"Validated {len(frame):,} rows over {len(reference):,} question/backend/top-k units "
        f"for base Qwen and {len(SPECIALIST_TAGS)} x {len(expected_seeds)} specialists."
    )


if __name__ == "__main__":
    main()

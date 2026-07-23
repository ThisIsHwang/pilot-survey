from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from stackpilot.common import read_jsonl_tolerant
from stackpilot.hard_rq0_contract import (
    BASE_POLICY_TAG,
    DEFAULT_DATASETS,
    DEFAULT_TOPKS,
    METRICS,
    POLICY_TAGS,
    RESULT_SCHEMA,
    RETRIEVER_BACKENDS,
    SPECIALIST_TAGS,
    episode_validation_error,
)

KEY_COLUMNS = ["question_id", "dataset", "backend", "topk"]
BOUNDED_METRICS = [*METRICS]
BINARY_METRICS = [
    "em",
    "recovery_at_2",
    "recovery_at_3",
    "full_recovery_at_2",
    "full_recovery_at_3",
]


def validate_expected_values(
    seeds: Sequence[int],
    backends: Sequence[str],
    topks: Sequence[int],
    datasets: Sequence[str],
) -> None:
    for name, values in (
        ("seeds", seeds),
        ("backends", backends),
        ("topks", topks),
        ("datasets", datasets),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(
                f"Expected {name} must be nonempty and unique: {list(values)}"
            )
    if set(backends) != set(RETRIEVER_BACKENDS):
        raise ValueError(
            f"Hard-RQ0 requires exactly backends {list(RETRIEVER_BACKENDS)}; "
            f"got {list(backends)}"
        )
    if any(isinstance(value, bool) or value < 1 for value in seeds):
        raise ValueError(
            f"Expected specialist seeds must be positive integers: {list(seeds)}"
        )
    if any(isinstance(value, bool) or value < 1 for value in topks):
        raise ValueError(
            f"Expected top-k values must be positive integers: {list(topks)}"
        )


def integer_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    array = values.to_numpy(dtype=float)
    invalid = ~np.isfinite(array) | (array != np.floor(array))
    if invalid.any():
        examples = frame.loc[invalid, column].head(10).tolist()
        raise RuntimeError(
            f"{column} must contain finite integers; examples={examples}"
        )
    return values.astype(int)


def validate_frame(
    frame: pd.DataFrame,
    expected_seeds: Sequence[int] = (13, 42, 87),
    expected_backends: Sequence[str] = RETRIEVER_BACKENDS,
    expected_topks: Sequence[int] = DEFAULT_TOPKS,
    expected_datasets: Sequence[str] = DEFAULT_DATASETS,
    max_search_turns: int = 4,
) -> tuple[pd.DataFrame, int, str]:
    validate_expected_values(
        expected_seeds, expected_backends, expected_topks, expected_datasets
    )
    if max_search_turns < 1:
        raise ValueError(f"max_search_turns must be positive; got {max_search_turns}")
    required = {
        "schema",
        "policy_tag",
        "seed",
        "run_signature",
        "evaluation_signature",
        *KEY_COLUMNS,
        *METRICS,
        "search_count",
        "question",
        "answers",
        "queries",
        "turns",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing result columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("Hard-RQ0 result set is empty")
    frame = frame.copy()
    for row in frame.to_dict(orient="records"):
        problem = episode_validation_error(row, max_search_turns)
        if problem is not None:
            raise RuntimeError(
                f"Invalid episode {row.get('policy_tag')}/seed{row.get('seed')}/"
                f"{row.get('question_id')}/{row.get('backend')}/k={row.get('topk')}: "
                f"{problem}"
            )

    frame["schema"] = integer_column(frame, "schema")
    schemas = set(frame["schema"])
    if schemas != {RESULT_SCHEMA}:
        raise RuntimeError(
            f"Expected result schema {RESULT_SCHEMA} only, found {sorted(schemas)}"
        )
    frame["seed"] = integer_column(frame, "seed")
    frame["topk"] = integer_column(frame, "topk")
    for column in ("policy_tag", "question_id", "dataset", "backend"):
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise RuntimeError(f"Result column {column} contains empty values")
        frame[column] = frame[column].astype(str)

    observed_tags = set(frame["policy_tag"])
    if observed_tags != set(POLICY_TAGS):
        raise RuntimeError(
            f"Expected policy tags {list(POLICY_TAGS)}, found {sorted(observed_tags)}"
        )
    observed_backends = set(frame["backend"])
    if observed_backends != set(expected_backends):
        raise RuntimeError(
            f"Expected backends {list(expected_backends)}, found {sorted(observed_backends)}"
        )
    observed_topks = set(frame["topk"])
    if observed_topks != set(expected_topks):
        raise RuntimeError(
            f"Expected top-k values {list(expected_topks)}, found {sorted(observed_topks)}"
        )
    observed_datasets = set(frame["dataset"])
    if observed_datasets != set(expected_datasets):
        raise RuntimeError(
            f"Expected datasets {list(expected_datasets)}, found {sorted(observed_datasets)}"
        )

    for column in ("run_signature", "evaluation_signature"):
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise RuntimeError(f"Result column {column} contains empty values")
        frame[column] = frame[column].astype(str)
    evaluation_signatures = set(frame["evaluation_signature"])
    if len(evaluation_signatures) != 1:
        raise RuntimeError(
            "Every policy/seed must share one evaluation signature; found "
            f"{sorted(evaluation_signatures)}"
        )
    evaluation_signature = next(iter(evaluation_signatures))
    signature_counts = frame.groupby(["policy_tag", "seed"])["run_signature"].nunique()
    mixed = signature_counts[signature_counts != 1]
    if not mixed.empty:
        raise RuntimeError(
            f"A policy/seed mixes multiple run signatures: {mixed.to_dict()}"
        )

    base_seeds = set(
        frame.loc[frame["policy_tag"] == BASE_POLICY_TAG, "seed"].astype(int)
    )
    if base_seeds != {0}:
        raise RuntimeError(
            f"Expected deterministic {BASE_POLICY_TAG} seed 0 only, found {sorted(base_seeds)}"
        )
    expected_seed_set = set(expected_seeds)
    for tag in SPECIALIST_TAGS:
        observed = set(frame.loc[frame["policy_tag"] == tag, "seed"].astype(int))
        if observed != expected_seed_set:
            raise RuntimeError(
                f"{tag} must contain exactly seeds {sorted(expected_seed_set)}, "
                f"found {sorted(observed)}"
            )

    duplicates = frame.duplicated(["policy_tag", "seed", *KEY_COLUMNS], keep=False)
    if duplicates.any():
        columns = ["policy_tag", "seed", *KEY_COLUMNS]
        sample = frame.loc[duplicates, columns].head(20)
        raise RuntimeError(
            f"Duplicate episode rows detected:\n{sample.to_string(index=False)}"
        )
    dataset_counts = frame.groupby("question_id")["dataset"].nunique()
    if (dataset_counts != 1).any():
        bad = dataset_counts[dataset_counts != 1].head(10).to_dict()
        raise RuntimeError(f"Question IDs map to multiple datasets: {bad}")

    for metric in (*BOUNDED_METRICS, "search_count"):
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad = frame.loc[~np.isfinite(values), metric].head(10).tolist()
            raise RuntimeError(f"Metric {metric} contains non-finite values: {bad}")
        frame[metric] = numeric.astype(float)
    for metric in BOUNDED_METRICS:
        invalid = (frame[metric] < 0.0) | (frame[metric] > 1.0)
        if invalid.any():
            raise RuntimeError(
                f"Metric {metric} must be in [0, 1]; "
                f"examples={frame.loc[invalid, metric].head(10).tolist()}"
            )
    for metric in BINARY_METRICS:
        invalid = ~frame[metric].isin((0.0, 1.0))
        if invalid.any():
            raise RuntimeError(
                f"Metric {metric} must be binary; "
                f"examples={frame.loc[invalid, metric].head(10).tolist()}"
            )
    invalid_search_count = (
        (frame["search_count"] < 0)
        | (frame["search_count"] > max_search_turns)
        | (frame["search_count"] != np.floor(frame["search_count"]))
    )
    if invalid_search_count.any():
        raise RuntimeError(
            f"search_count must be an integer in [0, {max_search_turns}]; "
            f"examples={frame.loc[invalid_search_count, 'search_count'].head(10).tolist()}"
        )
    tolerance = 1e-9
    monotonic = (
        (frame["turn1_support_recall"] <= frame["turn2_support_recall"] + tolerance)
        & (frame["turn2_support_recall"] <= frame["turn3_support_recall"] + tolerance)
        & (frame["turn3_support_recall"] <= frame["support_recall"] + tolerance)
    )
    if not monotonic.all():
        raise RuntimeError("Support recall must be monotonic across search turns")

    reference_frame = frame[frame["policy_tag"] == BASE_POLICY_TAG]
    question_units = set(
        map(
            tuple,
            reference_frame[["question_id", "dataset"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    expected_cells = {
        (question_id, dataset, backend, int(topk))
        for question_id, dataset in question_units
        for backend in expected_backends
        for topk in expected_topks
    }
    reference = set(
        map(tuple, reference_frame[KEY_COLUMNS].itertuples(index=False, name=None))
    )
    if reference != expected_cells:
        raise RuntimeError(
            "Base Qwen does not contain the exact question x backend x top-k grid; "
            f"missing={list(expected_cells - reference)[:10]}, "
            f"extra={list(reference - expected_cells)[:10]}"
        )
    for (tag, seed), group in frame.groupby(["policy_tag", "seed"]):
        observed = set(
            map(tuple, group[KEY_COLUMNS].itertuples(index=False, name=None))
        )
        if observed != reference:
            raise RuntimeError(
                f"{tag}/seed{seed} uses a different evaluation grid; "
                f"missing={list(reference - observed)[:10]}, "
                f"extra={list(observed - reference)[:10]}"
            )
    return frame, len(reference), evaluation_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="work/hard_rq0/runs/pilot/results/policies"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(13, 42, 87))
    parser.add_argument(
        "--backends", nargs="+", choices=RETRIEVER_BACKENDS, default=RETRIEVER_BACKENDS
    )
    parser.add_argument("--topks", nargs="+", type=int, default=DEFAULT_TOPKS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-search-turns", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.jsonl")):
        rows.extend(read_jsonl_tolerant(path))
    if not rows:
        raise RuntimeError(f"No hard-RQ0 results found under {args.results_dir}")
    _, reference_size, evaluation_signature = validate_frame(
        pd.DataFrame(rows),
        expected_seeds=args.seeds,
        expected_backends=args.backends,
        expected_topks=args.topks,
        expected_datasets=args.datasets,
        max_search_turns=args.max_search_turns,
    )
    print(
        f"Validated {len(rows):,} rows over {reference_size:,} "
        "question/backend/top-k units for base Qwen and "
        f"{len(SPECIALIST_TAGS)} x {len(set(args.seeds))} specialists "
        f"(evaluation signature {evaluation_signature[:12]})."
    )


if __name__ == "__main__":
    main()

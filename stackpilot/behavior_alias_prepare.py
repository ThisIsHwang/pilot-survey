from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.behavior_alias_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    balanced_trim,
    canonical_signature,
    file_sha256,
    load_config,
    read_jsonl,
)


def validate_state(row: dict[str, Any]) -> None:
    required = {
        "state_id",
        "question_id",
        "question",
        "dataset",
        "backend",
        "topk",
        "source_turn",
        "support_titles",
        "prior_turns",
        "prefix_support_recall",
    }
    missing = required - set(row)
    if missing:
        raise RuntimeError(f"Causal-audit state is missing fields: {sorted(missing)}")
    if row["backend"] not in {"bm25", "e5"}:
        raise RuntimeError(f"Unsupported backend in state {row['state_id']}: {row['backend']}")
    if not isinstance(row["prior_turns"], list):
        raise RuntimeError(f"State {row['state_id']} prior_turns must be a list")
    if not isinstance(row["support_titles"], list) or not row["support_titles"]:
        raise RuntimeError(f"State {row['state_id']} needs nonempty support_titles")


def select_states(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    count_per_backend: int,
    seed: int,
) -> list[dict[str, Any]]:
    source = cfg["source"]
    allowed_datasets = {str(value) for value in source["datasets"]}
    allowed_turns = {int(value) for value in source["intervention_turns"]}
    allowed_topks = {int(value) for value in source["topks"]}
    maximum_prefix = float(source["maximum_prefix_support_recall"])
    filtered = [
        dict(row)
        for row in rows
        if str(row["dataset"]) in allowed_datasets
        and int(row["source_turn"]) in allowed_turns
        and int(row["topk"]) in allowed_topks
        and float(row["prefix_support_recall"]) <= maximum_prefix
    ]
    selected: list[dict[str, Any]] = []
    for backend_index, backend in enumerate(("bm25", "e5")):
        pool = [row for row in filtered if row["backend"] == backend]
        picked = balanced_trim(
            pool,
            count_per_backend,
            seed=seed + backend_index,
            group_keys=("dataset", "source_turn", "policy_tag", "policy_seed"),
        )
        if len(picked) < count_per_backend:
            raise RuntimeError(
                f"Only {len(picked)} eligible {backend} states; profile requests "
                f"{count_per_backend}"
            )
        selected.extend(picked)
    identities = [str(row["state_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Selected behavior-alias states contain duplicate state IDs")
    return sorted(selected, key=lambda row: (row["backend"], row["dataset"], row["state_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EXP-015 behavior-alias states.")
    parser.add_argument("--config", default="configs/behavior_alias_pilot.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--states-file", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    configured_source_profile = str(cfg["source"]["causal_audit_profile"])
    source_profile = args.profile if configured_source_profile == "same" else configured_source_profile
    states_file = Path(
        args.states_file
        or cfg["source"].get("states_file")
        or Path(cfg["source"]["causal_audit_root"])
        / "states"
        / source_profile
        / "states.jsonl"
    ).expanduser().resolve()
    if not states_file.is_file():
        raise RuntimeError(
            f"Missing causal-query state bank: {states_file}. Run EXP-013/014 first "
            "or pass --states-file."
        )
    rows = read_jsonl(states_file)
    for row in rows:
        validate_state(row)
    selected = select_states(
        rows,
        cfg=cfg,
        count_per_backend=int(profile["states_per_backend"]),
        seed=int(cfg["source"]["selection_seed"]),
    )

    root = Path(args.output_root or cfg["work_dir"]).resolve() / "states" / args.profile
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "states.jsonl"
    atomic_write_jsonl(output_path, selected)
    manifest = {
        "schema": SCHEMA,
        "experiment_id": "EXP-015",
        "profile": args.profile,
        "source_states_file": str(states_file),
        "source_states_sha256": file_sha256(states_file),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "selected_states": len(selected),
        "states_per_backend": int(profile["states_per_backend"]),
        "output_sha256": file_sha256(output_path),
    }
    manifest["signature"] = canonical_signature(manifest)
    atomic_write_json(root / "manifest.json", manifest)
    counts: dict[str, int] = {}
    for row in selected:
        key = f"{row['backend']}:{row['dataset']}:turn{row['source_turn']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({**manifest, "strata": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from stackpilot.query_equivalence_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    discover_paths,
    file_sha256,
    inspect_equivalence_state,
    load_config,
    split_value,
)


def _patterns(cfg: dict[str, Any], provided: list[str] | None) -> list[str]:
    if provided:
        return list(provided)
    environment = os.environ.get("QUERY_EQUIVALENCE_INPUTS", "").strip()
    if environment:
        return [part for part in environment.replace("\n", os.pathsep).split(os.pathsep) if part]
    return [str(value) for value in cfg["source"]["input_globs"]]


def prepare(
    *,
    config_path: Path,
    profile: str,
    inputs: list[str] | None,
    output_root: Path | None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if profile not in cfg["profiles"]:
        raise KeyError(f"Unknown profile {profile!r}")
    paths = discover_paths(_patterns(cfg, inputs))
    if not paths:
        raise RuntimeError(
            "No EXP-014 state-result JSON matched. Set QUERY_EQUIVALENCE_INPUTS to "
            "work/causal_query_audit/results/<profile>/states/*/*.json files."
        )
    run_signatures: set[str] = set()
    source_manifest: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    split_seed = int(cfg["splits"]["seed"])
    train_ratio = float(cfg["splits"]["train_ratio"])
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("splits.train_ratio must be in (0, 1)")

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} is not a JSON object")
        inspected = inspect_equivalence_state(payload, cfg=cfg)
        split = "train" if split_value(inspected["question_id"], split_seed) < train_ratio else "heldout"
        inspected["split"] = split
        inspected["source_path"] = str(path)
        run_signature = str(payload.get("run_signature", ""))
        if run_signature:
            run_signatures.add(run_signature)
        audit_rows.append(inspected)
        if int(inspected["eligible"]) == 1:
            eligible_rows.append(inspected)
        source_manifest.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "state_id": inspected["state_id"],
                "eligible": int(inspected["eligible"]),
            }
        )

    if len(run_signatures) != 1:
        raise RuntimeError(
            "EXP-014 inputs must come from one causal-query run signature; found "
            f"{sorted(run_signatures)}"
        )
    root = (output_root or Path(cfg["work_dir"])).resolve() / "prepared" / profile
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "state_audit.jsonl"
    eligible_path = root / "eligible_states.jsonl"
    atomic_write_jsonl(audit_path, audit_rows)
    atomic_write_jsonl(eligible_path, eligible_rows)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in audit_rows:
        key = f"{row['dataset']}:turn-{row['source_turn']}:{row['split']}"
        counts[str(row["backend"])][key] += 1
    direct_states = sum(int(row["direct_candidate_count"] > 0) for row in audit_rows)
    nontrivial_states = sum(int(row["nontrivial_best_class"]) for row in audit_rows)
    factual_replaceable = sum(
        int(row["nontrivial_best_class"] and row["factual_in_best_class"])
        for row in audit_rows
    )
    manifest = {
        "schema": SCHEMA,
        "profile": profile,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_run_signature": next(iter(run_signatures)),
        "sources": source_manifest,
        "audit_states": len(audit_rows),
        "direct_states": direct_states,
        "nontrivial_equivalence_states": nontrivial_states,
        "eligible_states": len(eligible_rows),
        "factual_replaceable_states": factual_replaceable,
        "nontrivial_equivalence_rate_among_direct": (
            nontrivial_states / direct_states if direct_states else 0.0
        ),
        "factual_replaceability_rate_among_direct": (
            factual_replaceable / direct_states if direct_states else 0.0
        ),
        "audit_path": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "eligible_path": str(eligible_path),
        "eligible_sha256": file_sha256(eligible_path),
        "counts": {backend: dict(values) for backend, values in counts.items()},
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare functional query-equivalence classes from EXP-014 state results."
    )
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    payload = prepare(
        config_path=Path(args.config).resolve(),
        profile=args.profile,
        inputs=args.inputs,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

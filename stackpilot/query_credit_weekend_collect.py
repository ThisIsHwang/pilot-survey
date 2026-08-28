from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from stackpilot.causal_query_common import load_causal_query_config
from stackpilot.causal_query_replay import _retrievers
from stackpilot.interface_causality_common import load_state_results
from stackpilot.query_credit_common import (
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    stable_hash,
)
from stackpilot.query_credit_weekend_collect_state import process_state
from stackpilot.query_credit_weekend_collect_support import (
    SCHEMA,
    _sensitivity_state_ids,
    _service_check,
    discover_inputs,
)
from stackpilot.query_credit_weekend_common import (
    apply_model_override,
    stable_balanced_sample,
)


def finalize(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    root = Path(cfg["work_dir"]).resolve() / profile_name
    cache_root = root / "cache"
    selection_path = root / "data" / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError(
            f"Missing precommitted selection manifest: {selection_path}. "
            "Start collection once before using --finalize-only."
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_run_signature = str(selection["run_signature"])
    allowed_state_ids = {str(value) for value in selection["selected_state_ids"]}
    payloads = []
    for path in sorted(cache_root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("schema") == SCHEMA
            and payload.get("candidates")
            and str(payload.get("run_signature", "")) == expected_run_signature
            and str(payload.get("state_id", "")) in allowed_state_ids
        ):
            payloads.append(payload)
    candidates = [row for payload in payloads for row in payload["candidates"]]
    prefixes = [
        {
            "state_id": payload["state_id"],
            "question_id": payload["question_id"],
            "dataset": payload["dataset"],
            "backend": payload["backend"],
            "prefix_messages": payload["prefix_messages"],
        }
        for payload in payloads
    ]
    raw = [row for payload in payloads for row in payload["raw_replays"]]
    manifest = [
        {
            key: payload[key]
            for key in (
                "state_id",
                "question_id",
                "dataset",
                "backend",
                "candidate_count",
                "direct_policy_candidate_fraction",
            )
        }
        | {
            "corpus_probe": payload.get("corpus_probe", []),
        }
        for payload in payloads
    ]
    output_root = root / "data"
    atomic_write_jsonl(output_root / "candidate_credits.jsonl", candidates)
    atomic_write_jsonl(output_root / "state_prefixes.jsonl", prefixes)
    atomic_write_jsonl(output_root / "raw_replays.jsonl", raw)
    atomic_write_jsonl(output_root / "state_manifest.jsonl", manifest)
    summary = {
        "schema": SCHEMA,
        "profile": profile_name,
        "run_signature": expected_run_signature,
        "selected_states": len(allowed_state_ids),
        "states": len(payloads),
        "candidates": len(candidates),
        "raw_replays": len(raw),
        "cells": {
            f"{dataset}/{backend}": int(count)
            for (dataset, backend), count in sorted(
                {
                    (payload["dataset"], payload["backend"]): sum(
                        1
                        for item in payloads
                        if item["dataset"] == payload["dataset"]
                        and item["backend"] == payload["backend"]
                    )
                    for payload in payloads
                }.items()
            )
        },
    }
    atomic_write_json(output_root / "collection_summary.json", summary)
    return summary


def run(
    cfg: dict[str, Any],
    causal_cfg: dict[str, Any],
    profile_name: str,
    inputs: Sequence[str] | None,
) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    paths = discover_inputs(cfg, inputs)
    all_results = load_state_results(paths)
    selected, cell_counts = stable_balanced_sample(
        all_results,
        datasets=[str(value) for value in cfg["collection"]["datasets"]],
        backends=[str(value) for value in profile["backends"]],
        per_cell=int(profile["states_per_cell"]),
        salt=str(cfg["collection"]["selection_salt"]),
    )
    expected_cells = len(cfg["collection"]["datasets"]) * len(profile["backends"])
    if len(cell_counts) != expected_cells or any(
        count < int(profile["minimum_states_per_cell"]) for count in cell_counts.values()
    ):
        raise RuntimeError(
            "Too few paired, outcome-blind states for the declared minimum: "
            f"{cell_counts}"
        )
    selected_state_ids = sorted(str(row["state"]["state_id"]) for row in selected)
    run_signature = stable_hash(
        "query-credit-weekend-run-v2",
        profile_name,
        json.dumps(profile, sort_keys=True),
        json.dumps(cfg["collection"], sort_keys=True),
        json.dumps(cfg["analysis"], sort_keys=True),
        str(cfg["model"].get("base_model", "")),
        str(cfg["model"].get("revision", "")),
        *selected_state_ids,
        length=32,
    )
    atomic_write_json(
        Path(cfg["work_dir"]).resolve() / profile_name / "data" / "selection.json",
        {
            "schema": SCHEMA,
            "profile": profile_name,
            "run_signature": run_signature,
            "selected_state_ids": selected_state_ids,
            "cell_targets": cell_counts,
        },
    )
    identities = _service_check(causal_cfg, profile["backends"])
    retrievers = _retrievers(causal_cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        revision=cfg["model"].get("revision"),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    cache_root = Path(cfg["work_dir"]).resolve() / profile_name / "cache"
    omission_state_ids = _sensitivity_state_ids(
        selected,
        per_cell=int(profile["omission_states_per_cell"]),
        salt=str(cfg["collection"]["omission_selection_salt"]),
    )
    errors = []
    with ThreadPoolExecutor(max_workers=int(profile["workers"])) as executor:
        futures = {
            executor.submit(
                process_state,
                result,
                cfg=cfg,
                causal_cfg=causal_cfg,
                profile_name=profile_name,
                retrievers=retrievers,
                tokenizer=tokenizer,
                cache_root=cache_root,
                omission_state_ids=omission_state_ids,
                run_signature=run_signature,
            ): str(result["state"]["state_id"])
            for result in selected
        }
        for future in as_completed(futures):
            state_id = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - persisted for resumable jobs
                errors.append(
                    {
                        "state_id": state_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                atomic_write_jsonl(
                    Path(cfg["work_dir"]).resolve()
                    / profile_name
                    / "data"
                    / "collection_errors.jsonl",
                    errors,
                )
    summary = finalize(cfg, profile_name)
    summary.update(
        {
            "selected_states": len(selected),
            "errors": len(errors),
            "cell_targets": cell_counts,
            "service_identities": identities,
        }
    )
    atomic_write_json(
        Path(cfg["work_dir"]).resolve()
        / profile_name
        / "data"
        / "collection_summary.json",
        summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fixed-cardinality document/action counterfactuals for a 72-hour H100 run."
    )
    parser.add_argument("--config", default="configs/query_credit_weekend.yaml")
    parser.add_argument("--causal-config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "single", "node8"), default="node8")
    parser.add_argument("--input", action="append")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    cfg = apply_model_override(load_config(args.config))
    if args.finalize_only:
        payload = finalize(cfg, args.profile)
    else:
        payload = run(
            cfg,
            load_causal_query_config(args.causal_config),
            args.profile,
            args.input,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

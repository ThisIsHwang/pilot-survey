from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from stackpilot.causal_query_common import (
    canonical_signature,
    load_causal_query_config,
    normalize_title,
    stable_hash,
)
from stackpilot.trace_common import (
    atomic_write_json,
    atomic_write_jsonl,
    discover_paths,
    file_sha256,
    read_jsonl_tolerant,
)

STATE_SCHEMA = 1


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def validate_raw_episode(row: dict[str, Any], source: Path) -> None:
    required = {
        "question_id",
        "question",
        "dataset",
        "backend",
        "topk",
        "turns",
        "answers",
        "support_titles",
    }
    missing = required - set(row)
    if missing:
        raise RuntimeError(f"{source} raw episode is missing fields: {sorted(missing)}")
    if not isinstance(row["turns"], list):
        raise RuntimeError(f"{source} raw episode turns must be a list")
    if not _strings(row["answers"]):
        raise RuntimeError(f"{source} raw episode has no answers")
    if not _strings(row["support_titles"]):
        raise RuntimeError(f"{source} raw episode has no supporting titles")
    if not str(row["question_id"]).strip() or not str(row["question"]).strip():
        raise RuntimeError(f"{source} raw episode has an empty question identity")


def _turn_titles(turn: dict[str, Any]) -> list[str]:
    return _strings(turn.get("observed_titles") or turn.get("retrieved_titles") or [])


def build_candidate_states(
    rows: list[tuple[dict[str, Any], Path]],
    *,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    source_cfg = cfg["source"]
    allowed_datasets = {str(value) for value in source_cfg.get("datasets", [])}
    allowed_tags = {str(value) for value in source_cfg.get("policy_tags", [])}
    allowed_topks = {int(value) for value in source_cfg.get("topks", [])}
    intervention_turns = {int(value) for value in source_cfg["intervention_turns"]}
    max_turns = int(cfg["agent"]["max_search_turns"])
    output: dict[str, dict[str, Any]] = {}

    for raw, source in rows:
        validate_raw_episode(raw, source)
        if bool(source_cfg.get("require_protocol_success", True)) and int(
            raw.get("protocol_failure", 0)
        ) != 0:
            continue
        dataset = str(raw["dataset"])
        policy_tag = str(raw.get("policy_tag", "unknown"))
        backend = str(raw["backend"])
        topk = int(raw["topk"])
        if allowed_datasets and dataset not in allowed_datasets:
            continue
        if allowed_tags and policy_tag not in allowed_tags:
            continue
        if allowed_topks and topk not in allowed_topks:
            continue
        if backend not in {"bm25", "e5"}:
            continue

        turns = raw["turns"]
        for source_turn in sorted(intervention_turns):
            if source_turn < 2 or source_turn > len(turns) or source_turn >= max_turns:
                continue
            factual_turn = turns[source_turn - 1]
            factual_query = str(factual_turn.get("query", "")).strip()
            if not factual_query:
                continue
            prior_turns = []
            valid = True
            for turn_index, turn in enumerate(turns[: source_turn - 1], start=1):
                query = str(turn.get("query", "")).strip()
                if not query:
                    valid = False
                    break
                prior_turns.append(
                    {
                        "turn": turn_index,
                        "query": query,
                        "observed_titles": _turn_titles(turn),
                        "support_recall": float(turn.get("support_recall", 0.0)),
                    }
                )
            if not valid:
                continue
            answers = _strings(raw["answers"])
            support_titles = _strings(raw["support_titles"])
            state_id = stable_hash(
                raw.get("run_signature", ""),
                policy_tag,
                int(raw.get("seed", 0)),
                raw["question_id"],
                backend,
                topk,
                source_turn,
                factual_query,
            )
            state = {
                "schema": STATE_SCHEMA,
                "state_id": state_id,
                "question_id": str(raw["question_id"]),
                "question": str(raw["question"]),
                "answers": answers,
                "support_titles": support_titles,
                "normalized_support_titles": sorted(
                    {normalize_title(value) for value in support_titles}
                ),
                "dataset": dataset,
                "backend": backend,
                "topk": topk,
                "policy_tag": policy_tag,
                "policy_seed": int(raw.get("seed", 0)),
                "source_turn": source_turn,
                "prior_turns": prior_turns,
                "factual_query": factual_query,
                "raw_factual_observed_titles": _turn_titles(factual_turn),
                "raw_factual_support_recall": float(
                    factual_turn.get("support_recall", 0.0)
                ),
                "raw_factual_evidence_gain": float(
                    factual_turn.get("evidence_gain", 0.0)
                ),
                "source_path": str(source),
                "source_run_signature": str(raw.get("run_signature", "")),
            }
            existing = output.get(state_id)
            if existing is not None and canonical_signature(existing) != canonical_signature(state):
                raise RuntimeError(f"Conflicting raw state for {state_id}")
            output[state_id] = state
    return list(output.values())


def select_balanced_states(
    candidates: list[dict[str, Any]],
    *,
    count_per_backend: int,
    seed: int,
    one_state_per_question_backend: bool,
) -> list[dict[str, Any]]:
    if count_per_backend < 1:
        raise ValueError("count_per_backend must be positive")
    selected: list[dict[str, Any]] = []
    for backend in ("bm25", "e5"):
        backend_rows = [row for row in candidates if row["backend"] == backend]
        strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in backend_rows:
            strata[(str(row["dataset"]), int(row["source_turn"]))].append(row)
        for stratum, rows in strata.items():
            rows.sort(
                key=lambda row: stable_hash(
                    seed,
                    backend,
                    stratum,
                    row["question_id"],
                    row["policy_tag"],
                    row["policy_seed"],
                    row["state_id"],
                )
            )
        keys = sorted(strata, key=repr)
        if not keys:
            raise RuntimeError(f"No causal-query states are available for {backend}")
        used_question_backend: set[tuple[str, str]] = set()
        backend_selected: list[dict[str, Any]] = []
        cursor = 0
        stagnant_rounds = 0
        while len(backend_selected) < count_per_backend and keys:
            key = keys[cursor % len(keys)]
            bucket = strata[key]
            chosen = None
            while bucket:
                candidate = bucket.pop(0)
                question_key = (str(candidate["question_id"]), backend)
                if one_state_per_question_backend and question_key in used_question_backend:
                    continue
                chosen = candidate
                break
            if chosen is not None:
                backend_selected.append(chosen)
                used_question_backend.add((str(chosen["question_id"]), backend))
                stagnant_rounds = 0
            else:
                keys.remove(key)
                stagnant_rounds += 1
                if not keys:
                    break
                cursor %= len(keys)
                continue
            cursor += 1
            if stagnant_rounds > len(keys) + 1:
                break
        if len(backend_selected) < count_per_backend:
            counts = {repr(key): len(rows) for key, rows in strata.items()}
            raise RuntimeError(
                f"Only {len(backend_selected)} unique {backend} states were available; "
                f"requested {count_per_backend}. Remaining strata: {counts}"
            )
        selected.extend(backend_selected)
    return sorted(
        selected,
        key=lambda row: (
            str(row["backend"]),
            str(row["dataset"]),
            int(row["source_turn"]),
            str(row["state_id"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare state-matched query intervention states for EXP-013."
    )
    parser.add_argument("--config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    cfg = load_causal_query_config(args.config)
    profile = cfg["profiles"][args.profile]
    patterns = args.inputs or list(cfg["source"]["input_globs"])
    paths = discover_paths(patterns)
    if not paths:
        raise RuntimeError(
            "No raw causal-query input JSONL matched. Set CAUSAL_QUERY_INPUTS; "
            "summary CSV reports are insufficient."
        )

    raw_rows: list[tuple[dict[str, Any], Path]] = []
    input_manifest = []
    for path in paths:
        rows = read_jsonl_tolerant(path)
        input_manifest.append(
            {"path": str(path), "sha256": file_sha256(path), "rows": len(rows)}
        )
        raw_rows.extend((row, path) for row in rows)

    candidates = build_candidate_states(raw_rows, cfg=cfg)
    selected = select_balanced_states(
        candidates,
        count_per_backend=int(profile["states_per_backend"]),
        seed=20260731,
        one_state_per_question_backend=bool(
            cfg["source"].get("one_state_per_question_backend", True)
        ),
    )

    work_root = Path(args.output_root or cfg["work_dir"]).resolve()
    state_root = work_root / "states" / args.profile
    state_root.mkdir(parents=True, exist_ok=True)
    states_path = state_root / "states.jsonl"
    atomic_write_jsonl(states_path, selected)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in selected:
        counts[str(row["backend"])][
            f"{row['dataset']}:turn-{row['source_turn']}"
        ] += 1
    manifest = {
        "schema": STATE_SCHEMA,
        "profile": args.profile,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "inputs": input_manifest,
        "candidate_states": len(candidates),
        "selected_states": len(selected),
        "states_path": str(states_path),
        "states_sha256": file_sha256(states_path),
        "counts": {backend: dict(values) for backend, values in counts.items()},
    }
    manifest["signature"] = canonical_signature(manifest)
    atomic_write_json(state_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

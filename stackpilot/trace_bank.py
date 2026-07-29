from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stackpilot.trace_common import (
    TRACE_SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    classify_episode,
    counterfactual_recovery_score,
    discover_paths,
    file_sha256,
    load_trace_config,
    question_split,
    read_jsonl_tolerant,
    reformulation_prompt,
    stable_hash,
)


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Episode field {key!r} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise RuntimeError(f"Episode field {key!r} is not finite: {value!r}")
    return result


def _episode_key(row: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (
        str(row.get("policy_tag", "unknown")),
        int(row.get("seed", 0)),
        str(row.get("question_id", "")),
        str(row.get("backend", "")),
        int(row.get("topk", -1)),
    )


def validate_episode(row: dict[str, Any], source: Path) -> None:
    required = {
        "question_id",
        "question",
        "dataset",
        "backend",
        "topk",
        "turns",
    }
    missing = required - set(row)
    if missing:
        raise RuntimeError(f"{source} episode is missing fields: {sorted(missing)}")
    if not isinstance(row["turns"], list):
        raise RuntimeError(f"{source} episode turns must be a list")
    if not str(row["question_id"]).strip() or not str(row["question"]).strip():
        raise RuntimeError(f"{source} episode has an empty question identity")
    if not str(row["backend"]).strip() or int(row["topk"]) <= 0:
        raise RuntimeError(f"{source} episode has an invalid retrieval view")


def episode_summary(
    row: dict[str, Any],
    *,
    source: Path,
    split_seed: int,
    train_ratio: float,
    calibration_ratio: float,
    fail_threshold: float,
    solve_threshold: float,
    min_recovery: float,
    search_cost: float,
) -> dict[str, Any]:
    turns = row["turns"]
    recalls = [float(turn.get("support_recall", 0.0)) for turn in turns]
    gains = [float(turn.get("evidence_gain", 0.0)) for turn in turns]
    turn1 = recalls[0] if recalls else _number(row, "turn1_support_recall")
    final = recalls[-1] if recalls else _number(row, "support_recall")
    search_count = int(row.get("search_count", len(turns)))
    if search_count != len(turns):
        # Hard-RQ0 stores one turn record per executed search. A mismatch means
        # the raw trajectory cannot support transition-level curricula.
        raise RuntimeError(
            f"{source} episode {_episode_key(row)} has search_count={search_count} "
            f"but {len(turns)} turn records"
        )
    label = classify_episode(
        turn1,
        final,
        fail_threshold=fail_threshold,
        solve_threshold=solve_threshold,
        min_recovery=min_recovery,
    )
    crs = counterfactual_recovery_score(
        turn1,
        final,
        fail_threshold=fail_threshold,
        solve_threshold=solve_threshold,
        min_recovery=min_recovery,
        search_count=search_count,
        search_cost=search_cost,
    )
    question_id = str(row["question_id"])
    view_id = f"{row['backend']}:k{int(row['topk'])}"
    policy_tag = str(row.get("policy_tag", "unknown"))
    policy_seed = int(row.get("seed", 0))
    episode_id = stable_hash(policy_tag, policy_seed, question_id, view_id)
    return {
        "schema": TRACE_SCHEMA,
        "episode_id": episode_id,
        "question_id": question_id,
        "question": str(row["question"]),
        "dataset": str(row["dataset"]),
        "policy_tag": policy_tag,
        "policy_seed": policy_seed,
        "backend": str(row["backend"]),
        "topk": int(row["topk"]),
        "view_id": view_id,
        "split": question_split(
            question_id,
            seed=split_seed,
            train_ratio=train_ratio,
            calibration_ratio=calibration_ratio,
        ),
        "search_count": search_count,
        "turn1_recall": turn1,
        "final_recall": final,
        "total_recovery": max(0.0, final - turn1),
        "max_turn_gain": max(gains, default=0.0),
        "answer_em": _number(row, "em"),
        "answer_f1": _number(row, "f1"),
        "protocol_failure": int(row.get("protocol_failure", 0)),
        "episode_class": label,
        "crs": crs,
        "source_path": str(source),
        "source_run_signature": str(row.get("run_signature", "")),
        "turns": turns,
    }


def add_group_features(episodes: list[dict[str, Any]]) -> None:
    reward_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    base_by_question: dict[str, list[float]] = defaultdict(list)
    fallback_by_question: dict[str, list[float]] = defaultdict(list)
    crs_by_question_view: dict[tuple[str, str], list[float]] = defaultdict(list)
    crs_by_question_backend: dict[tuple[str, str], list[float]] = defaultdict(list)

    for episode in episodes:
        reward_proxy = (
            0.5 * float(episode["answer_em"])
            + 0.5 * float(episode["final_recall"])
            - 0.02 * float(episode["search_count"])
        )
        reward_groups[(episode["question_id"], episode["view_id"])].append(
            reward_proxy
        )
        fallback_by_question[episode["question_id"]].append(
            float(episode["final_recall"])
        )
        if episode["policy_tag"] == "base-qwen":
            base_by_question[episode["question_id"]].append(
                float(episode["final_recall"])
            )
        crs_by_question_view[(episode["question_id"], episode["view_id"])].append(
            float(episode["crs"])
        )
        crs_by_question_backend[
            (episode["question_id"], episode["backend"])
        ].append(float(episode["crs"]))

    mean_crs = {
        key: float(np.mean(values)) for key, values in crs_by_question_view.items()
    }
    mean_backend_crs = {
        key: float(np.mean(values))
        for key, values in crs_by_question_backend.items()
    }
    views_by_question: dict[str, set[str]] = defaultdict(set)
    backends_by_question: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        views_by_question[episode["question_id"]].add(episode["view_id"])
        backends_by_question[episode["question_id"]].add(episode["backend"])

    for episode in episodes:
        group = reward_groups[(episode["question_id"], episode["view_id"])]
        episode["reward_variance"] = float(np.var(group)) if len(group) > 1 else 0.0
        difficulty_values = base_by_question.get(episode["question_id"]) or fallback_by_question[
            episode["question_id"]
        ]
        episode["question_difficulty"] = 1.0 - float(np.mean(difficulty_values))
        other_views = views_by_question[episode["question_id"]] - {episode["view_id"]}
        other_view_crs = [
            mean_crs[(episode["question_id"], view_id)] for view_id in other_views
        ]
        other_backends = backends_by_question[episode["question_id"]] - {
            episode["backend"]
        }
        other_backend_crs = [
            mean_backend_crs[(episode["question_id"], backend)]
            for backend in other_backends
        ]
        episode["other_view_crs"] = (
            float(np.mean(other_view_crs)) if other_view_crs else 0.0
        )
        episode["cross_backend_crs"] = (
            float(np.mean(other_backend_crs)) if other_backend_crs else 0.0
        )
        episode["portable_recovery_proxy"] = float(episode["crs"]) * float(
            episode["cross_backend_crs"]
        )
        episode["paired_view_count"] = len(views_by_question[episode["question_id"]])
        episode["paired_backend_count"] = len(
            backends_by_question[episode["question_id"]]
        )


def build_transitions(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for episode in episodes:
        turns = episode["turns"]
        prior_queries: list[str] = []
        prior_titles: list[list[str]] = []
        for turn_index, turn in enumerate(turns):
            query = str(turn.get("query", "")).strip()
            if not query:
                raise RuntimeError(
                    f"Episode {episode['episode_id']} turn {turn_index + 1} has no query"
                )
            observed_titles = [
                str(value).strip()
                for value in (
                    turn.get("observed_titles")
                    or turn.get("retrieved_titles")
                    or []
                )
                if str(value).strip()
            ]
            if turn_index >= 1:
                prompt = reformulation_prompt(
                    question=episode["question"],
                    prior_queries=prior_queries,
                    prior_observed_titles=prior_titles,
                )
                evidence_gain = float(turn.get("evidence_gain", 0.0))
                transition_id = stable_hash(
                    episode["episode_id"], turn_index + 1, query
                )
                transitions.append(
                    {
                        "schema": TRACE_SCHEMA,
                        "transition_id": transition_id,
                        "episode_id": episode["episode_id"],
                        "question_id": episode["question_id"],
                        "question": episode["question"],
                        "dataset": episode["dataset"],
                        "policy_tag": episode["policy_tag"],
                        "policy_seed": episode["policy_seed"],
                        "backend": episode["backend"],
                        "topk": episode["topk"],
                        "view_id": episode["view_id"],
                        "split": episode["split"],
                        "source_turn": turn_index + 1,
                        "prompt": prompt,
                        "target": query,
                        "evidence_gain": evidence_gain,
                        "positive_gain": int(evidence_gain > 1e-12),
                        "episode_class": episode["episode_class"],
                        "search_count": episode["search_count"],
                        "turn1_recall": episode["turn1_recall"],
                        "final_recall": episode["final_recall"],
                        "total_recovery": episode["total_recovery"],
                        "crs": episode["crs"],
                        "portable_recovery_proxy": episode[
                            "portable_recovery_proxy"
                        ],
                        "other_view_crs": episode["other_view_crs"],
                        "cross_backend_crs": episode["cross_backend_crs"],
                        "reward_variance": episode["reward_variance"],
                        "question_difficulty": episode["question_difficulty"],
                        "paired_view_count": episode["paired_view_count"],
                        "paired_backend_count": episode["paired_backend_count"],
                    }
                )
            prior_queries.append(query)
            prior_titles.append(observed_titles)
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the TRACE episode/transition bank from raw Hard-RQ0 trajectories."
    )
    parser.add_argument("--config", default="configs/trace_go.yaml")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    cfg = load_trace_config(args.config)
    bank_cfg = cfg["bank"]
    patterns = args.inputs or list(bank_cfg["input_globs"])
    paths = discover_paths(patterns)
    if not paths:
        raise RuntimeError(
            "No raw episode JSONL files matched. Set TRACE_INPUTS or pass --inputs. "
            "Summary CSVs are insufficient; TRACE needs per-turn trajectory records."
        )

    output_root = Path(args.output_root or cfg["work_dir"]).resolve() / "bank"
    output_root.mkdir(parents=True, exist_ok=True)
    split_cfg = cfg["splits"]
    recovery_cfg = cfg["recovery"]

    selected_tags = {str(value) for value in bank_cfg.get("policy_tags", [])}
    by_key: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    input_manifest = []
    for path in paths:
        rows = read_jsonl_tolerant(path)
        input_manifest.append(
            {"path": str(path), "sha256": file_sha256(path), "rows": len(rows)}
        )
        for raw in rows:
            validate_episode(raw, path)
            if selected_tags and str(raw.get("policy_tag", "unknown")) not in selected_tags:
                continue
            if int(raw.get("protocol_failure", 0)) != 0:
                continue
            summary = episode_summary(
                raw,
                source=path,
                split_seed=int(split_cfg["seed"]),
                train_ratio=float(split_cfg["train_ratio"]),
                calibration_ratio=float(split_cfg["calibration_ratio"]),
                fail_threshold=float(recovery_cfg["fail_threshold"]),
                solve_threshold=float(recovery_cfg["solve_threshold"]),
                min_recovery=float(recovery_cfg["min_recovery"]),
                search_cost=float(recovery_cfg["search_cost"]),
            )
            key = _episode_key(raw)
            existing = by_key.get(key)
            if existing is not None:
                if canonical_signature(existing) != canonical_signature(summary):
                    raise RuntimeError(
                        f"Conflicting duplicate episode for {key}: "
                        f"{existing['source_path']} vs {path}"
                    )
                continue
            by_key[key] = summary

    episodes = sorted(
        by_key.values(),
        key=lambda row: (
            row["split"],
            row["question_id"],
            row["view_id"],
            row["policy_tag"],
            row["policy_seed"],
        ),
    )
    if not episodes:
        raise RuntimeError("No valid TRACE episodes remained after filtering")
    add_group_features(episodes)
    transitions = build_transitions(episodes)
    if not transitions:
        raise RuntimeError(
            "No query-reformulation transitions were found. The input trajectories "
            "must include at least two executed searches."
        )

    split_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    for episode in episodes:
        split_counts[episode["split"]] += 1
        class_counts[episode["episode_class"]] += 1

    manifest = {
        "schema": TRACE_SCHEMA,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "inputs": input_manifest,
        "episode_count": len(episodes),
        "transition_count": len(transitions),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "signature": canonical_signature(
            {
                "config": cfg,
                "inputs": input_manifest,
                "episode_ids": [row["episode_id"] for row in episodes],
            }
        ),
    }
    for episode in episodes:
        episode.pop("turns", None)
    atomic_write_jsonl(output_root / "episodes.jsonl", episodes)
    atomic_write_jsonl(output_root / "transitions.jsonl", transitions)
    atomic_write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

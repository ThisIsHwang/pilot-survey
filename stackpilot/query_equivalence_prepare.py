from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from stackpilot.query_equivalence_common import (
    EquivalenceThresholds,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_signature,
    class_summary,
    equivalence_classes,
    equivalence_edges,
    hash_split,
    load_equivalence_config,
    stable_hash,
)


def discover_result_paths(patterns: list[str]) -> list[Path]:
    output: set[Path] = set()
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        if expanded.is_file():
            output.add(expanded.resolve())
            continue
        for raw in glob.glob(str(expanded), recursive=True):
            path = Path(raw)
            if path.is_file() and path.suffix == ".json":
                output.add(path.resolve())
    return sorted(output)


def thresholds_from_config(cfg: dict[str, Any]) -> EquivalenceThresholds:
    section = cfg["equivalence"]
    return EquivalenceThresholds(
        support_recall_tolerance=float(section["support_recall_tolerance"]),
        answer_f1_tolerance=float(section["answer_f1_tolerance"]),
        search_count_tolerance=int(section["search_count_tolerance"]),
        require_same_support_set=bool(section["require_same_support_set"]),
        require_same_answer_em=bool(section["require_same_answer_em"]),
        require_same_protocol_status=bool(section["require_same_protocol_status"]),
    )


def validate_result(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = payload.get("state")
    candidates = payload.get("candidates")
    if not isinstance(state, dict) or not isinstance(candidates, list) or len(candidates) < 2:
        raise RuntimeError(f"Invalid causal-query state result: {path}")
    required_state = {
        "state_id", "question_id", "question", "dataset", "backend", "topk",
        "source_turn", "support_titles",
    }
    missing = required_state - set(state)
    if missing:
        raise RuntimeError(f"{path} state misses {sorted(missing)}")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"{path} candidate {index} is not an object")
        for key in (
            "candidate_id", "query", "style", "immediate_support_gain",
            "final_support_recall", "answer_em", "answer_f1",
        ):
            if key not in candidate:
                raise RuntimeError(f"{path} candidate {index} misses {key}")
    return state, candidates


def paired_state_key(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(state["question_id"]), str(state["dataset"]), int(state["source_turn"]),
        int(state["topk"]), str(state.get("policy_tag", "")),
        int(state.get("policy_seed", 0)),
    )


def _pair_relation_agreement(
    left_edges: set[tuple[str, str]],
    right_edges: set[tuple[str, str]],
    styles: list[str],
) -> float:
    if len(styles) < 2:
        return 1.0 if left_edges == right_edges else 0.0
    all_pairs = {
        tuple(sorted((styles[left], styles[right])))
        for left in range(len(styles))
        for right in range(left + 1, len(styles))
    }
    return sum((pair in left_edges) == (pair in right_edges) for pair in all_pairs) / len(all_pairs)


def prepare(config_path: str, result_patterns: list[str] | None = None) -> dict[str, Any]:
    cfg = load_equivalence_config(config_path)
    root = Path(cfg["work_dir"]).resolve()
    output_root = root / "prepared"
    output_root.mkdir(parents=True, exist_ok=True)
    patterns = result_patterns or [str(value) for value in cfg["inputs"]["result_globs"]]
    paths = discover_result_paths(patterns)
    if not paths:
        raise RuntimeError(f"No EXP-013 state result JSON found for patterns: {patterns}")

    thresholds = thresholds_from_config(cfg)
    epsilon = float(cfg["equivalence"]["direct_epsilon"])
    split_cfg = cfg["split"]
    state_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    paired_records: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    run_signatures: set[str] = set()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state, candidates = validate_result(payload, path)
        run_signature = str(payload.get("run_signature", ""))
        if run_signature:
            run_signatures.add(run_signature)
        split = hash_split(
            str(state["question_id"]), seed=int(split_cfg["seed"]),
            train_ratio=float(split_cfg["train_ratio"]),
        )
        groups = equivalence_classes(state, candidates, thresholds)
        summaries: list[dict[str, Any]] = []
        candidate_to_class: dict[int, str] = {}
        for group in groups:
            summary = class_summary(state, candidates, group, epsilon=epsilon)
            summary.update({
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "question": str(state["question"]),
                "dataset": str(state["dataset"]),
                "backend": str(state["backend"]),
                "topk": int(state["topk"]),
                "source_turn": int(state["source_turn"]),
                "policy_tag": str(state.get("policy_tag", "")),
                "policy_seed": int(state.get("policy_seed", 0)),
                "split": split,
            })
            summaries.append(summary)
            class_rows.append(summary)
            for index in group:
                candidate_to_class[index] = str(summary["class_id"])

        factual_index = next((
            index for index, candidate in enumerate(candidates)
            if str(candidate.get("origin", "")) == "factual"
            or str(candidate.get("style", "")) == "factual"
        ), None)
        factual_class = candidate_to_class.get(factual_index, "") if factual_index is not None else ""
        factual_summary = next((row for row in summaries if row["class_id"] == factual_class), None)
        nontrivial_direct_classes = [
            row for row in summaries if row["class_size"] >= 2 and row["contains_direct"]
        ]
        state_row = {
            "state_id": str(state["state_id"]),
            "question_id": str(state["question_id"]),
            "question": str(state["question"]),
            "dataset": str(state["dataset"]),
            "backend": str(state["backend"]),
            "topk": int(state["topk"]),
            "source_turn": int(state["source_turn"]),
            "policy_tag": str(state.get("policy_tag", "")),
            "policy_seed": int(state.get("policy_seed", 0)),
            "split": split,
            "candidate_count": len(candidates),
            "direct_candidate_count": sum(
                float(candidate.get("immediate_support_gain", 0.0)) > epsilon
                for candidate in candidates
            ),
            "equivalence_class_count": len(groups),
            "largest_class_size": max(len(group) for group in groups),
            "nontrivial_direct_class_count": len(nontrivial_direct_classes),
            "factual_class_id": factual_class,
            "factual_class_size": int(factual_summary["class_size"]) if factual_summary else 0,
            "factual_class_contains_direct": bool(factual_summary and factual_summary["contains_direct"]),
            "factual_direct": bool(
                factual_index is not None
                and float(candidates[factual_index].get("immediate_support_gain", 0.0)) > epsilon
            ),
            "candidate_styles": sorted(str(candidate.get("style", "")) for candidate in candidates),
            "equivalence_edges": [list(edge) for edge in sorted(equivalence_edges(state, candidates, thresholds))],
            "source_file": str(path),
        }
        state_rows.append(state_row)
        paired_records.setdefault(paired_state_key(state), {})[str(state["backend"])] = state_row

        prior_queries = [str(turn.get("query", "")) for turn in state.get("prior_turns", [])]
        prior_titles = [
            [str(value) for value in turn.get("observed_titles", []) or []]
            for turn in state.get("prior_turns", [])
        ]
        prompt_lines = [
            "Generate the next search query that is most likely to retrieve useful evidence.",
            "Return only the query, without explanation or XML tags.", "",
            f"Question: {state['question']}",
        ]
        for turn_index, query in enumerate(prior_queries, 1):
            prompt_lines.append(f"Previous query {turn_index}: {query}")
            titles = prior_titles[turn_index - 1]
            prompt_lines.append("Observed titles: " + (" | ".join(titles) if titles else "(none)"))
        prompt_lines.extend(["", "Next query:"])
        prompt = "\n".join(prompt_lines)
        for index, candidate in enumerate(candidates):
            class_row = next(row for row in summaries if row["class_id"] == candidate_to_class[index])
            candidate_rows.append({
                "candidate_id": str(candidate["candidate_id"]),
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "dataset": str(state["dataset"]),
                "backend": str(state["backend"]),
                "topk": int(state["topk"]),
                "source_turn": int(state["source_turn"]),
                "policy_tag": str(state.get("policy_tag", "")),
                "policy_seed": int(state.get("policy_seed", 0)),
                "split": split,
                "class_id": candidate_to_class[index],
                "style": str(candidate.get("style", "")),
                "origin": str(candidate.get("origin", "")),
                "query": str(candidate["query"]),
                "prompt": prompt,
                "immediate_support_gain": float(candidate["immediate_support_gain"]),
                "final_support_recall": float(candidate["final_support_recall"]),
                "answer_em": int(candidate.get("answer_em", 0)),
                "answer_f1": float(candidate.get("answer_f1", 0.0)),
                "total_search_count": int(candidate.get("total_search_count", 0)),
                "suffix_search_count": int(candidate.get("suffix_search_count", 0)),
                "support_tqe": float(candidate.get("support_tqe", 0.0)),
                "composite_tqe": float(candidate.get("composite_tqe", 0.0)),
                "final_support_set": list(class_row["final_support_set"]),
            })

    paired_rows: list[dict[str, Any]] = []
    for key, by_backend in paired_records.items():
        if "bm25" not in by_backend or "e5" not in by_backend:
            continue
        left, right = by_backend["bm25"], by_backend["e5"]
        left_edges = {tuple(edge) for edge in left["equivalence_edges"]}
        right_edges = {tuple(edge) for edge in right["equivalence_edges"]}
        union = left_edges | right_edges
        styles = sorted(set(left.get("candidate_styles", [])) | set(right.get("candidate_styles", [])))
        paired_rows.append({
            "pair_id": stable_hash(*key),
            "question_id": key[0], "dataset": key[1], "source_turn": key[2],
            "topk": key[3], "policy_tag": key[4], "policy_seed": key[5],
            "bm25_state_id": left["state_id"], "e5_state_id": right["state_id"],
            "bm25_edges": [list(edge) for edge in sorted(left_edges)],
            "e5_edges": [list(edge) for edge in sorted(right_edges)],
            "edge_jaccard": (len(left_edges & right_edges) / len(union) if union else 1.0),
            "relation_agreement": _pair_relation_agreement(left_edges, right_edges, styles),
        })

    atomic_write_jsonl(output_root / "states.jsonl", state_rows)
    atomic_write_jsonl(output_root / "classes.jsonl", class_rows)
    atomic_write_jsonl(output_root / "candidates.jsonl", candidate_rows)
    atomic_write_jsonl(output_root / "paired_states.jsonl", paired_rows)
    manifest = {
        "schema": 1,
        "experiment_id": str(cfg["experiment_id"]),
        "config_path": str(Path(config_path).resolve()),
        "source_files": [str(path) for path in paths],
        "run_signatures": sorted(run_signatures),
        "states": len(state_rows), "classes": len(class_rows),
        "candidates": len(candidate_rows), "paired_states": len(paired_rows),
        "thresholds": thresholds.__dict__,
    }
    manifest["signature"] = canonical_signature(manifest)
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EXP-014 query equivalence classes.")
    parser.add_argument("--config", default="configs/query_equivalence.yaml")
    parser.add_argument("--results", nargs="*", default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

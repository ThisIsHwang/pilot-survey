from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA = 1
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != SCHEMA:
        raise ValueError(f"Unsupported behavior-quotient config: {config_path}")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_hash(*parts: object, length: int = 24) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def normalize_title(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def word_tokens(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value))]


def token_set(value: Any) -> set[str]:
    return set(word_tokens(value))


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def discover_paths(patterns: Sequence[str], *, suffixes: tuple[str, ...] = (".json",)) -> list[Path]:
    output: dict[str, Path] = {}
    for pattern in patterns:
        expanded = os.path.expanduser(str(pattern))
        candidate = Path(expanded)
        if candidate.is_file():
            output[str(candidate.resolve())] = candidate.resolve()
            continue
        for raw in glob.glob(expanded, recursive=True):
            path = Path(raw).resolve()
            if path.is_file() and (not suffixes or path.suffix in suffixes):
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def source_patterns(cfg: dict[str, Any], provided: Sequence[str] | None = None) -> list[str]:
    if provided:
        return [str(value) for value in provided]
    environment = os.environ.get("BEHAVIOR_QUOTIENT_INPUTS", "").strip()
    if environment:
        normalized = environment.replace("\n", os.pathsep)
        return [part for part in normalized.split(os.pathsep) if part]
    return [str(value) for value in cfg["source"]["input_globs"]]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def candidate_titles(candidate: dict[str, Any], *, final: bool = False) -> list[str]:
    if not final:
        return _string_list(candidate.get("intervention_observed_titles"))
    titles: list[str] = []
    for turn in candidate.get("branch_turns", []) or []:
        if isinstance(turn, dict):
            titles.extend(_string_list(turn.get("observed_titles")))
    if not titles:
        titles = _string_list(candidate.get("intervention_observed_titles"))
    return titles


def ranked_transition(candidate: dict[str, Any], *, topn: int | None = None) -> tuple[str, ...]:
    titles = [normalize_title(value) for value in candidate_titles(candidate, final=False)]
    if topn is not None:
        titles = titles[: int(topn)]
    return tuple(titles)


def unordered_transition(candidate: dict[str, Any], *, topn: int | None = None) -> tuple[str, ...]:
    return tuple(sorted(set(ranked_transition(candidate, topn=topn))))


def full_trajectory_signature(candidate: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    turns: list[tuple[str, ...]] = []
    branch_turns = candidate.get("branch_turns", []) or []
    if isinstance(branch_turns, list):
        for turn in branch_turns:
            if isinstance(turn, dict):
                titles = tuple(normalize_title(value) for value in _string_list(turn.get("observed_titles")))
                if titles:
                    turns.append(titles)
    if not turns:
        immediate = ranked_transition(candidate)
        if immediate:
            turns.append(immediate)
    return tuple(turns)


def gold_support_set(candidate: dict[str, Any], state: dict[str, Any], *, final: bool) -> tuple[str, ...]:
    gold = {normalize_title(value) for value in _string_list(state.get("support_titles"))}
    observed = {normalize_title(value) for value in candidate_titles(candidate, final=final)}
    return tuple(sorted(gold & observed))


def validate_state_result(payload: dict[str, Any], path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = str(path) if path else str(payload.get("state_signature", "state"))
    state = payload.get("state")
    candidates = payload.get("candidates")
    if not isinstance(state, dict) or not isinstance(candidates, list) or len(candidates) < 2:
        raise RuntimeError(f"Invalid causal-query state result: {label}")
    required = {"state_id", "question_id", "question", "dataset", "backend", "topk", "support_titles"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"{label} misses state keys {sorted(missing)}")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"{label} candidate {index} is not an object")
        for key in ("candidate_id", "query", "immediate_support_gain", "final_support_recall"):
            if key not in candidate:
                raise RuntimeError(f"{label} candidate {index} misses {key}")
        for key in ("immediate_support_gain", "final_support_recall", "answer_f1", "support_tqe", "composite_tqe"):
            number = float(candidate.get(key, 0.0))
            if not math.isfinite(number):
                raise RuntimeError(f"{label} candidate {index} has non-finite {key}")
    return state, candidates


def load_state_results(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = discover_paths(patterns)
    if not paths:
        raise RuntimeError(f"No state result matched {list(patterns)}")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    run_signatures: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state, _ = validate_state_result(payload, path)
        state_id = str(state["state_id"])
        if state_id in seen:
            continue
        seen.add(state_id)
        run_signature = str(payload.get("run_signature", ""))
        if run_signature:
            run_signatures.add(run_signature)
        payload["_source_path"] = str(path)
        output.append(payload)
    if len(run_signatures) > 1:
        raise RuntimeError(f"State inputs mix run signatures: {sorted(run_signatures)}")
    return output


def balanced_subset(results: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(results):
        return sorted((dict(row) for row in results), key=lambda row: str(row["state"]["state_id"]))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        state = result["state"]
        buckets[(str(state["backend"]), str(state["dataset"]))].append(dict(result))
    for key in buckets:
        buckets[key].sort(key=lambda row: stable_hash("bq-balanced", row["state"]["state_id"]))
    active = sorted(buckets)
    output: list[dict[str, Any]] = []
    cursor = 0
    while active and len(output) < limit:
        key = active[cursor % len(active)]
        if buckets[key]:
            output.append(buckets[key].pop(0))
            cursor += 1
        else:
            active.remove(key)
            if active:
                cursor %= len(active)
    return output


def candidate_reward(candidate: dict[str, Any], cfg: dict[str, Any]) -> float:
    weights = cfg["reward"]
    reward = (
        float(weights["support"]) * float(candidate.get("final_support_recall", 0.0))
        + float(weights["answer_f1"]) * float(candidate.get("answer_f1", 0.0))
        + float(weights["immediate_gain"]) * float(candidate.get("immediate_support_gain", 0.0))
        - float(weights["search_cost"]) * float(candidate.get("total_search_count", 0.0))
        - float(weights["protocol_cost"]) * float(candidate.get("protocol_failure", 0.0))
    )
    if not math.isfinite(reward):
        raise RuntimeError(f"Non-finite candidate reward: {reward}")
    return reward


def effective_count(counts: Sequence[float]) -> float:
    array = np.asarray(counts, dtype=np.float64)
    total = float(array.sum())
    if total <= 0:
        return 0.0
    probabilities = array / total
    return float(1.0 / np.square(probabilities).sum())


def normalize_advantages(values: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return array
    standard = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if not math.isfinite(standard) or standard <= epsilon:
        return np.zeros_like(array)
    return (array - float(array.mean())) / (standard + epsilon)


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    cluster_key: str,
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(dict(row))
    keys = sorted(grouped)
    if not keys:
        raise RuntimeError("Cannot bootstrap empty rows")
    observed = float(statistic([item for key in keys for item in grouped[key]]))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for draw_index in range(int(samples)):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        draw_rows: list[dict[str, Any]] = []
        for occurrence, key in enumerate(sampled):
            for row in grouped[str(key)]:
                copy = dict(row)
                copy["_bootstrap_cluster"] = f"{key}:{occurrence}"
                draw_rows.append(copy)
        draws[draw_index] = float(statistic(draw_rows))
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise RuntimeError("All bootstrap draws are non-finite")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_clusters": float(len(keys)),
        "n_rows": float(len(rows)),
        "finite_draws": float(len(finite)),
    }


def markdown_table(frame: Any, digits: int = 4) -> str:
    import pandas as pd

    if frame is None or frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    output = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers)) ) + " |",
    ]
    output.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(output)

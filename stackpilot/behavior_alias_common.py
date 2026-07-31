from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA = 1
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CONTROL_RE = re.compile(r"</?(?:search|answer|think|information)>", re.IGNORECASE)


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Behavior-alias config must be a mapping: {path}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_hash(*parts: object, length: int = 20) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts, length=16), 16) % (2**31 - 1)


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_query(value: str) -> str:
    return " ".join(str(value).strip().split())


def word_tokens(value: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(value)]


def token_set(value: str) -> set[str]:
    return set(word_tokens(value))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def valid_query(
    query: str,
    *,
    question: str,
    observed_titles: Sequence[str],
    minimum_tokens: int,
    maximum_tokens: int,
) -> bool:
    query = normalize_query(query)
    tokens = word_tokens(query)
    if not query or CONTROL_RE.search(query):
        return False
    if not minimum_tokens <= len(tokens) <= maximum_tokens:
        return False
    known = token_set(question)
    for title in observed_titles:
        known.update(token_set(title))
    return bool(set(tokens) & known)


def behavior_key(transition_ids: Sequence[str]) -> str:
    # Exact visible ranked transition. Relaxed overlap never defines the primary
    # quotient class because that would silently merge different tool outcomes.
    normalized = [normalize_title(value) for value in transition_ids if normalize_title(value)]
    return canonical_signature(normalized)[:24]


def support_recall(support_titles: Sequence[str], observed_titles: Iterable[str]) -> float:
    gold = {normalize_title(value) for value in support_titles if normalize_title(value)}
    if not gold:
        return 0.0
    observed = {normalize_title(value) for value in observed_titles if normalize_title(value)}
    return len(gold & observed) / len(gold)


def entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0]
    if not len(values):
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log(values)).sum())


def effective_count(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if not len(values) or values.sum() <= 0:
        return 0.0
    values = values / values.sum()
    return float(1.0 / np.square(values).sum())


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def balanced_trim(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in group_keys)].append(dict(row))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets, key=repr)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while keys and len(selected) < count:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            keys.remove(key)
            if keys:
                cursor %= len(keys)
        else:
            cursor += 1
    return selected


def choose_injection_class(
    classes: Sequence[dict[str, Any]],
    *,
    minimum_reward_gap: float,
) -> str | None:
    if len(classes) < 2:
        return None
    best = max(float(row["reward"]) for row in classes)
    eligible = [
        row
        for row in classes
        if best - float(row["reward"]) >= minimum_reward_gap
    ]
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda row: (
            int(row["natural_alias_count"]),
            -float(row["reward"]),
            str(row["class_id"]),
        ),
    )
    return str(chosen["class_id"])


def class_distribution(candidates: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        class_id = row.get("class_id", row.get("behavior_class_id"))
        if class_id is None:
            raise RuntimeError("Candidate is missing class_id/behavior_class_id")
        counts[str(class_id)] += 1
    return dict(counts)


def natural_alias_metrics(candidates: Sequence[dict[str, Any]]) -> dict[str, float]:
    counts = class_distribution(candidates)
    total = len(candidates)
    unique = len(counts)
    probabilities = [value / max(1, total) for value in counts.values()]
    return {
        "surface_queries": float(total),
        "behavior_classes": float(unique),
        "alias_fraction": 1.0 - unique / max(1, total),
        "largest_class_share": max(counts.values(), default=0) / max(1, total),
        "surface_entropy": math.log(max(1, total)),
        "behavior_entropy": entropy(probabilities),
        "within_class_entropy": math.log(max(1, total)) - entropy(probabilities),
        "effective_behavior_count": effective_count(probabilities),
    }


def build_injected_pool(
    state: dict[str, Any],
    *,
    multiplicity: int,
) -> list[dict[str, Any]]:
    if multiplicity <= 0:
        raise ValueError("multiplicity must be positive")
    target = str(state["injection_class_id"])
    pool: list[dict[str, Any]] = []
    for class_row in state["classes"]:
        aliases = list(class_row["queries"])
        if not aliases:
            raise RuntimeError(f"Class {class_row['class_id']} has no query aliases")
        count = multiplicity if str(class_row["class_id"]) == target else 1
        for index in range(count):
            query = str(aliases[index % len(aliases)])
            pool.append(
                {
                    "entry_id": stable_hash(
                        state["state_id"], class_row["class_id"], index, query
                    ),
                    "class_id": str(class_row["class_id"]),
                    "query": query,
                    "observed_titles": list(class_row["observed_titles"]),
                    "support_gain": float(class_row["support_gain"]),
                    "reward": float(class_row["reward"]),
                    "is_exact_copy": int(index >= len(aliases)),
                }
            )
    return pool


def _surface_select(
    pool: Sequence[dict[str, Any]],
    *,
    budget: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    return [dict(pool[rng.randrange(len(pool))]) for _ in range(budget)]


def _quotient_select(
    pool: Sequence[dict[str, Any]],
    *,
    budget: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_class[str(row["class_id"])].append(dict(row))
    classes = sorted(by_class)
    selected: list[dict[str, Any]] = []
    while len(selected) < budget:
        order = list(classes)
        rng.shuffle(order)
        for class_id in order:
            selected.append(dict(rng.choice(by_class[class_id])))
            if len(selected) >= budget:
                break
    return selected


def _text_diverse_select(
    pool: Sequence[dict[str, Any]],
    *,
    budget: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in pool]
    if not remaining:
        return []
    selected = [remaining.pop(rng.randrange(len(remaining)))]
    while len(selected) < budget:
        if not remaining:
            remaining = [dict(row) for row in pool]
        selected_tokens = [token_set(row["query"]) for row in selected]
        scores = []
        for index, row in enumerate(remaining):
            tokens = token_set(row["query"])
            min_distance = min(1.0 - jaccard(tokens, prior) for prior in selected_tokens)
            scores.append((min_distance, rng.random(), index))
        _, _, index = max(scores)
        selected.append(remaining.pop(index))
    return selected


def select_queries(
    pool: Sequence[dict[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    if budget <= 0 or not pool:
        raise ValueError("selection requires a non-empty pool and positive budget")
    rng = random.Random(seed)
    if method == "surface":
        return _surface_select(pool, budget=budget, rng=rng)
    if method == "quotient":
        return _quotient_select(pool, budget=budget, rng=rng)
    if method == "text-diverse":
        return _text_diverse_select(pool, budget=budget, rng=rng)
    raise ValueError(f"Unknown selection method: {method}")


def selected_metrics(
    state: dict[str, Any],
    selected: Sequence[dict[str, Any]],
) -> dict[str, float]:
    class_counts = class_distribution(selected)
    probabilities = [value / len(selected) for value in class_counts.values()]
    prefix_titles = {
        normalize_title(title)
        for turn in state["prior_turns"]
        for title in turn.get("observed_titles", [])
        if normalize_title(title)
    }
    observed = set(prefix_titles)
    for row in selected:
        observed.update(normalize_title(value) for value in row["observed_titles"])
    final_recall = support_recall(state["support_titles"], observed)
    prefix_recall = float(state["prefix_support_recall"])
    rewards = [float(row["reward"]) for row in selected]
    gains = [float(row["support_gain"]) for row in selected]
    unique = len(class_counts)
    available = len(state["classes"])
    return {
        "unique_classes": float(unique),
        "class_coverage": unique / max(1, min(len(selected), available)),
        "effective_behavior_count": effective_count(probabilities),
        "behavior_entropy": entropy(probabilities),
        "surface_entropy": math.log(max(1, len(selected))),
        "best_reward": max(rewards, default=0.0),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_variance": float(np.var(rewards)) if rewards else 0.0,
        "best_immediate_gain": max(gains, default=0.0),
        "union_support_recall": final_recall,
        "union_support_gain": final_recall - prefix_recall,
        "duplicate_call_fraction": 1.0 - unique / max(1, len(selected)),
    }


def bootstrap_by_state(
    rows: Sequence[dict[str, Any]],
    *,
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(dict(row))
    state_ids = sorted(by_state)
    if not state_ids:
        raise RuntimeError("No state rows available for bootstrap")
    observed = float(statistic([row for state in state_ids for row in by_state[state]]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(state_ids, size=len(state_ids), replace=True)
        expanded: list[dict[str, Any]] = []
        for draw_id, state_id in enumerate(sampled):
            for row in by_state[str(state_id)]:
                copy = dict(row)
                copy["_bootstrap_state"] = f"{draw_id}:{state_id}"
                expanded.append(copy)
        draws[index] = float(statistic(expanded))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_states": float(len(state_ids)),
        "n_rows": float(len(rows)),
    }

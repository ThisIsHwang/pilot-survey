from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TRACE_SCHEMA = 1


def load_trace_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TRACE config must be a mapping: {config_path}")
    return payload


def stable_hash(*parts: object, length: int = 20) -> str:
    text = "\n".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_jsonl_tolerant(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("rb") as handle:
        lines = handle.readlines()
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                break
            raise RuntimeError(f"Invalid JSONL at {file_path}:{index + 1}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


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
    os.replace(temporary, target)


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
    os.replace(temporary, target)


def discover_paths(patterns: Sequence[str]) -> list[Path]:
    discovered: set[Path] = set()
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        if expanded.is_file():
            discovered.add(expanded.resolve())
            continue
        # pathlib does not support absolute glob patterns through Path.glob.
        import glob

        for raw_path in glob.glob(str(expanded), recursive=True):
            path = Path(raw_path)
            if path.is_file():
                discovered.add(path.resolve())
    return sorted(discovered)


def question_split(
    question_id: str,
    *,
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
) -> str:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= calibration_ratio < 1.0 - train_ratio):
        raise ValueError("calibration_ratio leaves no held-out split")
    value = int(stable_hash(seed, question_id, length=16), 16) / float(16**16)
    if value < train_ratio:
        return "train"
    if value < train_ratio + calibration_ratio:
        return "calibration"
    return "heldout"


def classify_episode(
    turn1_recall: float,
    final_recall: float,
    *,
    fail_threshold: float,
    solve_threshold: float,
    min_recovery: float,
) -> str:
    if turn1_recall >= solve_threshold:
        return "easy"
    recovery = final_recall - turn1_recall
    if (
        turn1_recall <= fail_threshold
        and final_recall >= solve_threshold
        and recovery >= min_recovery
    ):
        return "recoverable"
    return "unrecoverable"


def counterfactual_recovery_score(
    turn1_recall: float,
    final_recall: float,
    *,
    fail_threshold: float,
    solve_threshold: float,
    min_recovery: float,
    search_count: int,
    search_cost: float,
) -> float:
    recovery = max(0.0, final_recall - turn1_recall)
    if turn1_recall > fail_threshold:
        return 0.0
    if final_recall < solve_threshold or recovery < min_recovery:
        return 0.0
    return max(0.0, recovery - search_cost * max(0, search_count - 1))


def approximate_tokens(text: str) -> int:
    # Stable planning proxy. Exact model-token counts are emitted by each job.
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def reformulation_prompt(
    *,
    question: str,
    prior_queries: Sequence[str],
    prior_observed_titles: Sequence[Sequence[str]],
) -> str:
    lines = [
        "Generate the next search query that is most likely to retrieve new evidence.",
        "Return only the query, without explanation or XML tags.",
        "",
        f"Question: {question.strip()}",
    ]
    for index, query in enumerate(prior_queries, start=1):
        lines.append(f"Previous query {index}: {query.strip()}")
        titles = prior_observed_titles[index - 1] if index - 1 < len(prior_observed_titles) else []
        if titles:
            lines.append("Observed titles:")
            lines.extend(f"- {str(title).strip()}" for title in titles if str(title).strip())
        else:
            lines.append("Observed titles: (none)")
    lines.append("")
    lines.append("Next query:")
    return "\n".join(lines)


def deterministic_sample(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    replace: bool = False,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not rows or count == 0:
        return []
    rng = random.Random(seed)
    if replace:
        return [rows[rng.randrange(len(rows))] for _ in range(count)]
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from a pool of {len(rows)}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[index] for index in indices[:count]]


def balanced_trim(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Round-robin deterministic sampling across marginal groups."""
    if count <= 0:
        return []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets, key=lambda value: repr(value))
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return selected


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    return (matrix - means) / scales, means, scales


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge_fit received incompatible shapes")
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return 0.0
    return 1.0 - float(((y - prediction) ** 2).sum()) / denominator


def hierarchical_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    variant_key: str,
    positive_variant: str,
    negative_variant: str,
    seed_key: str,
    example_key: str,
    samples: int,
    random_seed: int,
) -> dict[str, float]:
    """Paired seed/example bootstrap for two variants on a common probe grid."""
    by_key: dict[tuple[Any, Any], dict[str, float]] = {}
    for row in rows:
        key = (row[seed_key], row[example_key])
        by_key.setdefault(key, {})[str(row[variant_key])] = float(row[value_key])
    paired = {
        key: values[positive_variant] - values[negative_variant]
        for key, values in by_key.items()
        if positive_variant in values and negative_variant in values
    }
    if not paired:
        raise RuntimeError("No paired rows are available for the requested contrast")
    seeds = sorted({key[0] for key in paired})
    values_by_seed: dict[Any, np.ndarray] = {}
    for seed in seeds:
        values_by_seed[seed] = np.asarray(
            [value for (row_seed, _), value in paired.items() if row_seed == seed],
            dtype=np.float64,
        )
    observed = float(np.mean([array.mean() for array in values_by_seed.values()]))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = values_by_seed[seed]
            sampled = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": float(len(seeds)),
        "n_pairs": float(len(paired)),
    }


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]

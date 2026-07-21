from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_id(*parts: str) -> str:
    text = "\n".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append checkpoint rows and make them durable before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                cursor = end - 1
                while cursor > 0:
                    cursor -= 1
                    f.seek(cursor)
                    if f.read(1) == b"\n":
                        cursor += 1
                        break
                else:
                    cursor = 0
                f.seek(cursor)
                tail = f.read(end - cursor)
                try:
                    json.loads(tail.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    f.seek(cursor)
                    f.truncate()
                else:
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
        f.seek(0, os.SEEK_END)
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_tolerant(path: str | Path) -> list[dict[str, Any]]:
    """Read append-only checkpoints while ignoring a truncated final line."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "rb") as f:
        lines = f.readlines()
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                is_incomplete_tail = line_number == len(
                    lines
                ) and not raw_line.endswith(b"\n")
                if not is_incomplete_tail:
                    raise RuntimeError(
                        f"Corrupt JSONL row {line_number} in {path}"
                    ) from exc
                print(f"Ignoring incomplete JSONL tail in {path}")
    return rows


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(
        min(pred_tokens.count(t), gold_tokens.count(t))
        for t in set(pred_tokens) & set(gold_tokens)
    )
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

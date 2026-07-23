from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import ensure_dir, load_config, read_jsonl, read_jsonl_tolerant
from stackpilot.experiment_registry import experiment_by_id, load_registry
from stackpilot.hard_policy_eval import balanced_limit, run_episode, validate_data_rows
from stackpilot.react_agent_eval import file_digest, model_identity
from stackpilot.retrieval_clients import RetrievalClient


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluation_signature(
    *,
    cfg: dict[str, Any],
    data_file: Path,
    rows: list[dict[str, Any]],
    experiment_id: str,
    run_id: str,
    tag: str,
    seed: int,
    backends: list[str],
    topks: list[int],
    inject_backend_id: bool,
) -> str:
    payload = {
        "schema": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tag": tag,
        "seed": seed,
        "model": model_identity(cfg),
        "data_path": str(data_file.resolve()),
        "data_sha256": file_digest(data_file),
        "question_ids": [str(row["id"]) for row in rows],
        "backends": sorted(backends),
        "topks": sorted(topks),
        "inject_backend_id": inject_backend_id,
        "agent": cfg["agent"],
        "llm": {
            "temperature": cfg["llm"]["temperature"],
            "max_tokens": cfg["llm"]["max_tokens"],
        },
    }
    return stable_signature(payload)


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    metrics = [
        "em",
        "f1",
        "support_recall",
        "turn1_support_recall",
        "turn2_support_recall",
        "turn3_support_recall",
        "turn2_evidence_gain",
        "turn3_evidence_gain",
        "recovery_at_2",
        "recovery_at_3",
        "full_recovery_at_2",
        "full_recovery_at_3",
        "search_count",
    ]
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(
            [
                "experiment_id",
                "run_id",
                "run_signature",
                "policy_tag",
                "seed",
                "dataset",
                "backend",
                "topk",
            ],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["dataset", "backend", "topk"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_rq0.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("bm25", "e5", "hybrid"),
        default=("bm25", "e5"),
    )
    parser.add_argument("--topks", nargs="+", type=int, default=(3, 5, 10))
    parser.add_argument("--bm25-port", type=int, default=8101)
    parser.add_argument("--e5-port", type=int, default=8102)
    parser.add_argument("--hybrid-port", type=int, default=8300)
    parser.add_argument(
        "--inject-backend-id",
        action="store_true",
        help="prepend the current backend as retrieval_environment metadata",
    )
    return parser.parse_args()


def archive_stale(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    archive_dir = output_path.parent / "archive"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_path = archive_dir / f"episodes.stale-{timestamp}.jsonl"
    atomic_jsonl(archive_path, rows)
    print(f"Archived {len(rows)} stale numbered-evaluation rows: {archive_path}")


def main() -> None:
    args = parse_args()
    experiment_by_id(load_registry(), args.experiment_id)
    cfg = load_config(args.config)
    data_file = Path(args.data_file).resolve()
    rows = balanced_limit(read_jsonl(data_file), args.limit)
    validate_data_rows(rows)
    backends = list(args.backends)
    topks = list(args.topks)
    if len(set(backends)) != len(backends):
        raise ValueError("backends must be unique")
    if len(set(topks)) != len(topks) or any(value < 1 for value in topks):
        raise ValueError("topks must be unique positive integers")

    run_signature = evaluation_signature(
        cfg=cfg,
        data_file=data_file,
        rows=rows,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        tag=args.tag,
        seed=args.seed,
        backends=backends,
        topks=topks,
        inject_backend_id=args.inject_backend_id,
    )
    expected_keys = {
        (str(item["id"]), backend, int(topk))
        for item in rows
        for backend in backends
        for topk in topks
    }
    urls = {
        "bm25": f"http://127.0.0.1:{args.bm25_port}/retrieve",
        "e5": f"http://127.0.0.1:{args.e5_port}/retrieve",
        "hybrid": f"http://127.0.0.1:{args.hybrid_port}/retrieve",
    }
    clients = {name: RetrievalClient(name, urls[name]) for name in backends}
    output_dir = ensure_dir(args.output_dir)
    output_path = Path(output_dir) / "episodes.jsonl"

    existing_rows = read_jsonl_tolerant(output_path) if output_path.is_file() else []
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    stale: list[dict[str, Any]] = []
    for row in existing_rows:
        try:
            key = (str(row["question_id"]), str(row["backend"]), int(row["topk"]))
        except (KeyError, TypeError, ValueError):
            stale.append(row)
            continue
        valid = (
            key in expected_keys
            and row.get("experiment_id") == args.experiment_id
            and row.get("run_id") == args.run_id
            and row.get("run_signature") == run_signature
            and row.get("policy_tag") == args.tag
            and int(row.get("seed", -1)) == args.seed
            and bool(row.get("backend_id_injected", False)) == args.inject_backend_id
        )
        if not valid or key in existing:
            stale.append(row)
            continue
        existing[key] = row
    archive_stale(output_path, stale)

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    current = dict(existing)
    for item in tqdm(rows, desc=f"{args.experiment_id}:{args.tag}"):
        for backend in backends:
            for topk in topks:
                key = (str(item["id"]), backend, int(topk))
                if key in current:
                    continue
                eval_item = dict(item)
                if args.inject_backend_id:
                    eval_item["question"] = (
                        f"<retrieval_environment>{backend}</retrieval_environment>\n"
                        f"{item['question']}"
                    )
                episode = run_episode(
                    client=client,
                    model=args.model,
                    retriever=clients[backend],
                    item=eval_item,
                    cfg=cfg,
                    topk=int(topk),
                    eval_seed=args.seed,
                )
                episode["question"] = str(item["question"])
                episode.update(
                    {
                        "schema": 1,
                        "experiment_id": args.experiment_id,
                        "run_id": args.run_id,
                        "run_signature": run_signature,
                        "policy_tag": args.tag,
                        "seed": args.seed,
                        "backend_id_injected": args.inject_backend_id,
                    }
                )
                current[key] = episode
                atomic_jsonl(output_path, [current[key] for key in sorted(current)])

    final_rows = [current[key] for key in sorted(current)]
    if set(current) != expected_keys:
        missing = sorted(expected_keys - set(current))[:10]
        raise RuntimeError(f"Numbered evaluation is incomplete; missing examples: {missing}")
    atomic_jsonl(output_path, final_rows)
    summary = summarize(final_rows)
    summary.to_csv(Path(output_dir) / "summary.csv", index=False)
    manifest = {
        "schema": 1,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "run_signature": run_signature,
        "policy_tag": args.tag,
        "seed": args.seed,
        "questions": len(rows),
        "episodes": len(final_rows),
        "backends": backends,
        "topks": topks,
        "backend_id_injected": args.inject_backend_id,
    }
    (Path(output_dir) / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

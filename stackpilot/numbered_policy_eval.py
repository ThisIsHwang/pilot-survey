from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import ensure_dir, load_config, read_jsonl
from stackpilot.experiment_registry import experiment_by_id, load_registry
from stackpilot.hard_policy_eval import balanced_limit, run_episode, validate_data_rows
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
            ["experiment_id", "run_id", "policy_tag", "seed", "dataset", "backend", "topk"],
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
    parser.add_argument("--backends", nargs="+", choices=("bm25", "e5", "hybrid"), default=("bm25", "e5"))
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


def main() -> None:
    args = parse_args()
    experiment_by_id(load_registry(), args.experiment_id)
    cfg = load_config(args.config)
    rows = balanced_limit(read_jsonl(args.data_file), args.limit)
    validate_data_rows(rows)
    if len(set(args.backends)) != len(args.backends):
        raise ValueError("backends must be unique")
    if len(set(args.topks)) != len(args.topks) or any(value < 1 for value in args.topks):
        raise ValueError("topks must be unique positive integers")

    urls = {
        "bm25": f"http://127.0.0.1:{args.bm25_port}/retrieve",
        "e5": f"http://127.0.0.1:{args.e5_port}/retrieve",
        "hybrid": f"http://127.0.0.1:{args.hybrid_port}/retrieve",
    }
    clients = {name: RetrievalClient(name, urls[name]) for name in args.backends}
    output_dir = ensure_dir(args.output_dir)
    output_path = Path(output_dir) / "episodes.jsonl"
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if output_path.is_file():
        for row in read_jsonl(output_path):
            if (
                row.get("experiment_id") == args.experiment_id
                and row.get("run_id") == args.run_id
                and row.get("policy_tag") == args.tag
                and int(row.get("seed", -1)) == args.seed
                and bool(row.get("backend_id_injected", False)) == args.inject_backend_id
            ):
                existing[(str(row["question_id"]), str(row["backend"]), int(row["topk"]))] = row

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    all_rows: list[dict[str, Any]] = []
    for item in tqdm(rows, desc=f"{args.experiment_id}:{args.tag}"):
        for backend in args.backends:
            for topk in args.topks:
                key = (str(item["id"]), backend, int(topk))
                if key in existing:
                    all_rows.append(existing[key])
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
                        "policy_tag": args.tag,
                        "seed": args.seed,
                        "backend_id_injected": args.inject_backend_id,
                    }
                )
                all_rows.append(episode)
                atomic_jsonl(output_path, all_rows)

    atomic_jsonl(output_path, all_rows)
    summary = summarize(all_rows)
    summary.to_csv(Path(output_dir) / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

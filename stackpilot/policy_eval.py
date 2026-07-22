from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import (
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.react_agent_eval import (
    RESULT_SCHEMA,
    file_digest,
    run_episode,
    run_identity,
)
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import check_retrievers, check_vllm, effective_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one served search policy on selected retrieval backends."
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("bm25", "e5", "colbert"),
        default=("bm25", "e5", "colbert"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("blind", "oracle_guidance"),
        default=("blind",),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[A-Za-z0-9._-]+", args.tag) is None:
        raise ValueError(f"Invalid policy tag: {args.tag!r}")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be positive; got {args.limit}")
    if len(set(args.backends)) != len(args.backends):
        raise ValueError(f"Duplicate backends are not allowed: {args.backends}")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError(f"Duplicate variants are not allowed: {args.variants}")
    cfg = load_config(args.config)
    check_retrievers(cfg, args.backends)
    check_vllm(cfg)

    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results" / "policies")
    output_path = out_dir / f"{args.tag}.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.csv"

    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    limit = int(cfg["agent"]["eval_examples"]) if args.limit is None else args.limit
    if limit < 1:
        raise ValueError(f"Evaluation limit must be positive; got {limit}")
    queries = queries[:limit]
    if not queries:
        raise RuntimeError("The selected policy-evaluation question set is empty")
    question_ids = [str(item["id"]) for item in queries]
    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError("Policy-evaluation question IDs must be unique")
    evaluation_context = {
        "schema": 1,
        "queries_eval_sha256": file_digest(work_dir / "data" / "queries_eval.jsonl"),
        "corpus_sha256": file_digest(work_dir / "data" / "corpus.jsonl"),
        "data_manifest_sha256": file_digest(work_dir / "data" / ".pilot-manifest.json"),
        "index_manifests": {
            backend: file_digest(
                work_dir / "indexes" / backend / ".pilot-manifest.json"
            )
            for backend in sorted(args.backends)
        },
        "evaluator_files": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "common.py",
                "policy_eval.py",
                "react_agent_eval.py",
                "retrieval_clients.py",
            )
        },
        "question_ids": question_ids,
        "backends": sorted(args.backends),
        "variants": sorted(args.variants),
        "protocol": {
            "seed": cfg["seed"],
            "retrieval_topk": cfg["retrieval"]["topk"],
            "agent": cfg["agent"],
            "llm_generation": {
                "temperature": cfg["llm"]["temperature"],
                "max_tokens": cfg["llm"]["max_tokens"],
            },
        },
    }
    evaluation_canonical = json.dumps(
        evaluation_context, sort_keys=True, separators=(",", ":")
    )
    evaluation_signature = hashlib.sha256(
        evaluation_canonical.encode("utf-8")
    ).hexdigest()
    run_signature, _ = run_identity(
        cfg, work_dir, args.backends, evaluation_context=evaluation_context
    )

    r_cfg = cfg["retrieval"]
    clients = {
        "bm25": RetrievalClient(
            "bm25", f"http://127.0.0.1:{r_cfg['bm25_port']}/retrieve"
        ),
        "e5": RetrievalClient("e5", f"http://127.0.0.1:{r_cfg['e5_port']}/retrieve"),
        "colbert": RetrievalClient(
            "colbert", f"http://127.0.0.1:{r_cfg['colbert_port']}/retrieve"
        ),
    }

    llm = OpenAI(
        base_url=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"],
        timeout=180.0,
        max_retries=5,
    )
    model = effective_model_name(cfg)
    selected_ids = {str(item["id"]) for item in queries}
    existing = read_jsonl_tolerant(output_path)
    row_by_key = {
        (
            str(row.get("question_id")),
            str(row.get("backend")),
            str(row.get("variant")),
        ): row
        for row in existing
        if row.get("schema") == RESULT_SCHEMA
        and row.get("run_signature") == run_signature
        and row.get("evaluation_signature") == evaluation_signature
        and row.get("policy_tag") == args.tag
        and str(row.get("question_id")) in selected_ids
        and str(row.get("backend")) in args.backends
        and str(row.get("variant")) in args.variants
    }

    total = len(queries) * len(args.backends) * len(args.variants)
    progress = tqdm(total=total, desc=f"policy eval: {args.tag}")
    for item in queries:
        for backend in args.backends:
            retriever = clients[backend]
            for variant in args.variants:
                key = (str(item["id"]), backend, variant)
                if key not in row_by_key:
                    row = run_episode(
                        llm,
                        model,
                        retriever,
                        item,
                        cfg,
                        oracle=variant == "oracle_guidance",
                    )
                    row.update(
                        {
                            "schema": RESULT_SCHEMA,
                            "run_signature": run_signature,
                            "evaluation_signature": evaluation_signature,
                            "policy_tag": args.tag,
                            "served_model": model,
                        }
                    )
                    append_jsonl(output_path, [row])
                    row_by_key[key] = row
                progress.update(1)
    progress.close()

    rows = list(row_by_key.values())
    if len(rows) != total:
        raise RuntimeError(
            f"Expected {total} policy-evaluation rows, found {len(rows)}"
        )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["backend", "variant"])[
            ["em", "f1", "support_recall", "search_count"]
        ]
        .mean()
        .reset_index()
    )
    summary.insert(0, "policy_tag", args.tag)
    summary.insert(1, "run_signature", run_signature)
    summary.insert(2, "evaluation_signature", evaluation_signature)
    summary.insert(3, "n_questions", len(queries))
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Episode results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

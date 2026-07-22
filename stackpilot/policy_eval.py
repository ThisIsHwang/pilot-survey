from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from stackpilot.common import append_jsonl, ensure_dir, load_config, read_jsonl, read_jsonl_tolerant
from stackpilot.react_agent_eval import RESULT_SCHEMA, run_episode, run_identity
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
    cfg = load_config(args.config)
    check_retrievers(cfg)
    check_vllm(cfg)

    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results" / "policies")
    output_path = out_dir / f"{args.tag}.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.csv"

    run_signature, _ = run_identity(cfg, work_dir)
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    limit = int(cfg["agent"]["eval_examples"]) if args.limit is None else args.limit
    queries = queries[:limit]

    r_cfg = cfg["retrieval"]
    clients = {
        "bm25": RetrievalClient("bm25", f"http://127.0.0.1:{r_cfg['bm25_port']}/retrieve"),
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
        and str(row.get("question_id")) in selected_ids
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
                            "policy_tag": args.tag,
                            "served_model": model,
                        }
                    )
                    append_jsonl(output_path, [row])
                    row_by_key[key] = row
                progress.update(1)
    progress.close()

    rows = list(row_by_key.values())
    if not rows:
        raise RuntimeError("No policy-evaluation rows were produced")

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
    summary.insert(2, "n_questions", len(queries))
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Episode results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

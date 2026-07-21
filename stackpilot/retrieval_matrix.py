from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from stackpilot.common import (
    append_jsonl,
    ensure_dir,
    load_config,
    read_jsonl,
    read_jsonl_tolerant,
)
from stackpilot.query_styles import LLMQueryGenerator, heuristic_candidates
from stackpilot.retrieval_clients import RetrievalClient
from stackpilot.service_checks import (
    check_retrievers,
    check_vllm,
    effective_model_name,
)


RESULT_SCHEMA = 2


def file_digest(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required run-identity file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(cfg: dict) -> dict:
    configured_model = str(cfg["llm"]["model"])
    model_path = (
        os.environ.get("MODEL_PATH") or os.environ.get("MODEL") or configured_model
    )
    identity = {
        "configured_model": configured_model,
        "model_path": model_path,
        "served_model_name": os.environ.get("SERVED_MODEL_NAME") or configured_model,
    }
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        identity["resolved_model_path"] = str(local_path.resolve())
        model_files = {
            path
            for pattern in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template*",
                "*.index.json",
                "*.safetensors",
                "*.bin",
            )
            for path in local_path.glob(pattern)
            if path.is_file()
        }
        local_files = []
        for path in sorted(model_files):
            state = {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if path.suffix not in {".safetensors", ".bin"}:
                state["sha256"] = file_digest(path)
            local_files.append(state)
        identity["local_files"] = local_files
    return identity


def run_identity(cfg: dict, work_dir: Path) -> tuple[str, dict]:
    artifacts = {
        "corpus": work_dir / "data" / "corpus.jsonl",
        "queries_eval": work_dir / "data" / "queries_eval.jsonl",
        "data_manifest": work_dir / "data" / ".pilot-manifest.json",
        "bm25_manifest": work_dir / "indexes" / "bm25" / ".pilot-manifest.json",
        "e5_manifest": work_dir / "indexes" / "e5" / ".pilot-manifest.json",
        "colbert_manifest": work_dir / "indexes" / "colbert" / ".pilot-manifest.json",
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "config": cfg,
        "artifacts": {name: file_digest(path) for name, path in artifacts.items()},
        "model": model_identity(cfg),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def support_recall(results: list[dict], support_titles: list[str]) -> float:
    gold = {x.strip().lower() for x in support_titles}
    if not gold:
        return 0.0
    retrieved = {str(x.get("title", "")).strip().lower() for x in results}
    return len(gold & retrieved) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    check_retrievers(cfg)

    work_dir = Path(cfg["work_dir"]).resolve()
    out_dir = ensure_dir(work_dir / "results")
    output_path = out_dir / "retrieval_matrix.jsonl"
    run_signature, _ = run_identity(cfg, work_dir)
    queries = read_jsonl(work_dir / "data" / "queries_eval.jsonl")
    if args.limit is not None:
        queries = queries[: args.limit]
    selected_ids = {str(item["id"]) for item in queries}

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
    source = cfg["query_generation"]["source"]
    if source not in {"heuristic", "vllm"}:
        raise ValueError(f"Unknown query_generation.source: {source}")
    generator = None
    if source == "vllm":
        check_vllm(cfg)
        generator = LLMQueryGenerator(
            api_base=cfg["llm"]["api_base"],
            api_key=cfg["llm"]["api_key"],
            model=effective_model_name(cfg),
            temperature=cfg["llm"]["temperature"],
            seed=int(cfg["seed"]),
        )

    existing = read_jsonl_tolerant(output_path)
    row_by_key = {
        (
            str(row.get("question_id")),
            str(row.get("backend")),
            str(row.get("style")),
        ): row
        for row in existing
        if row.get("schema") == RESULT_SCHEMA
        and row.get("run_signature") == run_signature
        and str(row.get("question_id")) in selected_ids
    }

    cache_path = Path(cfg["query_generation"]["cache_file"]).resolve()
    cached_candidates = {}
    if generator:
        for row in read_jsonl_tolerant(cache_path):
            candidates = row.get("candidates")
            if (
                row.get("schema") == RESULT_SCHEMA
                and row.get("run_signature") == run_signature
                and isinstance(candidates, dict)
                and set(candidates) == {"semantic", "keyword", "exact", "decompose"}
            ):
                cached_candidates[str(row.get("question_id"))] = candidates

    styles = ("semantic", "keyword", "exact", "decompose")
    for item in tqdm(queries, desc="retrieval matrix"):
        question_id = str(item["id"])
        if generator:
            candidates = cached_candidates.get(question_id)
            if candidates is None:
                candidates = generator.generate(item["question"])
                append_jsonl(
                    cache_path,
                    [
                        {
                            "schema": RESULT_SCHEMA,
                            "run_signature": run_signature,
                            "question_id": question_id,
                            "question": item["question"],
                            "candidates": candidates,
                        }
                    ],
                )
                cached_candidates[question_id] = candidates
        else:
            candidates = heuristic_candidates(item["question"])

        for backend, client in clients.items():
            required_keys = [
                (question_id, backend, style) for style in (*styles, "rrf_ensemble")
            ]
            if all(key in row_by_key for key in required_keys):
                continue

            new_rows = []
            style_results = {}
            for style in styles:
                query = candidates[style]
                results = client.search(query, int(r_cfg["topk"]))
                style_results[style] = results
                recall = support_recall(results, item["support_titles"])
                new_rows.append(
                    {
                        "schema": RESULT_SCHEMA,
                        "run_signature": run_signature,
                        "question_id": question_id,
                        "backend": backend,
                        "style": style,
                        "query": query,
                        "support_recall": recall,
                        "support_hit": float(recall > 0),
                        "full_support": float(recall == 1.0),
                        "retrieved_titles": [result["title"] for result in results],
                    }
                )

            rrf_scores = defaultdict(float)
            doc_payload = {}
            for results in style_results.values():
                for result in results:
                    key = result["id"] or result["title"].strip().lower()
                    if not key:
                        continue
                    rrf_scores[key] += 1.0 / (60.0 + result["rank"])
                    doc_payload[key] = result
            fused = [
                doc_payload[key]
                for key, _ in sorted(
                    rrf_scores.items(), key=lambda value: value[1], reverse=True
                )[: int(r_cfg["topk"])]
            ]
            recall = support_recall(fused, item["support_titles"])
            new_rows.append(
                {
                    "schema": RESULT_SCHEMA,
                    "run_signature": run_signature,
                    "question_id": question_id,
                    "backend": backend,
                    "style": "rrf_ensemble",
                    "query": " | ".join(candidates.values()),
                    "support_recall": recall,
                    "support_hit": float(recall > 0),
                    "full_support": float(recall == 1.0),
                    "retrieved_titles": [result["title"] for result in fused],
                }
            )
            append_jsonl(output_path, new_rows)
            for row in new_rows:
                row_by_key[(question_id, backend, row["style"])] = row

    rows = list(row_by_key.values())
    if not rows:
        raise RuntimeError("No retrieval-matrix rows were produced")
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["backend", "style"])[
            ["support_recall", "support_hit", "full_support"]
        ]
        .mean()
        .reset_index()
    )
    summary.insert(0, "run_signature", run_signature)
    summary.insert(1, "n_questions", len(queries))
    summary.to_csv(out_dir / "retrieval_matrix_summary.csv", index=False)

    style_rows = frame[frame["style"] != "rrf_ensemble"]
    oracle = (
        style_rows.groupby(["question_id", "backend"])["support_recall"]
        .max()
        .groupby("backend")
        .mean()
        .rename("style_oracle_support_recall")
        .reset_index()
    )
    oracle.insert(0, "run_signature", run_signature)
    oracle.insert(1, "n_questions", len(queries))
    oracle.to_csv(out_dir / "retrieval_style_oracle.csv", index=False)
    fixed = frame.groupby(["backend", "style"])["support_recall"].mean().unstack()
    print("\nMean supporting-title recall")
    print(fixed.round(4).to_string())
    print("\nPer-question query-style oracle (ensemble excluded)")
    print(oracle.round(4).to_string(index=False))
    print(f"\nSaved resumable results to {out_dir}")


if __name__ == "__main__":
    main()

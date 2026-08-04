from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    balanced_subset,
    cluster_bootstrap,
    load_config,
    load_state_results,
    markdown_table,
    normalize_title,
    source_patterns,
    stable_hash,
)
from stackpilot.response_feedback_runtime import feedback_text
from stackpilot.retrieval_clients import RetrievalClient

EXPERIMENT_ID = "EXP-030"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def visible_prefix(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    titles: list[str] = []
    prefix = result.get("prefix")
    records = prefix.get("records", []) if isinstance(prefix, dict) else []
    if not records:
        records = result["state"].get("prior_turns", []) or []
    for record in records:
        if not isinstance(record, dict):
            continue
        query = str(record.get("query", "")).strip()
        if query:
            queries.append(query)
        titles.extend(_strings(record.get("observed_titles")))
    return queries, titles


def state_prompt(result: dict[str, Any], extra_feedback: str = "") -> str:
    state = result["state"]
    previous_queries, previous_titles = visible_prefix(result)
    lines = [
        "You are a search agent. Produce exactly one next web-search query.",
        "Do not answer the question. Return only the query text without XML tags.",
        f"Question: {state['question']}",
    ]
    if previous_queries:
        lines.append("Previous queries: " + " || ".join(previous_queries[-4:]))
    if previous_titles:
        lines.append(
            "Previously observed document titles: "
            + "; ".join(previous_titles[-16:])
        )
    if extra_feedback:
        lines.append(extra_feedback.strip())
    lines.append("Next query:")
    return "\n".join(lines)


def parse_query(text: str) -> str:
    value = str(text).strip()
    if "<search>" in value:
        value = value.split("<search>", 1)[1].split("</search>", 1)[0]
    value = value.splitlines()[0] if value.splitlines() else value
    value = value.strip().strip('"').strip("'")
    if not value or "<answer>" in value or len(value) > 512:
        return ""
    return " ".join(value.split())


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def generate_queries(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    cfg: dict[str, Any],
    seed: int,
) -> list[str]:
    import torch

    if not prompts:
        return []
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg["response_feedback"]["prompt_token_budget"]),
    )
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(cfg["model"]["temperature"]),
            top_p=float(cfg["model"]["top_p"]),
            max_new_tokens=int(cfg["model"]["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
        )
    prompt_width = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        output[:, prompt_width:], skip_special_tokens=True
    )
    return [parse_query(value) for value in decoded]


def execute_query(
    client: RetrievalClient,
    query: str,
    *,
    topk: int,
    support_titles: Sequence[str],
    prefix_titles: Sequence[str],
) -> dict[str, Any]:
    gold = {normalize_title(value) for value in support_titles}
    prefix = {normalize_title(value) for value in prefix_titles}
    before = len(gold & prefix) / max(1, len(gold))
    if not query:
        return {
            "query": "",
            "invalid": 1,
            "titles": [],
            "signature": (),
            "evidence_gain": 0.0,
            "support_recall_after": float(before),
        }
    results = client.search(query, topk)
    titles = [str(row.get("title", "")).strip() for row in results]
    titles = [title for title in titles if title]
    signature = tuple(normalize_title(title) for title in titles)
    after = len(gold & (prefix | set(signature))) / max(1, len(gold))
    return {
        "query": query,
        "invalid": 0,
        "titles": titles,
        "signature": signature,
        "evidence_gain": float(after - before),
        "support_recall_after": float(after),
    }


def mode_rollout(
    result: dict[str, Any],
    *,
    mode: str,
    model: Any,
    tokenizer: Any,
    client: RetrievalClient,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = result["state"]
    total = int(cfg["response_feedback"]["total_queries"])
    first_count = int(cfg["response_feedback"]["first_phase_queries"])
    if not 0 < first_count < total:
        raise RuntimeError(
            "response-feedback query budget must have two nonempty phases"
        )
    _previous_queries, prefix_titles = visible_prefix(result)
    base_prompt = state_prompt(result)
    # Common random numbers: all three conditions use identical first-phase
    # samples and the same second-phase decoding seed. Only the visible feedback
    # text differs.
    state_seed = int(
        stable_hash(EXPERIMENT_ID, state["state_id"], "initial", length=15), 16
    )
    second_seed = int(
        stable_hash(EXPERIMENT_ID, state["state_id"], "second", length=15), 16
    )
    first_queries = generate_queries(
        model,
        tokenizer,
        [base_prompt] * first_count,
        cfg=cfg,
        seed=state_seed,
    )
    first_rows = [
        execute_query(
            client,
            query,
            topk=int(state["topk"]),
            support_titles=state.get("support_titles", []),
            prefix_titles=prefix_titles,
        )
        for query in first_queries
    ]

    if mode == "iid":
        second_feedback = ""
    elif mode == "text-feedback":
        attempted = [row["query"] for row in first_rows if row["query"]]
        second_feedback = feedback_text(
            attempted,
            maximum_chars=int(
                cfg["response_feedback"]["maximum_feedback_chars"]
            ),
            mode="text-feedback",
        )
    elif mode == "response-feedback":
        observed: list[str] = []
        for row in first_rows:
            observed.extend(row["titles"])
        second_feedback = feedback_text(
            observed[
                : int(cfg["response_feedback"]["maximum_feedback_titles"])
            ],
            maximum_chars=int(
                cfg["response_feedback"]["maximum_feedback_chars"]
            ),
            mode="response-feedback",
        )
    else:
        raise ValueError(f"Unknown feedback mode: {mode}")

    second_queries = generate_queries(
        model,
        tokenizer,
        [state_prompt(result, second_feedback)] * (total - first_count),
        cfg=cfg,
        seed=second_seed,
    )
    second_rows = [
        execute_query(
            client,
            query,
            topk=int(state["topk"]),
            support_titles=state.get("support_titles", []),
            prefix_titles=prefix_titles,
        )
        for query in second_queries
    ]
    rows = first_rows + second_rows
    for index, row in enumerate(rows):
        row.update(
            {
                "state_id": str(state["state_id"]),
                "question_id": str(state["question_id"]),
                "backend": str(state["backend"]),
                "dataset": str(state["dataset"]),
                "mode": mode,
                "phase": "initial" if index < first_count else "feedback",
                "query_index": index,
            }
        )

    signatures = {
        tuple(row["signature"]) for row in rows if not row["invalid"]
    }
    gold = {normalize_title(value) for value in state.get("support_titles", [])}
    union_titles = {normalize_title(value) for value in prefix_titles}
    for row in rows:
        union_titles.update(row["signature"])
    union_recall = len(gold & union_titles) / max(1, len(gold))
    summary = {
        "state_id": str(state["state_id"]),
        "question_id": str(state["question_id"]),
        "backend": str(state["backend"]),
        "dataset": str(state["dataset"]),
        "mode": mode,
        "queries": total,
        "unique_behaviors": len(signatures),
        "behavior_coverage": len(signatures) / max(1, total),
        "duplicate_rate": 1.0 - len(signatures) / max(1, total),
        "union_support_recall": float(union_recall),
        "best_evidence_gain": max(
            (row["evidence_gain"] for row in rows), default=0.0
        ),
        "any_gain": int(
            any(float(row["evidence_gain"]) > 0 for row in rows)
        ),
        "invalid_rate": float(np.mean([row["invalid"] for row in rows])),
    }
    return rows, summary


def _paired_bootstrap(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = frame.pivot_table(
        index=["state_id", "backend"],
        columns="mode",
        values=metric,
        aggfunc="first",
    )
    paired = pivot[[left, right]].dropna().reset_index()
    rows = [
        {
            "cluster": str(row["state_id"]),
            "difference": float(row[left] - row[right]),
        }
        for _, row in paired.iterrows()
    ]
    return cluster_bootstrap(
        rows,
        cluster_key="cluster",
        statistic=lambda values: float(
            np.mean([item["difference"] for item in values])
        ),
        samples=samples,
        seed=seed,
    )


def run(
    cfg: dict[str, Any],
    profile_name: str,
    inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    profile = cfg["profiles"][profile_name]
    model_ref = os.environ.get("BASE_MODEL", "").strip() or str(
        cfg["model"]["base_model"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=cfg["model"].get("revision"),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if str(cfg["model"].get("dtype")) == "bfloat16"
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=cfg["model"].get("revision"),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    model.eval()

    clients = {
        "bm25": RetrievalClient(
            "bm25",
            str(cfg["retrieval"]["bm25_url"]),
            timeout=int(cfg["retrieval"]["timeout"]),
            retries=int(cfg["retrieval"]["retries"]),
        ),
        "e5": RetrievalClient(
            "e5",
            str(cfg["retrieval"]["e5_url"]),
            timeout=int(cfg["retrieval"]["timeout"]),
            retries=int(cfg["retrieval"]["retries"]),
        ),
    }
    results = balanced_subset(
        load_state_results(source_patterns(cfg, inputs)), int(profile["states"])
    )
    query_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for result in results:
        backend = str(result["state"]["backend"])
        for mode in cfg["response_feedback"]["modes"]:
            rows, summary = mode_rollout(
                result,
                mode=str(mode),
                model=model,
                tokenizer=tokenizer,
                client=clients[backend],
                cfg=cfg,
            )
            query_rows.extend(rows)
            summaries.append(summary)

    frame = pd.DataFrame(summaries)
    query_frame = pd.DataFrame(query_rows)
    output_dir = (
        Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "state_metrics.csv", index=False)
    query_frame.to_csv(output_dir / "query_rows.csv", index=False)
    means = frame.groupby(["backend", "mode"], as_index=False).mean(
        numeric_only=True
    )
    means.to_csv(output_dir / "variant_means.csv", index=False)

    contrasts: list[dict[str, Any]] = []
    metrics = (
        "behavior_coverage",
        "union_support_recall",
        "duplicate_rate",
        "invalid_rate",
    )
    for backend in sorted(frame["backend"].unique()):
        backend_frame = frame[frame["backend"] == backend]
        for left, right in (
            ("response-feedback", "iid"),
            ("response-feedback", "text-feedback"),
        ):
            for metric_index, metric in enumerate(metrics):
                result = _paired_bootstrap(
                    backend_frame,
                    left=left,
                    right=right,
                    metric=metric,
                    samples=int(profile["bootstrap_samples"]),
                    seed=30030 + len(contrasts) + metric_index,
                )
                contrasts.append(
                    {
                        "backend": backend,
                        "contrast": f"{left}-minus-{right}",
                        "metric": metric,
                        **result,
                    }
                )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output_dir / "paired_contrasts.csv", index=False)

    gate = cfg["gates"][EXPERIMENT_ID]
    iid = contrast_frame[
        contrast_frame["contrast"] == "response-feedback-minus-iid"
    ]
    coverage = iid[iid["metric"] == "behavior_coverage"]
    recall = iid[iid["metric"] == "union_support_recall"]
    invalid = iid[iid["metric"] == "invalid_rate"]
    go = bool(
        len(coverage) >= 2
        and (
            coverage["estimate"]
            >= float(gate["minimum_feedback_coverage_gain"])
        ).all()
        and (coverage["ci_low"] > 0).all()
        and len(recall) >= 2
        and (
            recall["estimate"]
            >= float(gate["minimum_feedback_union_recall_gain"])
        ).all()
        and (recall["ci_low"] > 0).all()
        and len(invalid) >= 2
        and (
            invalid["estimate"]
            <= float(gate["maximum_invalid_increase"])
        ).all()
    )
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "states": int(frame["state_id"].nunique()),
        "queries": int(len(query_frame)),
        "go": go,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-030 Response-feedback rollout audit",
        "",
        f"Profile: `{profile_name}`. Every method spends exactly eight target-retriever calls per state. Response feedback contains only document titles visible to first-phase sibling rollouts.",
        "",
        "## Means",
        "",
        markdown_table(means),
        "",
        "## Paired contrasts",
        "",
        markdown_table(contrast_frame),
        "",
        f"Decision: **{'GO' if go else 'NO-GO'}**.",
        "",
    ]
    (output_dir / "EXP030_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    payload = run(load_config(args.config), args.profile, args.input)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

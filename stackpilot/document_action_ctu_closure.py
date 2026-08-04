from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stackpilot.behavior_quotient_common import (
    atomic_write_json,
    candidate_reward,
    cluster_bootstrap,
    load_config,
    load_state_results,
    markdown_table,
    ranked_transition,
    source_patterns,
)

EXPERIMENT_ID = "EXP-034"


def _discover_document_ctu(
    cfg: dict[str, Any], profile: str, provided: Sequence[str] | None = None
) -> list[Path]:
    patterns = list(provided or [])
    if not patterns:
        environment = os.environ.get("BEHAVIOR_FEEDBACK_DOCUMENT_CTU", "").strip()
        if environment:
            patterns = [
                part
                for part in environment.replace("\n", os.pathsep).split(os.pathsep)
                if part
            ]
        else:
            patterns = [
                str(value).format(profile=profile)
                for value in cfg["ctu_closure"]["document_ctu_globs"]
            ]
    output: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file():
                output[str(path)] = path
    return [output[key] for key in sorted(output)]


def _read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def _factual_candidate(result: dict[str, Any]) -> dict[str, Any]:
    for candidate in result["candidates"]:
        if int(candidate.get("protocol_failure", 0)) != 0:
            continue
        if str(candidate.get("origin", "")) == "factual" or str(
            candidate.get("style", "")
        ) == "factual":
            return candidate
    raise RuntimeError(f"State {result['state']['state_id']} has no valid factual query")


def _class_tqe(result: dict[str, Any], factual: dict[str, Any]) -> float:
    signature = ranked_transition(factual)
    members = [
        candidate
        for candidate in result["candidates"]
        if int(candidate.get("protocol_failure", 0)) == 0
        and ranked_transition(candidate) == signature
    ]
    if not members:
        return float(factual.get("support_tqe", 0.0))
    values = [float(candidate.get("support_tqe", 0.0)) for candidate in members]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite behavior-class TQE")
    return float(np.mean(values))


def state_rows(
    cfg: dict[str, Any],
    state_results: Sequence[dict[str, Any]],
    document_rows: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    by_state = {str(result["state"]["state_id"]): result for result in state_results}
    doc_frame = pd.DataFrame(document_rows)
    required = {"state_id", "document_ctu"}
    missing = required - set(doc_frame.columns)
    if missing:
        raise RuntimeError(f"Document CTU rows miss {sorted(missing)}")
    for column in ("document_ctu", "support_ctu", "answer_ctu", "search_ctu"):
        if column not in doc_frame.columns:
            doc_frame[column] = 0.0
        doc_frame[column] = pd.to_numeric(doc_frame[column], errors="raise")
        if not np.isfinite(doc_frame[column]).all():
            raise RuntimeError(f"Non-finite document CTU metric: {column}")

    threshold = float(cfg["ctu_closure"]["minimum_positive_document_ctu"])
    output: list[dict[str, Any]] = []
    for state_id, group in doc_frame.groupby(doc_frame["state_id"].astype(str)):
        result = by_state.get(str(state_id))
        if result is None:
            continue
        factual = _factual_candidate(result)
        query_tqe = float(factual.get("support_tqe", 0.0))
        query_composite_tqe = float(factual.get("composite_tqe", 0.0))
        class_tqe = _class_tqe(result, factual)
        documents = group.sort_values(
            ["document_ctu", "document_rank"], ascending=[False, True]
        )
        best = documents.iloc[0]
        positive_documents = documents[documents["document_ctu"] > threshold]
        doc_positive = int(len(positive_documents) > 0)
        query_positive = int(query_tqe > 0.0)
        class_positive = int(class_tqe > 0.0)
        output.append(
            {
                "state_id": str(state_id),
                "question_id": str(result["state"]["question_id"]),
                "backend": str(result["state"]["backend"]),
                "dataset": str(result["state"]["dataset"]),
                "document_count": int(len(documents)),
                "positive_document_count": int(len(positive_documents)),
                "maximum_document_ctu": float(best["document_ctu"]),
                "maximum_support_ctu": float(documents["support_ctu"].max()),
                "maximum_answer_ctu": float(documents["answer_ctu"].max()),
                "factual_query_tqe": query_tqe,
                "factual_query_composite_tqe": query_composite_tqe,
                "behavior_class_tqe": class_tqe,
                "document_positive": doc_positive,
                "query_positive": query_positive,
                "class_positive": class_positive,
                "document_query_disagreement": int(doc_positive and not query_positive),
                "document_class_disagreement": int(doc_positive and not class_positive),
                "query_class_sign_disagreement": int(query_positive != class_positive),
                "best_document_title": str(best.get("document_title", "")),
            }
        )
    if not output:
        raise RuntimeError("No document CTU states joined the causal-query results")
    return pd.DataFrame(output)


def _bootstrap_rate(
    frame: pd.DataFrame,
    column: str,
    *,
    samples: int,
    seed: int,
    condition: str | None = None,
) -> dict[str, float]:
    subset = frame if condition is None else frame[frame[condition] == 1]
    if subset.empty:
        return {
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_clusters": 0.0,
            "n_rows": 0.0,
            "finite_draws": 0.0,
        }
    rows = [
        {"cluster": str(row.state_id), "value": float(getattr(row, column))}
        for row in subset.itertuples()
    ]
    return cluster_bootstrap(
        rows,
        cluster_key="cluster",
        statistic=lambda values: float(np.mean([item["value"] for item in values])),
        samples=samples,
        seed=seed,
    )


def run(
    cfg: dict[str, Any],
    profile_name: str,
    *,
    inputs: Sequence[str] | None = None,
    document_inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = cfg["profiles"][profile_name]
    state_results = load_state_results(source_patterns(cfg, inputs))
    paths = _discover_document_ctu(cfg, profile_name, document_inputs)
    if not paths:
        raise RuntimeError(
            "No document CTU JSONL was found. Run interface_causality/run_document_ctu.sh first."
        )
    frame = state_rows(cfg, state_results, _read_jsonl(paths))
    maximum_states = int(profile["ctu_states"])
    if maximum_states > 0 and len(frame) > maximum_states:
        frame = frame.sort_values("state_id").head(maximum_states).copy()

    samples = int(profile["bootstrap_samples"])
    all_state_disagreement = _bootstrap_rate(
        frame,
        "document_query_disagreement",
        samples=samples,
        seed=34034,
    )
    conditional_disagreement = _bootstrap_rate(
        frame,
        "document_query_disagreement",
        samples=samples,
        seed=34035,
        condition="document_positive",
    )
    class_disagreement = _bootstrap_rate(
        frame,
        "document_class_disagreement",
        samples=samples,
        seed=34036,
    )
    query_class = _bootstrap_rate(
        frame,
        "query_class_sign_disagreement",
        samples=samples,
        seed=34037,
    )

    correlations = []
    for right in ("factual_query_tqe", "behavior_class_tqe"):
        value = frame[["maximum_document_ctu", right]].corr(method="spearman").iloc[0, 1]
        correlations.append(
            {
                "left": "maximum_document_ctu",
                "right": right,
                "spearman": 0.0 if pd.isna(value) else float(value),
            }
        )
    correlations_frame = pd.DataFrame(correlations)

    closure = float(cfg["gates"][EXPERIMENT_ID]["closure_threshold"])
    reopening = float(cfg["gates"][EXPERIMENT_ID]["reopening_threshold"])
    estimate = float(all_state_disagreement["estimate"])
    if estimate < closure and float(all_state_disagreement["ci_high"]) < reopening:
        disposition = "CLOSE-CREDIT-DIRECTION"
    elif estimate >= reopening and float(all_state_disagreement["ci_low"]) > closure:
        disposition = "REOPEN-OBSERVATION-LEVEL-ANALYSIS"
    else:
        disposition = "INCONCLUSIVE"

    output_dir = Path(cfg["work_dir"]).resolve() / "reports" / profile_name / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "state_credit_levels.csv", index=False)
    correlations_frame.to_csv(output_dir / "credit_correlations.csv", index=False)
    metrics = pd.DataFrame(
        [
            {"metric": "document-positive_query-nonpositive_all_states", **all_state_disagreement},
            {"metric": "document-positive_query-nonpositive_conditional", **conditional_disagreement},
            {"metric": "document-positive_class-nonpositive_all_states", **class_disagreement},
            {"metric": "query-class_sign_disagreement", **query_class},
        ]
    )
    metrics.to_csv(output_dir / "prevalence.csv", index=False)
    payload = {
        "schema": 1,
        "experiment_id": EXPERIMENT_ID,
        "profile": profile_name,
        "states": int(frame["state_id"].nunique()),
        "document_ctu_files": len(paths),
        "disposition": disposition,
        "go": disposition == "REOPEN-OBSERVATION-LEVEL-ANALYSIS",
        "all_state_disagreement": all_state_disagreement,
        "conditional_disagreement": conditional_disagreement,
    }
    atomic_write_json(output_dir / "decision.json", payload)
    report = [
        "# EXP-034 Document–action credit closure",
        "",
        f"Profile: `{profile_name}`. The same states compare document-omission CTU, factual-query TQE, and response-induced behavior-class TQE.",
        "",
        "## Prevalence",
        "",
        markdown_table(metrics),
        "",
        "## Correlations",
        "",
        markdown_table(correlations_frame),
        "",
        f"Disposition: **{disposition}**.",
        "",
        "Below 10% closes the causal-credit direction; at least 20% reopens only an observation-level analysis, not query-level GRPO.",
        "",
    ]
    (output_dir / "EXP034_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_feedback.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--input", action="append", default=None)
    parser.add_argument("--document-input", action="append", default=None)
    args = parser.parse_args()
    payload = run(
        load_config(args.config),
        args.profile,
        inputs=args.input,
        document_inputs=args.document_input,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stackpilot.behavior_alias_common import atomic_write_json, load_config
from stackpilot.behavior_alias_simulation import (
    bootstrap_contrast,
    cell_summary,
    load_results,
    natural_rows,
    natural_summary,
    qualitative_examples,
    simulate_state,
)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report EXP-015 behavior-alias pilot.")
    parser.add_argument("--config", default="configs/behavior_alias_pilot.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--states-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = cfg["profiles"][args.profile]
    work_root = Path(cfg["work_dir"]).resolve()
    states_root = Path(args.states_root or work_root / "states" / args.profile)
    manifest = json.loads((states_root / "manifest.json").read_text(encoding="utf-8"))
    expected_states = int(manifest["selected_states"])
    result_root = Path(
        args.result_root or work_root / "results" / args.profile / "states"
    )
    results = load_results(result_root, expected_states)
    natural = natural_rows(results)

    methods = [str(value) for value in cfg["simulation"]["methods"]]
    multiplicities = [int(value) for value in cfg["simulation"]["multiplicities"]]
    if sorted(multiplicities) != multiplicities or len(set(multiplicities)) != len(multiplicities):
        raise RuntimeError("Alias multiplicities must be sorted unique integers")
    simulation_rows: list[dict[str, Any]] = []
    for state in results:
        simulation_rows.extend(
            simulate_state(
                state,
                methods=methods,
                multiplicities=multiplicities,
                budget=int(cfg["simulation"]["call_budget"]),
                draws=int(profile["simulation_draws"]),
            )
        )
    simulation = pd.DataFrame(simulation_rows)
    if simulation.empty:
        raise RuntimeError("No eligible states remained for alias-injection simulation")
    if not np.isfinite(simulation.select_dtypes(include=["number"])).all().all():
        raise RuntimeError("Simulation produced non-finite values")

    minimum_m = min(multiplicities)
    maximum_m = max(multiplicities)
    bootstrap_samples = int(profile["bootstrap_samples"])
    specs = [
        ("surface_coverage_drop", "class_coverage", "surface", minimum_m, "surface", maximum_m),
        ("quotient_coverage_drop", "class_coverage", "quotient", minimum_m, "quotient", maximum_m),
        ("quotient_coverage_gain_at_max", "class_coverage", "quotient", maximum_m, "surface", maximum_m),
        ("text_diverse_coverage_gain_at_max", "class_coverage", "text-diverse", maximum_m, "surface", maximum_m),
        ("quotient_utility_gain_at_max", "union_support_gain", "quotient", maximum_m, "surface", maximum_m),
        ("quotient_best_gain_at_max", "best_immediate_gain", "quotient", maximum_m, "surface", maximum_m),
    ]
    contrasts = []
    for index, (name, metric, method_a, mult_a, method_b, mult_b) in enumerate(specs):
        values = bootstrap_contrast(
            simulation,
            metric=metric,
            method_a=method_a,
            multiplicity_a=mult_a,
            method_b=method_b,
            multiplicity_b=mult_b,
            samples=bootstrap_samples,
            seed=15000 + index,
        )
        contrasts.append({"contrast": name, **values})
    contrast_frame = pd.DataFrame(contrasts)
    contrast_lookup = {
        str(row.contrast): row for row in contrast_frame.itertuples(index=False)
    }

    backend_utility = []
    for backend, group in simulation.groupby("backend"):
        values = bootstrap_contrast(
            group,
            metric="union_support_gain",
            method_a="quotient",
            multiplicity_a=maximum_m,
            method_b="surface",
            multiplicity_b=maximum_m,
            samples=bootstrap_samples,
            seed=15100 + len(backend_utility),
        )
        backend_utility.append({"backend": backend, **values})
    backend_utility_frame = pd.DataFrame(backend_utility)

    natural_frame = natural_summary(natural)
    cells = cell_summary(simulation)
    examples = qualitative_examples(
        results,
        simulation,
        maximum_multiplicity=maximum_m,
        count=int(cfg["analysis"]["top_examples"]),
    )

    output_dir = Path(args.output_dir or work_root / "reports" / args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    natural.to_csv(output_dir / "natural_state_metrics.csv", index=False)
    natural_frame.to_csv(output_dir / "natural_alias_summary.csv", index=False)
    simulation.to_csv(output_dir / "simulation_state_means.csv", index=False)
    cells.to_csv(output_dir / "cell_summary.csv", index=False)
    contrast_frame.to_csv(output_dir / "contrasts.csv", index=False)
    backend_utility_frame.to_csv(output_dir / "backend_utility.csv", index=False)
    examples.to_csv(output_dir / "qualitative_examples.csv", index=False)

    combined_natural = natural_frame[natural_frame["scope"] == "combined"].iloc[0]
    gate = cfg["gate"]
    surface_drop = contrast_lookup["surface_coverage_drop"]
    quotient_drop = contrast_lookup["quotient_coverage_drop"]
    quotient_coverage = contrast_lookup["quotient_coverage_gain_at_max"]
    quotient_utility = contrast_lookup["quotient_utility_gain_at_max"]
    decision = {
        "natural_aliasing_present": bool(
            float(combined_natural["state_alias_rate"])
            >= float(gate["minimum_natural_alias_state_rate"])
        ),
        "surface_coverage_is_alias_sensitive": bool(
            float(surface_drop.estimate) >= float(gate["minimum_surface_coverage_drop"])
            and float(surface_drop.ci_low) > 0.0
        ),
        "quotient_is_duplication_stable": bool(
            float(quotient_drop.estimate) <= float(gate["maximum_quotient_coverage_drop"])
        ),
        "quotient_recovers_coverage": bool(
            float(quotient_coverage.estimate)
            >= float(gate["minimum_quotient_coverage_gain"])
            and float(quotient_coverage.ci_low) > 0.0
        ),
        "quotient_recovers_utility": bool(
            float(quotient_utility.estimate)
            >= float(gate["minimum_quotient_union_gain"])
            and float(quotient_utility.ci_low) > 0.0
        ),
        "utility_nonnegative_in_both_backends": bool(
            (backend_utility_frame["estimate"] >= 0.0).all()
        ),
    }
    decision["all_conditions"] = all(decision.values())
    atomic_write_json(output_dir / "decision.json", decision)

    report = [
        "# EXP-015 Behavioral Alias Injection Pilot",
        "",
        f"Profile: `{args.profile}`. Frozen query states: {expected_states}. Eligible injection states: {simulation['state_id'].nunique()}.",
        "",
        "This is a pre-RL structural pilot. Frozen Qwen2.5-7B samples multiple next-query strings from the same unresolved state. Queries are executed against the real BM25 or E5 backend and quotient classes are defined by exact visible ranked document-ID transitions (with title fallback).",
        "",
        "## Natural behavioral aliasing",
        "",
        markdown_table(natural_frame),
        "",
        "`state_alias_rate` is the fraction of states where distinct strings induced the same exact visible transition. `within_class_entropy` is the surface entropy not explained by behavior classes.",
        "",
        "## Fixed-call-budget alias injection",
        "",
        markdown_table(cells),
        "",
        f"One non-best class is injected with multiplicity {multiplicities}. Every selector receives {int(cfg['simulation']['call_budget'])} calls. Surface samples strings, text-diverse maximizes lexical distance, and quotient balances behavior classes.",
        "",
        "## Primary contrasts",
        "",
        markdown_table(contrast_frame),
        "",
        "Positive coverage-drop values mean behavior coverage falls as alias multiplicity rises. Positive quotient-at-max contrasts favor class selection over string selection.",
        "",
        "## Utility by backend at maximum multiplicity",
        "",
        markdown_table(backend_utility_frame),
        "",
        "## Qualitative high-impact states",
        "",
        markdown_table(examples, digits=3),
        "",
        "## Decision",
        "",
        *[
            f"- {name.replace('_', ' ')}: **{'PASS' if value else 'FAIL'}**"
            for name, value in decision.items()
            if name != "all_conditions"
        ],
        "",
        (
            "**GO:** implement a learned outcome-equivalence predictor and BQ-GRPO micro-training experiment."
            if decision["all_conditions"]
            else "**NO-GO:** natural language aliases do not yet justify quotient-space policy optimization."
        ),
        "",
        "A GO is not a final RL result. The next stage must compare standard GRPO, text-diverse sampling, retrieval-result deduplication, and BQ-GRPO under the same environment-call budget.",
    ]
    (output_dir / "BEHAVIOR_ALIAS_PILOT_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()

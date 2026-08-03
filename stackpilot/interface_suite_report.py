from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stackpilot.interface_causality_common import atomic_write_json, load_config, markdown_table

EXPERIMENTS = ("EXP-020", "EXP-021", "EXP-022", "EXP-023")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the interface-causality project-selection suite.")
    parser.add_argument("--config", default="configs/interface_causality.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["work_dir"]).resolve() / "reports" / args.profile
    rows: list[dict[str, Any]] = []
    payloads = {}
    for experiment in EXPERIMENTS:
        path = root / experiment / "decision.json"
        if not path.is_file():
            raise RuntimeError(f"Missing {path}; run the corresponding experiment first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[experiment] = payload
        rows.append(
            {
                "experiment": experiment,
                "go": int(bool(payload.get("go"))),
                "profile": str(payload.get("profile", args.profile)),
                "states_or_pairs": float(
                    payload.get("states")
                    or payload.get("test_pairs")
                    or payload.get("paired_backend_states")
                    or 0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir or root / "suite").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "suite_decisions.csv", index=False)

    gradient_path = root / "EXP-020-gradient" / "decision.json"
    gradient_payload = (
        json.loads(gradient_path.read_text(encoding="utf-8"))
        if gradient_path.is_file()
        else None
    )
    alias_mechanism = bool(
        payloads["EXP-020"].get("go")
        and gradient_payload is not None
        and gradient_payload.get("go")
    )
    mechanistic = bool(alias_mechanism or payloads["EXP-021"].get("go"))
    open_interface_gap = bool(payloads["EXP-022"].get("go"))
    environment_conditioning = bool(payloads["EXP-023"].get("go"))
    proceed = bool(mechanistic and (open_interface_gap or environment_conditioning))
    decision = {
        "suite_id": str(cfg["suite_id"]),
        "profile": args.profile,
        "proceed_to_method": proceed,
        "mechanistic_support": mechanistic,
        "alias_simulation_support": bool(payloads["EXP-020"].get("go")),
        "alias_gradient_support": (
            bool(gradient_payload.get("go")) if gradient_payload is not None else None
        ),
        "finite_menu_expressivity_gap": open_interface_gap,
        "environment_conditioned_equivalence_support": environment_conditioning,
        "experiments": payloads,
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = [
        "# Interface-causality suite report",
        "",
        f"Profile: `{args.profile}`.",
        "",
        "## Experiment decisions",
        "",
        markdown_table(frame),
        "",
        "## Suite decision",
        "",
        "```text",
        json.dumps(
            {
                key: value
                for key, value in decision.items()
                if key != "experiments"
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        f"Overall: **{'PROCEED' if proceed else 'STOP OR REPOSITION'}**.",
        "",
        "A method paper is justified only when the audit identifies a real alias/credit mechanism and either finite menus leave a measurable expressivity gap or behavioral equivalence is demonstrably environment-conditioned. Otherwise the safest contribution is an analysis/boundary-condition paper rather than another credit-assignment algorithm.",
        "",
    ]
    (output_dir / "INTERFACE_CAUSALITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

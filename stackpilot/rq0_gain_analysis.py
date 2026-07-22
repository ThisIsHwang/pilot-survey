from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stackpilot.common import ensure_dir

SPECIALISTS = {"bm25-specialist": "bm25", "e5-specialist": "e5"}
METRICS = ("em", "f1", "support_recall")
START = "<!-- gain-over-base:start -->"
END = "<!-- gain-over-base:end -->"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [max(len(str(headers[i])), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="work/results/policies")
    parser.add_argument("--output-dir", default="work/results/rq0")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    frames = []
    for tag in ("base-qwen", *SPECIALISTS):
        path = results_dir / f"{tag}_summary.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing {path}")
        frame = pd.read_csv(path)
        frame["policy_tag"] = tag
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    blind = summary[summary["variant"] == "blind"].copy()

    gain_rows = []
    excess_rows = []
    for metric in METRICS:
        pivot = blind.pivot_table(index="policy_tag", columns="backend", values=metric, aggfunc="mean")
        for specialist, home in SPECIALISTS.items():
            away = "e5" if home == "bm25" else "bm25"
            home_gain = float(pivot.loc[specialist, home] - pivot.loc["base-qwen", home])
            away_gain = float(pivot.loc[specialist, away] - pivot.loc["base-qwen", away])
            for backend in ("bm25", "e5"):
                gain_rows.append(
                    {
                        "policy_tag": specialist,
                        "backend": backend,
                        "metric": metric,
                        "gain_over_base": float(
                            pivot.loc[specialist, backend] - pivot.loc["base-qwen", backend]
                        ),
                    }
                )
            excess_rows.append(
                {
                    "policy_tag": specialist,
                    "home_backend": home,
                    "metric": metric,
                    "home_gain": home_gain,
                    "away_gain": away_gain,
                    "home_backend_excess": home_gain - away_gain,
                }
            )

    gains = pd.DataFrame(gain_rows)
    excess = pd.DataFrame(excess_rows)
    gains.to_csv(output_dir / "gain_over_base.csv", index=False)
    excess.to_csv(output_dir / "home_backend_excess.csv", index=False)

    block = "\n".join(
        [
            START,
            "## Gain over base Qwen",
            "",
            "A gain that appears equally on BM25 and E5 is a general RL effect, not retriever specialization.",
            "",
            markdown_table(gains),
            "",
            "## Home-backend excess gain",
            "",
            "`home gain - away gain` is the Policy × Backend interaction after subtracting the base backend gap.",
            "",
            markdown_table(excess),
            END,
        ]
    )
    report_path = output_dir / "RQ0_REPORT.md"
    text = report_path.read_text(encoding="utf-8") if report_path.is_file() else "# RQ0 report\n"
    if START in text and END in text:
        prefix = text.split(START, 1)[0].rstrip()
        suffix = text.split(END, 1)[1].lstrip()
        text = f"{prefix}\n\n{block}\n\n{suffix}".rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    report_path.write_text(text, encoding="utf-8")
    print(markdown_table(gains))
    print(markdown_table(excess))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from stackpilot.multipositive_common import atomic_write_jsonl, load_config, read_jsonl, stable_hash


def generate(config_path: Path, profile: str, output: Path, limit: int | None) -> int:
    cfg = load_config(config_path)
    root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    states = [row for row in read_jsonl(root / "states.jsonl") if row.get("split") == "heldout"]
    states = sorted(states, key=lambda row: stable_hash("external", row["question_id"], row["state_id"]))
    if limit is not None:
        states = states[:limit]
    ext = cfg["external_generator"]
    api_base = os.environ.get("MULTIPOSITIVE_GENERATOR_API_BASE", str(ext["api_base"]))
    api_key = os.environ.get("MULTIPOSITIVE_GENERATOR_API_KEY", str(ext["api_key"]))
    model = os.environ.get("MULTIPOSITIVE_GENERATOR_MODEL", str(ext["model"]))
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=240, max_retries=3)
    rows = []
    for state in states:
        prompt = "\n".join([
            "Write one effective next web-search query for the unresolved information need.",
            "Do not imitate a named rewrite style, do not explain, and return only the query.",
            "Use the question and observed search history, but do not answer the question.",
            "",
            str(state["prompt"]),
        ])
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You independently formulate concise search queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(ext["temperature"]),
            max_tokens=int(ext["max_tokens"]),
            seed=int(stable_hash("external", state["state_id"], length=8), 16),
        )
        query = " ".join((response.choices[0].message.content or "").strip().strip('"').splitlines()[0].split())
        if len(query.split()) < 2:
            raise RuntimeError(f"External generator returned invalid query for {state['state_id']}: {query!r}")
        rows.append({
            "state_id": str(state["state_id"]),
            "question_id": str(state["question_id"]),
            "backend": str(state["backend"]),
            "query": query,
            "generator_model": model,
            "generator_api_base": api_base,
        })
    atomic_write_jsonl(output, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent held-out query targets.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output or Path(cfg["work_dir"]) / "external" / args.profile / "queries.jsonl").resolve()
    count = generate(Path(args.config).resolve(), args.profile, output, args.limit)
    print(json.dumps({"output": str(output), "queries": count}, indent=2))


if __name__ == "__main__":
    main()

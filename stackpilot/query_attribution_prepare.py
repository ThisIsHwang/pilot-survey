from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.query_attribution_common import SCHEMA, atomic_write_json, atomic_write_jsonl, file_sha256, load_config, read_jsonl, signature


def enrich_state(row: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(row.get("source_path", "")))
    if not source.is_file():
        raise RuntimeError(f"Original causal-audit state is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    state = payload["state"]
    prefix = payload.get("prefix", {})
    prefix_titles = [str(value) for record in prefix.get("records", []) for value in record.get("observed_titles", []) or []]
    return {**row, "support_titles": [str(value) for value in state.get("support_titles", [])], "answers": [str(value) for value in state.get("answers", [])], "prefix_observed_titles": prefix_titles, "prefix_support_recall": float(prefix.get("prefix_recall", 0.0)), "source_run_signature": str(payload.get("run_signature", row.get("source_run_signature", ""))), "source_state_signature": str(payload.get("state_signature", row.get("source_state_signature", "")))}


def prepare(config_path: Path, profile: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    source_root = Path(cfg["source_prepared_root"]).resolve() / profile
    audit_path = source_root / "state_audit.jsonl"
    source_manifest_path = source_root / "manifest.json"
    if not audit_path.is_file() or not source_manifest_path.is_file():
        raise RuntimeError(f"EXP-015 prepared states are missing under {source_root}; run query_equivalence/prepare.sh")
    rows = [enrich_state(row) for row in read_jsonl(audit_path)]
    run_signatures = {str(row.get("source_run_signature", "")) for row in rows}
    if len(run_signatures) != 1 or not next(iter(run_signatures)):
        raise RuntimeError(f"Mixed or empty causal-audit run signatures: {sorted(run_signatures)}")
    output_root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    output_root.mkdir(parents=True, exist_ok=True)
    states_path = output_root / "states.jsonl"
    atomic_write_jsonl(states_path, rows)
    manifest = {"schema": SCHEMA, "suite_id": cfg["suite_id"], "profile": profile, "config_path": str(config_path.resolve()), "config_sha256": file_sha256(config_path), "source_manifest_path": str(source_manifest_path.resolve()), "source_manifest_sha256": file_sha256(source_manifest_path), "source_run_signature": next(iter(run_signatures)), "states": len(rows), "states_path": str(states_path), "states_sha256": file_sha256(states_path), "backend_counts": {backend: sum(str(row["backend"]) == backend for row in rows) for backend in ("bm25", "e5")}}
    manifest["signature"] = signature(manifest)
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the attribution hypothesis matrix.")
    parser.add_argument("--config", default="configs/query_attribution.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.config).resolve(), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

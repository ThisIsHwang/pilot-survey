from __future__ import annotations

import argparse
import json
from pathlib import Path

from stackpilot.multipositive_common import (
    SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    load_config,
    read_jsonl,
    signature,
)


def prepare(config_path: Path, profile: str) -> dict:
    cfg = load_config(config_path)
    source_root = Path(cfg["source_prepared_root"]).resolve() / profile
    source_states = source_root / "states.jsonl"
    source_manifest = source_root / "manifest.json"
    if not source_states.is_file() or not source_manifest.is_file():
        raise RuntimeError(f"Missing query-attribution prepared states under {source_root}")
    rows = read_jsonl(source_states)
    output_root = Path(cfg["work_dir"]).resolve() / "prepared" / profile
    output_root.mkdir(parents=True, exist_ok=True)
    states_path = output_root / "states.jsonl"
    atomic_write_jsonl(states_path, rows)
    manifest = {
        "schema": SCHEMA,
        "suite_id": cfg["suite_id"],
        "profile": profile,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "states": len(rows),
        "states_path": str(states_path),
        "states_sha256": file_sha256(states_path),
    }
    manifest["signature"] = signature(manifest)
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-positive generalization states.")
    parser.add_argument("--config", default="configs/multipositive_generalization.yaml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.config).resolve(), args.profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

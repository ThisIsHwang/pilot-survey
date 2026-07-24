from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stackpilot.prepare_hard_rq0 import DATA_PREP_SCHEMA

RUN_COMPLETION_SCHEMA = 3
DATA_MANIFEST_RELATIVE_PATH = Path("data/.hard-rq0-data-manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Missing or invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        # Callers treat every malformed/stale artifact as one domain error.
        raise RuntimeError(  # noqa: TRY004
            f"{label} must contain a JSON object: {path}"
        )
    return payload


def specialist_key(backend: str, seed: int) -> str:
    return f"{backend}-seed{seed}"


def current_input_provenance(
    root: Path,
    profile: str,
    seeds: Iterable[int],
) -> dict[str, Any]:
    root = root.resolve()
    normalized_seeds = sorted({int(seed) for seed in seeds})
    data_manifest = root / DATA_MANIFEST_RELATIVE_PATH
    data_payload = read_json(data_manifest, "hard-RQ0 data manifest")
    if data_payload.get("schema") != DATA_PREP_SCHEMA:
        raise RuntimeError(
            f"Hard-RQ0 data manifest must use schema {DATA_PREP_SCHEMA}: "
            f"{data_manifest}"
        )

    specialists: dict[str, Any] = {}
    for seed in normalized_seeds:
        for backend in ("bm25", "e5"):
            key = specialist_key(backend, seed)
            relative_marker = (
                Path("checkpoints")
                / f"hard-rq0-{backend}-seed{seed}-{profile}"
                / ".complete.json"
            )
            marker = root / relative_marker
            payload = read_json(marker, f"{key} completion marker")
            expected = {
                "schema": 2,
                "backend": backend,
                "seed": seed,
                "profile": profile,
            }
            for field, value in expected.items():
                if payload.get(field) != value:
                    raise RuntimeError(
                        f"{key} completion marker {field}={payload.get(field)!r}, "
                        f"expected {value!r}: {marker}"
                    )
            training_signature = str(payload.get("training_signature") or "").strip()
            if not training_signature:
                raise RuntimeError(
                    f"{key} completion marker has no training signature: {marker}"
                )
            specialists[key] = {
                "marker": relative_marker.as_posix(),
                "marker_sha256": file_sha256(marker),
                "training_signature": training_signature,
            }

    return {
        "data_manifest": {
            "path": DATA_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": file_sha256(data_manifest),
        },
        "specialists": specialists,
    }


def validate_run_completion(
    root: Path,
    profile: str,
    result_set: str,
    seeds: Iterable[int],
    *,
    marker_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    required_seeds = sorted({int(seed) for seed in seeds})
    marker = (
        marker_path.resolve()
        if marker_path is not None
        else root / "runs" / result_set / ".complete.json"
    )
    payload = read_json(marker, "EXP-002 run completion marker")
    if (
        payload.get("schema") != RUN_COMPLETION_SCHEMA
        or payload.get("profile") != profile
        or payload.get("result_set") != result_set
    ):
        raise RuntimeError(
            "EXP-002 run marker does not match the required schema/profile/result "
            f"set ({RUN_COMPLETION_SCHEMA}, {profile!r}, {result_set!r}): {marker}"
        )
    raw_available_seeds = payload.get("seeds")
    if not isinstance(raw_available_seeds, list):
        raise RuntimeError(  # noqa: TRY004
            f"EXP-002 run marker has no valid seed list: {marker}"
        )
    try:
        available_seeds = {int(value) for value in raw_available_seeds}
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"EXP-002 run marker contains an invalid seed list: {marker}"
        ) from exc
    missing_seeds = sorted(set(required_seeds) - available_seeds)
    if missing_seeds:
        raise RuntimeError(
            f"EXP-002 run marker is missing specialist seeds {missing_seeds}: {marker}"
        )

    current = current_input_provenance(root, profile, required_seeds)
    recorded = payload.get("input_provenance")
    if not isinstance(recorded, dict):
        raise RuntimeError(  # noqa: TRY004
            f"EXP-002 run marker has no input provenance: {marker}"
        )
    if recorded.get("data_manifest") != current["data_manifest"]:
        raise RuntimeError(
            f"EXP-002 run marker is stale for the current hard-RQ0 data: {marker}"
        )
    recorded_specialists = recorded.get("specialists")
    if not isinstance(recorded_specialists, dict):
        raise RuntimeError(  # noqa: TRY004
            f"EXP-002 run marker has no specialist provenance: {marker}"
        )
    for key, expected in current["specialists"].items():
        if recorded_specialists.get(key) != expected:
            raise RuntimeError(
                f"EXP-002 run marker is stale for specialist {key}: {marker}"
            )
    return payload

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

EXPERIMENT_ID_RE = re.compile(r"^EXP-[0-9]{3}$")
RUN_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_STATUSES = {"planned", "active", "completed", "rejected"}


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments" / "registry.json"


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else default_registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(payload)
    return payload


def validate_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema") != 1:
        raise ValueError(f"Unsupported experiment registry schema: {payload.get('schema')!r}")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Experiment registry must contain a non-empty experiments list")

    ids: list[str] = []
    slugs: list[str] = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment entry {index} is not an object")
        experiment_id = str(experiment.get("id", ""))
        slug = str(experiment.get("slug", ""))
        status = str(experiment.get("status", ""))
        entrypoint = str(experiment.get("entrypoint", ""))
        if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
            raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
        if not slug or RUN_TOKEN_RE.search(slug):
            raise ValueError(f"Invalid experiment slug for {experiment_id}: {slug!r}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status for {experiment_id}: {status!r}")
        if not entrypoint:
            raise ValueError(f"Missing entrypoint for {experiment_id}")
        ids.append(experiment_id)
        slugs.append(slug)

    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique")
    if len(slugs) != len(set(slugs)):
        raise ValueError("Experiment slugs must be unique")
    numeric_ids = [int(experiment_id.split("-")[1]) for experiment_id in ids]
    if numeric_ids != sorted(numeric_ids):
        raise ValueError("Experiment IDs must appear in ascending numeric order")

    known_ids = set(ids)
    for experiment in experiments:
        parent = experiment.get("parent")
        if parent is not None and parent not in known_ids:
            raise ValueError(f"Unknown parent {parent!r} for {experiment['id']}")


def experiment_by_id(payload: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for experiment in payload["experiments"]:
        if experiment["id"] == experiment_id:
            return experiment
    raise KeyError(f"Unknown experiment ID: {experiment_id}")


def sanitize_token(value: str) -> str:
    token = RUN_TOKEN_RE.sub("-", str(value).strip()).strip("-")
    if not token:
        raise ValueError(f"Run token is empty after sanitization: {value!r}")
    return token


def make_run_id(
    experiment_id: str,
    *,
    seed: int | None = None,
    profile: str | None = None,
    variant: str | None = None,
) -> str:
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    parts = [experiment_id]
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        parts.append(f"seed-{seed:03d}")
    if profile is not None:
        parts.append(f"profile-{sanitize_token(profile)}")
    if variant is not None:
        parts.append(f"variant-{sanitize_token(variant)}")
    return "__".join(parts)


def experiment_root(work_root: str | Path, experiment_id: str) -> Path:
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    return Path(work_root) / "experiments" / experiment_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(default_registry_path()))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("list")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("experiment_id")
    run_parser = subparsers.add_parser("run-id")
    run_parser.add_argument("experiment_id")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--profile")
    run_parser.add_argument("--variant")
    args = parser.parse_args()

    payload = load_registry(args.registry)
    if args.command == "validate":
        print(f"valid: {len(payload['experiments'])} experiments")
    elif args.command == "list":
        for experiment in payload["experiments"]:
            print(
                f"{experiment['id']}\t{experiment['status']}\t"
                f"{experiment['slug']}\t{experiment['entrypoint']}"
            )
    elif args.command == "show":
        print(json.dumps(experiment_by_id(payload, args.experiment_id), indent=2))
    elif args.command == "run-id":
        experiment_by_id(payload, args.experiment_id)
        print(
            make_run_id(
                args.experiment_id,
                seed=args.seed,
                profile=args.profile,
                variant=args.variant,
            )
        )


if __name__ == "__main__":
    main()

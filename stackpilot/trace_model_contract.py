from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.trace_common import (
    atomic_write_json,
    canonical_signature,
    file_sha256,
    load_trace_config,
    read_jsonl,
)

MODEL_CONTRACT_SCHEMA = 1


def planned_model(jobs: list[dict[str, Any]]) -> tuple[str, bool]:
    if not jobs:
        raise RuntimeError("TRACE model contract received an empty job plan")
    model_refs: set[str] = set()
    trust_values: set[bool] = set()
    for index, job in enumerate(jobs):
        model_ref = str(job.get("base_model", "")).strip()
        if not model_ref:
            raise RuntimeError(f"TRACE job {index} has no base_model")
        model_refs.add(model_ref)
        trust_values.add(bool(job.get("trust_remote_code", False)))
    if len(model_refs) != 1:
        raise RuntimeError(
            "TRACE jobs must use one shared base model; found "
            + ", ".join(sorted(model_refs))
        )
    if len(trust_values) != 1:
        raise RuntimeError("TRACE jobs disagree on trust_remote_code")
    return next(iter(model_refs)), next(iter(trust_values))


def validate_parameter_count(
    parameter_count: int,
    *,
    minimum_parameters: int,
    maximum_parameters: int,
) -> None:
    if minimum_parameters <= 0:
        raise ValueError("minimum_parameters must be positive")
    if maximum_parameters < minimum_parameters:
        raise ValueError("maximum_parameters must be >= minimum_parameters")
    if not minimum_parameters <= parameter_count <= maximum_parameters:
        raise RuntimeError(
            "TRACE requires a 7B-class checkpoint with "
            f"{minimum_parameters:,}..{maximum_parameters:,} parameters; "
            f"the planned model has {parameter_count:,}"
        )


def count_parameters_without_weights(
    model_ref: str,
    *,
    trust_remote_code: bool,
) -> tuple[int, str, str]:
    # Build only meta tensors. This validates the architecture and exact parameter
    # count without allocating the 7B checkpoint on CPU or CUDA.
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return parameter_count, type(model).__name__, str(getattr(config, "model_type", ""))


def load_plan_contract(
    jobs_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, dict[str, Any]]:
    jobs = read_jsonl(jobs_path)
    manifest_path = jobs_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing TRACE plan manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("jobs_sha256") != file_sha256(jobs_path):
        raise RuntimeError(
            "TRACE jobs.jsonl does not match its plan manifest; rerun trace_go/plan.sh"
        )

    config_path = Path(str(manifest.get("config_path", ""))).expanduser()
    if not config_path.is_file():
        raise RuntimeError(f"TRACE plan config is missing: {config_path}")
    if manifest.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError(
            "TRACE config changed after planning; rerun trace_go/plan.sh before launching jobs"
        )
    config = load_trace_config(config_path)
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise RuntimeError("TRACE config is missing the model mapping")
    return jobs, manifest, config_path.resolve(), model_config


def verify_model_contract(jobs_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    jobs_path = jobs_path.resolve()
    jobs, manifest, config_path, model_config = load_plan_contract(jobs_path)
    model_ref, trust_remote_code = planned_model(jobs)
    minimum_parameters = int(model_config.get("minimum_parameters", 0))
    maximum_parameters = int(model_config.get("maximum_parameters", 0))
    parameter_count, architecture, model_type = count_parameters_without_weights(
        model_ref,
        trust_remote_code=trust_remote_code,
    )
    validate_parameter_count(
        parameter_count,
        minimum_parameters=minimum_parameters,
        maximum_parameters=maximum_parameters,
    )
    payload = {
        "schema": MODEL_CONTRACT_SCHEMA,
        "status": "valid",
        "profile": manifest.get("profile"),
        "jobs_path": str(jobs_path),
        "jobs_sha256": file_sha256(jobs_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "base_model": model_ref,
        "trust_remote_code": trust_remote_code,
        "architecture": architecture,
        "model_type": model_type,
        "parameter_count": parameter_count,
        "minimum_parameters": minimum_parameters,
        "maximum_parameters": maximum_parameters,
    }
    payload["signature"] = canonical_signature(payload)
    destination = output_path or jobs_path.parent / "model_contract.json"
    atomic_write_json(destination, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a TRACE job plan uses one 7B-class base checkpoint."
    )
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = verify_model_contract(
        Path(args.jobs),
        Path(args.output).resolve() if args.output else None,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

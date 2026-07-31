from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stackpilot.causal_query_common import (
    canonical_signature,
    load_causal_query_config,
)
from stackpilot.trace_common import atomic_write_json

MODEL_CONTRACT_SCHEMA = 1


def validate_parameter_count(
    parameter_count: int,
    *,
    minimum_parameters: int,
    maximum_parameters: int,
) -> None:
    if minimum_parameters <= 0 or maximum_parameters < minimum_parameters:
        raise ValueError("Invalid causal-query parameter-count bounds")
    if not minimum_parameters <= parameter_count <= maximum_parameters:
        raise RuntimeError(
            "EXP-013 requires a 7B-class checkpoint with "
            f"{minimum_parameters:,}..{maximum_parameters:,} parameters; "
            f"the selected model has {parameter_count:,}"
        )


def count_parameters_without_weights(
    model_ref: str,
    *,
    trust_remote_code: bool,
) -> tuple[int, str, str]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=Path(model_ref).is_dir(),
    )
    # PyTorch's meta-device context constructs the complete architecture without
    # loading or allocating checkpoint tensors on CPU/CUDA.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return parameter_count, type(model).__name__, str(getattr(config, "model_type", ""))


def verify_model_contract(
    *,
    config_path: Path,
    model_ref: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    cfg = load_causal_query_config(config_path)
    model_cfg = cfg["model"]
    selected = str(model_ref or model_cfg["base_model"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    parameter_count, architecture, model_type = count_parameters_without_weights(
        selected,
        trust_remote_code=trust_remote_code,
    )
    minimum = int(model_cfg["minimum_parameters"])
    maximum = int(model_cfg["maximum_parameters"])
    validate_parameter_count(
        parameter_count,
        minimum_parameters=minimum,
        maximum_parameters=maximum,
    )
    payload = {
        "schema": MODEL_CONTRACT_SCHEMA,
        "status": "valid",
        "config_path": str(config_path.resolve()),
        "base_model": selected,
        "trust_remote_code": trust_remote_code,
        "architecture": architecture,
        "model_type": model_type,
        "parameter_count": parameter_count,
        "minimum_parameters": minimum,
        "maximum_parameters": maximum,
    }
    payload["signature"] = canonical_signature(payload)
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Qwen2.5-7B model contract for EXP-013."
    )
    parser.add_argument("--config", default="configs/causal_query_audit.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = verify_model_contract(
        config_path=Path(args.config),
        model_ref=args.model,
        output_path=Path(args.output).resolve() if args.output else None,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

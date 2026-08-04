from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import stackpilot.query_attribution_lora_job as base

_ORIGINAL_OBJECTIVE = base.objective_loss


def objective_loss(losses: Any, batch: dict[str, Any], objective: dict[str, Any]) -> Any:
    import torch

    kind = str(objective["kind"])
    if kind not in {"set_mass", "set_mass_plus_variance"}:
        return _ORIGINAL_OBJECTIVE(losses, batch, objective)
    indices = batch["group_indices"].cuda(non_blocking=True)
    weights = batch["weights"].cuda(non_blocking=True)
    temperature = float(objective.get("temperature", 1.0))
    coefficient = float(objective.get("coefficient", 0.0))
    if temperature <= 0.0:
        raise ValueError("set-mass temperature must be positive")
    group_losses = []
    for group_index in range(len(batch["group_ids"])):
        mask = indices.eq(group_index)
        values = losses[mask]
        local_weights = weights[mask]
        local_weights = local_weights / local_weights.sum().clamp_min(1e-8)
        set_mass = -temperature * torch.logsumexp(
            -values / temperature + torch.log(local_weights.clamp_min(1e-8)), dim=0
        )
        if kind == "set_mass_plus_variance":
            mean = (local_weights * values).sum()
            variance = (local_weights * (values - mean).pow(2)).sum()
            set_mass = set_mass + coefficient * variance
        group_losses.append(set_mass)
    return torch.stack(group_losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one multi-positive LoRA job.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base.objective_loss = objective_loss
    base.run(Path(args.job).resolve(), force=args.force)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from hard_rq0._patch_searchr1_behavior_quotient_impl import *  # noqa: F401,F403
from hard_rq0._patch_searchr1_behavior_quotient_impl import (
    patch_generation,
    patch_trainer,
)


def patch(search_r1_root: Path) -> None:
    """Apply behavior telemetry, feedback rollout, and optional query credit."""

    patch_generation(search_r1_root)
    patch_trainer(search_r1_root)
    from hard_rq0.patch_searchr1_response_feedback import patch as patch_feedback

    patch_feedback(search_r1_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())

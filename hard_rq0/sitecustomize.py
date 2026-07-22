"""Seed Python, NumPy, and PyTorch in Search-R1 driver and Ray workers."""
from __future__ import annotations

import os
import random

seed_text = os.environ.get("RQ0_SEED")
if seed_text is not None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    seed = int(seed_text) + rank
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

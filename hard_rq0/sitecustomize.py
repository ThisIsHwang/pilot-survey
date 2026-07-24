"""Seed CPU RNGs without initializing CUDA before Ray assigns worker GPUs."""
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
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        # torch.manual_seed() also queues a CUDA seed.  Keep this startup hook
        # strictly CPU-only: Ray narrows CUDA_VISIBLE_DEVICES only after the
        # worker interpreter (and therefore sitecustomize) has started.
        torch.default_generator.manual_seed(seed)

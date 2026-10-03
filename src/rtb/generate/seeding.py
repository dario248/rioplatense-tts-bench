"""Seeding of every random source available in the current environment.

Imports numpy and torch lazily, because the evaluation environment has neither. Returns which
sources were actually seeded: "seeded" with torch missing is not the same experiment.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = False) -> list[str]:
    """Seed python, numpy, torch and cuda when present. Returns the sources seeded."""
    seeded = ["python"]
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded.append("numpy")

    try:
        import torch
    except ImportError:
        return seeded

    torch.manual_seed(seed)
    seeded.append("torch")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeded.append("cuda")
    if deterministic:
        # Puede degradar la latencia; no se usa en las corridas que reportan timings.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seeded.append("cudnn_deterministic")
    return seeded

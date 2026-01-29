from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set RNG seeds across python/numpy/torch.

    Note: full determinism can reduce performance and is not always possible
    with some CUDA ops. For CPU-only runs it's usually fine.

    Parameters
    ----------
    seed:
        Random seed.
    deterministic:
        If True, forces deterministic algorithms where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_all(seed: int, deterministic: bool = False) -> None:
    """Backward-compatible alias for :func:`set_global_seed`.

    Some older training scripts expect a function named `seed_all`.
    This simply forwards to `set_global_seed`.
    """
    set_global_seed(seed=seed, deterministic=deterministic)

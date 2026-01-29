from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Split:
    train_folders: List[str]
    val_folders: List[str]
    test_folders: List[str]


def split_folders(
    folders: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 123,
) -> Split:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    folders = sorted(set(folders))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(folders))
    n = len(folders)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train = [folders[i] for i in train_idx]
    val = [folders[i] for i in val_idx]
    test = [folders[i] for i in test_idx]

    return Split(train, val, test)

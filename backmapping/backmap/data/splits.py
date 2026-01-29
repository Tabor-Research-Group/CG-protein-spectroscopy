from __future__ import annotations

"""Train/val/test splits.

The *recommended* split for this project is by **protein folder** (i.e., leave
whole proteins out of training). This prevents leakage across frames of the same
protein.

However, tiny debug pickles sometimes contain only 1 folder. In that case, a
folder split would yield empty val/test sets. To keep pipelines debuggable, we
provide a deterministic fallback to a random split by oscillator index.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Split:
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]

    train_folders: List[str] | None = None
    val_folders: List[str] | None = None
    test_folders: List[str] | None = None


def _check_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    s = float(train_frac) + float(val_frac) + float(test_frac)
    if not np.isclose(s, 1.0):
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0; got {s}")


def split_by_folder(
    folders: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 123,
) -> Tuple[List[str], List[str], List[str]]:
    """Split unique folder names."""
    _check_fracs(train_frac, val_frac, test_frac)

    uniq = sorted(set([str(f) for f in folders]))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(uniq))

    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_folders = [uniq[i] for i in train_idx]
    val_folders = [uniq[i] for i in val_idx]
    test_folders = [uniq[i] for i in test_idx]

    return train_folders, val_folders, test_folders


def split_indices(
    *,
    folders_by_index: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 123,
    split_by: str = "folder",
    min_items_per_split: int = 1,
) -> Split:
    """Return train/val/test index lists.

    Parameters
    ----------
    folders_by_index:
        List of folder names for each oscillator index.
    split_by:
        "folder" (recommended) or "random".
    min_items_per_split:
        If a split would contain < min_items_per_split items, we fallback to a
        random split.
    """
    _check_fracs(train_frac, val_frac, test_frac)

    n = len(folders_by_index)
    all_idx = np.arange(n)

    if split_by not in {"folder", "random"}:
        raise ValueError(f"split_by must be 'folder' or 'random'; got {split_by}")

    rng = np.random.default_rng(int(seed))

    if split_by == "folder":
        train_folders, val_folders, test_folders = split_by_folder(
            folders_by_index, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        train_set = set(train_folders)
        val_set = set(val_folders)
        test_set = set(test_folders)

        train_idx = [i for i, f in enumerate(folders_by_index) if f in train_set]
        val_idx = [i for i, f in enumerate(folders_by_index) if f in val_set]
        test_idx = [i for i, f in enumerate(folders_by_index) if f in test_set]

        # If the dataset is tiny (e.g., only 1 folder), fallback so the pipeline
        # still has something to validate/test.
        if (
            len(train_idx) < min_items_per_split
            or len(val_idx) < min_items_per_split
            or len(test_idx) < min_items_per_split
        ):
            split_by = "random"  # fallback
        else:
            return Split(
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                train_folders=train_folders,
                val_folders=val_folders,
                test_folders=test_folders,
            )

    # Random split fallback
    perm = rng.permutation(n)
    n_train = max(min_items_per_split, int(round(train_frac * n)))
    n_val = max(min_items_per_split, int(round(val_frac * n)))
    n_test = n - n_train - n_val
    if n_test < min_items_per_split:
        # borrow from train
        deficit = min_items_per_split - n_test
        n_train = max(min_items_per_split, n_train - deficit)
        n_test = min_items_per_split

    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train : n_train + n_val].tolist()
    test_idx = perm[n_train + n_val : n_train + n_val + n_test].tolist()

    return Split(train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)

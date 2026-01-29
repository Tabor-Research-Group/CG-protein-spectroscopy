from __future__ import annotations

"""I/O utilities.

The provided datasets are Python pickles that may contain NumPy arrays.

Important:
  * Pickles created under NumPy 2.x may reference the module path
    `numpy._core`, which does not exist in NumPy 1.x.
  * If you unpickle such a file under NumPy 1.x, you will see:
        ModuleNotFoundError: No module named 'numpy._core'

This module provides a safe loader that patches module aliases so the same
pickle can be loaded on NumPy 1.x and 2.x.
"""

from typing import Any

import pickle
import sys
import types

import numpy as np


def load_pickle_numpy_compat(path: str) -> Any:
    """Load a pickle with cross-version NumPy compatibility.

    Parameters
    ----------
    path:
        Path to a .pkl file.

    Returns
    -------
    Any
        Unpickled object.
    """
    # If the pickle was produced under NumPy 2.x, it may contain references
    # to 'numpy._core'. Under NumPy 1.x, that module does not exist.
    # We create a lightweight alias pointing at np.core so the unpickler can
    # resolve the references.
    if "numpy._core" not in sys.modules:
        mod = types.ModuleType("numpy._core")
        mod.__dict__.update(np.core.__dict__)
        sys.modules["numpy._core"] = mod
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        if hasattr(np.core, "_multiarray_umath"):
            sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

    with open(path, "rb") as f:
        return pickle.load(f)

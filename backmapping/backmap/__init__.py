"""Diffusion-based CG→atomistic backmapping for an oscillator dataset.

This package intentionally keeps dependencies minimal:
- PyTorch for learning
- NumPy for data handling
- matplotlib for plotting
- PyYAML for configs

Entry points
-----------
See :mod:`scripts.train` and :mod:`scripts.infer` for production-ready CLI tools.
"""

__all__ = [
    "config",
    "data",
    "geometry",
    "model",
    "physics",
    "train",
    "utils",
]

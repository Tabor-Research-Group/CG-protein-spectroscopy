"""Training utilities (losses, evaluation, plotting).

The model architecture lives in :mod:`backmap.model`.
This package contains everything around training:
- loss computation and metric extraction
- evaluation loops (train/val/test)
- per-epoch plotting
- visualization helpers (writing PDB overlays)
"""

from .losses import LossBreakdown, compute_losses
from .metrics import BatchMetrics, compute_batch_metrics
from .plotting import plot_epoch_metrics

__all__ = [
    "LossBreakdown",
    "compute_losses",
    "BatchMetrics",
    "compute_batch_metrics",
    "plot_epoch_metrics",
]

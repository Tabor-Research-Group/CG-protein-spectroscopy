from __future__ import annotations

import torch


def rbf_expand(
    distances: torch.Tensor,
    num_centers: int = 16,
    rbf_max_dist: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gaussian radial basis expansion.

    Parameters
    ----------
    distances:
        Tensor [...], distances in Å.
    num_centers:
        Number of RBF centers.
    rbf_max_dist:
        Maximum distance for centers (centers in [0, rbf_max_dist]).
    eps:
        Numerical epsilon.

    Returns
    -------
    Tensor [..., num_centers]
    """
    # centers linearly spaced from 0..rbf_max_dist
    centers = torch.linspace(0.0, float(rbf_max_dist), int(num_centers), device=distances.device, dtype=distances.dtype)
    # width so that adjacent centers overlap reasonably
    # Use sigma = (max_dist / num_centers)
    sigma = float(rbf_max_dist) / float(num_centers)
    gamma = 1.0 / (2.0 * (sigma ** 2) + eps)

    d = distances.unsqueeze(-1)  # [..., 1]
    return torch.exp(-gamma * (d - centers) ** 2)

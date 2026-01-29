from __future__ import annotations

import torch


def dihedral_angle(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute dihedral angle for quadruplets, matching MDAnalysis convention.

    Uses the standard arctan2 formulation:

      b1 = p2 - p1
      b2 = p3 - p2
      b3 = p4 - p3
      n1 = b1 x b2
      n2 = b2 x b3
      m1 = n1 x (b2/||b2||)
      angle = atan2( dot(m1, n2), dot(n1, n2) )

    Returns angles in radians in [-pi, pi].

    All operations are differentiable in PyTorch.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    b2_norm = torch.sqrt(torch.clamp((b2 * b2).sum(dim=-1, keepdim=True), min=eps))
    b2_u = b2 / b2_norm

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    n1_norm = torch.sqrt(torch.clamp((n1 * n1).sum(dim=-1, keepdim=True), min=eps))
    n2_norm = torch.sqrt(torch.clamp((n2 * n2).sum(dim=-1, keepdim=True), min=eps))

    n1_u = n1 / n1_norm
    n2_u = n2 / n2_norm

    m1 = torch.cross(n1_u, b2_u, dim=-1)

    x = (n1_u * n2_u).sum(dim=-1)
    y = (m1 * n2_u).sum(dim=-1)

    return torch.atan2(y, x)


def angle_to_sincos(angle_rad: torch.Tensor) -> torch.Tensor:
    """Return [cos(angle), sin(angle)] for a tensor of angles."""
    return torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)], dim=-1)

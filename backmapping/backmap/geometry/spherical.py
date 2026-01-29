from __future__ import annotations

from typing import Tuple

import torch


def cartesian_to_spherical_sincos(
    v: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert cartesian vectors to (r, cosθ, sinθ, cosφ, sinφ).

    Convention (in local coordinates):
      - r = ||v||
      - θ: polar angle from +z axis
      - φ: azimuth angle in x-y plane from +x axis

    Returns
    -------
    sph: [...,5] = [r, cosθ, sinθ, cosφ, sinφ]
    """
    if v.shape[-1] != 3:
        raise ValueError(f"v must have last dim 3, got {v.shape}")
    x, y, z = v.unbind(dim=-1)
    r = torch.sqrt(torch.clamp(x * x + y * y + z * z, min=eps))
    # theta
    cos_theta = torch.clamp(z / r, min=-1.0, max=1.0)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
    # phi
    xy = torch.sqrt(torch.clamp(x * x + y * y, min=eps))
    cos_phi = x / xy
    sin_phi = y / xy
    # handle the x=y=0 case by setting phi=0 => cos=1, sin=0
    is_pole = (x * x + y * y) < 1e-12
    cos_phi = torch.where(is_pole, torch.ones_like(cos_phi), cos_phi)
    sin_phi = torch.where(is_pole, torch.zeros_like(sin_phi), sin_phi)
    return torch.stack([r, cos_theta, sin_theta, cos_phi, sin_phi], dim=-1)


def spherical_sincos_to_cartesian(
    sph: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert (r, cosθ, sinθ, cosφ, sinφ) to cartesian vector."""
    if sph.shape[-1] != 5:
        raise ValueError(f"sph must have last dim 5, got {sph.shape}")
    r, cos_theta, sin_theta, cos_phi, sin_phi = sph.unbind(dim=-1)

    # Normalize trig pairs for numerical stability
    ct_st = torch.stack([cos_theta, sin_theta.abs()], dim=-1)
    cp_sp = torch.stack([cos_phi, sin_phi], dim=-1)

    ct_st = ct_st / torch.sqrt(torch.clamp((ct_st * ct_st).sum(dim=-1, keepdim=True), min=eps))
    cp_sp = cp_sp / torch.sqrt(torch.clamp((cp_sp * cp_sp).sum(dim=-1, keepdim=True), min=eps))

    cos_theta_n = ct_st[..., 0]
    sin_theta_n = ct_st[..., 1]
    cos_phi_n = cp_sp[..., 0]
    sin_phi_n = cp_sp[..., 1]

    x = r * sin_theta_n * cos_phi_n
    y = r * sin_theta_n * sin_phi_n
    z = r * cos_theta_n
    return torch.stack([x, y, z], dim=-1)

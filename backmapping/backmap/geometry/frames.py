from __future__ import annotations

"""Local residue frames and SE(3) transforms.

This module defines:

* compute_residue_local_frames: build a right-handed orthonormal frame per residue
* global_to_local / local_to_global: coordinate transforms using those frames
* clamp_norm: stable norm clamping for diffusion stability

Conventions
-----------
Frames are stored as rotation matrices R with columns [e1, e2, e3].
Given origin O (typically BB bead):

  local = R^T (global - O)
  global = O + R local

All operations are differentiable in PyTorch.
"""

from typing import Optional

import torch


def _safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ||v|| with shape [...,1] and an eps floor."""
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=eps))


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / _safe_norm(v, eps=eps)


def _pick_arbitrary_perp(v: torch.Tensor) -> torch.Tensor:
    """Pick an axis that is not too aligned with v (batch-safe)."""
    abs_v = v.abs()
    e1 = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    e3 = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=v.dtype).expand_as(v)
    idx = abs_v.argmin(dim=-1)  # [N]
    return torch.where(idx[:, None] == 0, e1, torch.where(idx[:, None] == 1, e2, e3))


def compute_residue_local_frames(
    bb_pos: torch.Tensor,
    sc1_pos: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a right-handed orthonormal frame per residue.

    Parameters
    ----------
    bb_pos:
        [N,3] backbone bead positions.
    sc1_pos:
        Optional [N,3] sidechain bead positions used to stabilize the e2 axis.
        Use NaN rows for residues without SC1.
    eps:
        Numerical epsilon.

    Returns
    -------
    R:
        [N,3,3] rotation matrices with columns [e1,e2,e3].
    """
    if bb_pos.ndim != 2 or bb_pos.shape[-1] != 3:
        raise ValueError(f"bb_pos must be [N,3], got {tuple(bb_pos.shape)}")
    N = int(bb_pos.shape[0])
    if N < 1:
        raise ValueError("bb_pos must have at least one residue")

    # N==1: by default, return identity.
    #
    # However, for *sidechain* oscillators we often have only one residue in the
    # local frame. In that case, an identity frame ties our spherical coordinates
    # to the *global* axes, which destroys rotational invariance and makes the
    # learning problem harder than necessary.
    #
    # If `sc1_pos` is provided (a reference point for this residue), we use the
    # direction (sc1_pos - bb_pos) to define a deterministic, right-handed frame:
    #   e1 = unit(sc1_pos - bb_pos)
    #   e2 = arbitrary perpendicular to e1 (deterministic)
    #   e3 = e1 x e2
    # This yields stable local spherical coordinates even for single-residue graphs.
    if N == 1:
        if sc1_pos is not None:
            if sc1_pos.shape != bb_pos.shape:
                raise ValueError(f"sc1_pos must have same shape as bb_pos, got {tuple(sc1_pos.shape)}")

            v = sc1_pos[0] - bb_pos[0]
            finite = bool(torch.isfinite(v).all().item())
            if finite and float(v.abs().sum().item()) > 0.0:
                e1 = _safe_normalize(v.view(1, 3), eps=eps)  # [1,3]
                # Deterministic perpendicular seed and Gram-Schmidt.
                e2_seed = _pick_arbitrary_perp(e1)
                proj = (e2_seed * e1).sum(dim=-1, keepdim=True) * e1
                e2 = e2_seed - proj
                e2 = _safe_normalize(e2, eps=eps)
                e3 = torch.cross(e1, e2, dim=-1)
                e3 = _safe_normalize(e3, eps=eps)
                # Re-orthogonalize e2 to reduce drift.
                e2 = torch.cross(e3, e1, dim=-1)
                e2 = _safe_normalize(e2, eps=eps)
                return torch.stack([e1, e2, e3], dim=-1)  # [1,3,3]

        # Fallback: identity
        return torch.eye(3, device=bb_pos.device, dtype=bb_pos.dtype).unsqueeze(0)  # [1,3,3]

    # Forward direction along chain (to next; last uses previous segment)
    fwd = torch.zeros_like(bb_pos)
    fwd[:-1] = bb_pos[1:] - bb_pos[:-1]
    fwd[-1] = bb_pos[-1] - bb_pos[-2]
    e1 = _safe_normalize(fwd, eps=eps)

    # Backward direction (towards previous; first uses next segment)
    bwd = torch.zeros_like(bb_pos)
    bwd[1:] = bb_pos[:-1] - bb_pos[1:]
    bwd[0] = bb_pos[0] - bb_pos[1]

    # Choose a second direction seed
    if sc1_pos is not None:
        if sc1_pos.shape != bb_pos.shape:
            raise ValueError(f"sc1_pos must have same shape as bb_pos, got {tuple(sc1_pos.shape)}")
        sc_dir = sc1_pos - bb_pos
        finite = torch.isfinite(sc_dir).all(dim=-1) & (sc_dir.abs().sum(dim=-1) > 0)
        e2_seed = torch.where(finite[:, None], sc_dir, bwd)
    else:
        e2_seed = bwd

    # Gram-Schmidt: make e2 perpendicular to e1
    proj = (e2_seed * e1).sum(dim=-1, keepdim=True) * e1
    e2 = e2_seed - proj

    # If e2 becomes too small (colinear), pick an arbitrary perpendicular
    small = (_safe_norm(e2, eps=eps).squeeze(-1) < 1e-3)
    if small.any():
        arb = _pick_arbitrary_perp(e1)
        proj2 = (arb * e1).sum(dim=-1, keepdim=True) * e1
        e2_alt = arb - proj2
        e2 = torch.where(small[:, None], e2_alt, e2)
    e2 = _safe_normalize(e2, eps=eps)

    e3 = torch.cross(e1, e2, dim=-1)
    e3 = _safe_normalize(e3, eps=eps)

    # Re-orthogonalize e2 to reduce numeric drift
    e2 = torch.cross(e3, e1, dim=-1)
    e2 = _safe_normalize(e2, eps=eps)

    return torch.stack([e1, e2, e3], dim=-1)  # [N,3,3]


def global_to_local(x_global: torch.Tensor, origin_global: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform global coordinates to local.

    x_local = R^T (x_global - origin)
    """
    v = x_global - origin_global
    return torch.matmul(R.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)


def local_to_global(x_local: torch.Tensor, origin_global: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform local coordinates to global."""
    v = torch.matmul(R, x_local.unsqueeze(-1)).squeeze(-1)
    return origin_global + v


def clamp_norm(v: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    """Clamp vector norms by rescaling vectors that exceed max_norm."""
    n = _safe_norm(v, eps=eps)
    scale = torch.clamp(float(max_norm) / n, max=1.0)
    return v * scale

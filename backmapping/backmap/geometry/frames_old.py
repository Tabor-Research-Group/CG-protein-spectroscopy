from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=eps))


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / _safe_norm(v, eps=eps)


def _pick_arbitrary_perp(v: torch.Tensor) -> torch.Tensor:
    """Pick a vector that is not too aligned with v (batch-safe)."""
    # v: [N,3]
    # choose axis with smallest absolute dot to v
    abs_v = v.abs()
    # candidate axes
    e1 = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    e3 = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=v.dtype).expand_as(v)
    # heuristic: if |vx| is smallest, use x-axis etc.
    idx = abs_v.argmin(dim=-1)  # [N]
    out = torch.where(idx[:, None] == 0, e1, torch.where(idx[:, None] == 1, e2, e3))
    return out


def compute_residue_local_frames(
    bb_pos: torch.Tensor,
    sc1_pos: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a right-handed orthonormal frame per residue from BB coordinates.

    Frame definition (for residue i):
      - origin O_i = BB_i
      - e1: forward direction along backbone
      - e2: roughly points towards previous residue (or towards SC1 if available)
      - e3 = e1 x e2

    All computations are SE(3)-equivariant: rotating the whole structure rotates
    each frame.

    Parameters
    ----------
    bb_pos:
        [N,3] BB positions in global coordinates.
    sc1_pos:
        Optional [N,3] SC1 positions (global). Used to stabilize terminal frames.
        If provided, entries for residues without SC1 can be set to NaN and will
        be ignored.
    eps:
        Numerical epsilon.

    Returns
    -------
    R:
        [N,3,3] rotation matrices with columns [e1,e2,e3].
        Local coordinates v_local = R^T (v_global - origin).
        Global from local v_global = origin + R v_local.
    """
    if bb_pos.ndim != 2 or bb_pos.shape[-1] != 3:
        raise ValueError(f"bb_pos must be [N,3], got {tuple(bb_pos.shape)}")
    N = bb_pos.shape[0]
    if N < 1:
        raise ValueError("bb_pos must have at least one residue")

    # forward (to next) and backward (to prev) vectors
    fwd = torch.zeros_like(bb_pos)
    bwd = torch.zeros_like(bb_pos)

    if N == 1:
        # Arbitrary frame
        e1 = torch.tensor([1.0, 0.0, 0.0], device=bb_pos.device, dtype=bb_pos.dtype).repeat(1, 1)
        e2 = torch.tensor([0.0, 1.0, 0.0], device=bb_pos.device, dtype=bb_pos.dtype).repeat(1, 1)
        e3 = torch.tensor([0.0, 0.0, 1.0], device=bb_pos.device, dtype=bb_pos.dtype).repeat(1, 1)
        R = torch.stack([e1, e2, e3], dim=-1).unsqueeze(0)  # [1,3,3]
        return R

    fwd[:-1] = bb_pos[1:] - bb_pos[:-1]
    fwd[-1] = bb_pos[-1] - bb_pos[-2]
    bwd[1:] = bb_pos[1:] - bb_pos[:-1]
    bwd[0] = bb_pos[1] - bb_pos[0]

    e1 = _safe_normalize(fwd, eps=eps)

    # Choose a second direction
    # Prefer backward direction for internal residues, SC1 direction for terminals if available
    e2_seed = torch.zeros_like(bb_pos)

    if sc1_pos is not None:
        if sc1_pos.shape != bb_pos.shape:
            raise ValueError(f"sc1_pos must have same shape as bb_pos, got {tuple(sc1_pos.shape)}")
        # Use SC1 direction when it's finite (not NaN)
        sc_dir = sc1_pos - bb_pos  # [N,3]
        finite = torch.isfinite(sc_dir).all(dim=-1) & (sc_dir.abs().sum(dim=-1) > 0)
        e2_seed = torch.where(finite[:, None], sc_dir, bwd)
    else:
        e2_seed = bwd

    # Gram-Schmidt: make e2 perpendicular to e1
    proj = (e2_seed * e1).sum(dim=-1, keepdim=True) * e1
    e2 = e2_seed - proj
    # If e2 is too small (colinear), pick an arbitrary perpendicular vector
    small = (_safe_norm(e2, eps=eps).squeeze(-1) < 1e-3)
    if small.any():
        arb = _pick_arbitrary_perp(e1)
        proj2 = (arb * e1).sum(dim=-1, keepdim=True) * e1
        e2_alt = arb - proj2
        e2 = torch.where(small[:, None], e2_alt, e2)
    e2 = _safe_normalize(e2, eps=eps)

    e3 = torch.cross(e1, e2, dim=-1)
    e3 = _safe_normalize(e3, eps=eps)

    # Re-orthogonalize e2 to reduce accumulated numeric error
    e2 = torch.cross(e3, e1, dim=-1)
    e2 = _safe_normalize(e2, eps=eps)

    R = torch.stack([e1, e2, e3], dim=-1)  # [N,3,3] columns are basis vectors
    return R


def global_to_local(
    x_global: torch.Tensor,
    origin_global: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Transform global coordinates to local frame.

    Parameters
    ----------
    x_global:
        [...,3]
    origin_global:
        [...,3] broadcastable to x_global
    R:
        [...,3,3] broadcastable to x_global

    Returns
    -------
    x_local:
        [...,3] where x_local = R^T (x_global - origin)
    """
    v = x_global - origin_global
    return torch.matmul(R.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)


def local_to_global(
    x_local: torch.Tensor,
    origin_global: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Transform local coordinates to global frame."""
    v = torch.matmul(R, x_local.unsqueeze(-1)).squeeze(-1)
    return origin_global + v


def clamp_norm(
    v: torch.Tensor,
    max_norm: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clamp vector norms by rescaling vectors that exceed max_norm."""
    n = _safe_norm(v, eps=eps)  # [...,1]
    scale = torch.clamp(max_norm / n, max=1.0)
    return v * scale

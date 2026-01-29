from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from backmap.geometry.dihedral import dihedral_angle, angle_to_sincos
from backmap.geometry.spherical import cartesian_to_spherical_sincos
from backmap.geometry.frames import global_to_local


def _safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


def _nan_to_num(x: torch.Tensor, nan: float = 0.0, posinf: float = 1e6, neginf: float = -1e6) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _clamp_loss(x: torch.Tensor, max_val: float) -> torch.Tensor:
    x = _nan_to_num(x)
    return torch.clamp(x, -max_val, max_val)


def spherical_reconstruction_loss(
    pred_local: torch.Tensor,
    true_local: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Loss on spherical sin/cos representation (rotation-invariant in local frame)."""
    pred_sph = cartesian_to_spherical_sincos(pred_local, eps=eps)
    true_sph = cartesian_to_spherical_sincos(true_local, eps=eps)

    # r: robust L1; angles: MSE on sin/cos
    r_loss = torch.nn.functional.smooth_l1_loss(pred_sph[..., 0], true_sph[..., 0], reduction="none")
    ang_loss = torch.nn.functional.mse_loss(pred_sph[..., 1:], true_sph[..., 1:], reduction="none").mean(dim=-1)

    loss = r_loss + ang_loss
    return _clamp_loss(loss, max_loss_value).mean()


def bead_distance_loss(
    pred_local: torch.Tensor,
    true_local: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distance-to-anchor-bead loss (equivalent to r-loss)."""
    r_pred = _safe_norm(pred_local, eps=eps)
    r_true = _safe_norm(true_local, eps=eps)
    loss = torch.nn.functional.smooth_l1_loss(r_pred, r_true, reduction="none")
    return _clamp_loss(loss, max_loss_value).mean()


def bond_length_loss(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    bond_pairs: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    if bond_pairs.numel() == 0:
        return pred_global.new_tensor(0.0)
    a = bond_pairs[:, 0]
    b = bond_pairs[:, 1]
    dp = pred_global[a] - pred_global[b]
    dt = true_global[a] - true_global[b]
    lp = _safe_norm(dp, eps=eps)
    lt = _safe_norm(dt, eps=eps)
    loss = torch.nn.functional.smooth_l1_loss(lp, lt, reduction="none")
    return _clamp_loss(loss, max_loss_value).mean()


def _angle_sincos(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return [cos(angle), sin(angle)] for angle p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = _safe_norm(v1, eps=eps)
    n2 = _safe_norm(v2, eps=eps)
    v1u = v1 / n1.unsqueeze(-1)
    v2u = v2 / n2.unsqueeze(-1)
    cos = torch.clamp((v1u * v2u).sum(dim=-1), -1.0, 1.0)
    sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
    return torch.stack([cos, sin], dim=-1)


def bond_angle_loss(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    angle_triples: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    if angle_triples.numel() == 0:
        return pred_global.new_tensor(0.0)
    a = angle_triples[:, 0]
    b = angle_triples[:, 1]
    c = angle_triples[:, 2]
    sp = _angle_sincos(pred_global[a], pred_global[b], pred_global[c], eps=eps)
    st = _angle_sincos(true_global[a], true_global[b], true_global[c], eps=eps)
    loss = torch.nn.functional.mse_loss(sp, st, reduction="none").mean(dim=-1)
    return _clamp_loss(loss, max_loss_value).mean()


def coulomb_loss(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    charged_idx: torch.Tensor,
    charges: torch.Tensor,
    charged_res_index: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
    exclude_neighbor_residues: int = 1,
    r_min: float = 0.5,
) -> torch.Tensor:
    """Electrostatic restraint: match long-range Coulomb energy (backbone only).

    We exclude interactions within the same residue and immediate neighbors (|Δres| <= exclude_neighbor_residues),
    as requested.
    """
    if charged_idx.numel() < 2:
        return pred_global.new_tensor(0.0)

    Xp = pred_global[charged_idx]  # [M,3]
    Xt = true_global[charged_idx]
    q = charges.to(dtype=pred_global.dtype)  # [M]
    res = charged_res_index  # [M]

    # pairwise distances
    # Compute full pairwise; M is small (<= 4 per residue).
    dp = Xp[:, None, :] - Xp[None, :, :]
    dt = Xt[:, None, :] - Xt[None, :, :]
    rp = torch.sqrt(torch.clamp((dp * dp).sum(dim=-1), min=eps))
    rt = torch.sqrt(torch.clamp((dt * dt).sum(dim=-1), min=eps))
    rp = torch.clamp(rp, min=r_min)
    rt = torch.clamp(rt, min=r_min)

    # mask: upper triangle, exclude self and nearby residues
    i = torch.arange(q.shape[0], device=pred_global.device)
    j = i[:, None]
    ii = i[None, :]
    # Actually create 2D indices
    idx_i = torch.arange(q.shape[0], device=pred_global.device).view(-1, 1)
    idx_j = torch.arange(q.shape[0], device=pred_global.device).view(1, -1)
    upper = idx_j > idx_i
    dr = (res.view(-1, 1) - res.view(1, -1)).abs()
    far = dr > exclude_neighbor_residues
    mask = upper & far

    qq = (q.view(-1, 1) * q.view(1, -1)).to(dtype=pred_global.dtype)
    Ep = (qq / rp) * mask
    Et = (qq / rt) * mask
    # sum energies
    Ep_sum = Ep.sum()
    Et_sum = Et.sum()

    loss = torch.nn.functional.smooth_l1_loss(Ep_sum, Et_sum, reduction="none")
    return _clamp_loss(loss, max_loss_value)


def dipole_loss(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    dip_C: torch.Tensor,
    dip_O: torch.Tensor,
    dip_Nn: torch.Tensor,
    dip_res_i: torch.Tensor,
    bb_origins: torch.Tensor,
    bb_frames: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dipole correlation restraint per peptide group.

    Dipole vector for peptide i is defined as:
        d = 0.665 * CO + 0.258 * CN
    where CO = O_i - C_i and CN = N_{i+1} - C_i.

    We compare unit dipole vectors (cosine similarity) in the local frame of residue i BB.
    """
    if dip_C.numel() == 0:
        return pred_global.new_tensor(0.0)

    CO_p = pred_global[dip_O] - pred_global[dip_C]
    CN_p = pred_global[dip_Nn] - pred_global[dip_C]
    d_p = 0.665 * CO_p + 0.258 * CN_p

    CO_t = true_global[dip_O] - true_global[dip_C]
    CN_t = true_global[dip_Nn] - true_global[dip_C]
    d_t = 0.665 * CO_t + 0.258 * CN_t

    # transform vectors to the local frame of residue i: d_local = R^T d_global
    R_i = bb_frames[dip_res_i]   # [K,3,3]
    d_p_loc = torch.matmul(R_i.transpose(-1, -2), d_p.unsqueeze(-1)).squeeze(-1)
    d_t_loc = torch.matmul(R_i.transpose(-1, -2), d_t.unsqueeze(-1)).squeeze(-1)

    u_p = d_p_loc / _safe_norm(d_p_loc, eps=eps).unsqueeze(-1)
    u_t = d_t_loc / _safe_norm(d_t_loc, eps=eps).unsqueeze(-1)

    corr = (u_p * u_t).sum(dim=-1)  # [K]
    loss = 1.0 - corr
    return _clamp_loss(loss, max_loss_value).mean()


def rama_loss(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    phi_idx: torch.Tensor,
    psi_idx: torch.Tensor,
    max_loss_value: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ramachandran loss using sin/cos of dihedral angles."""
    losses = []

    if phi_idx.numel() > 0:
        p1, p2, p3, p4 = [phi_idx[:, k] for k in range(4)]
        ang_p = dihedral_angle(pred_global[p1], pred_global[p2], pred_global[p3], pred_global[p4], eps=eps)
        ang_t = dihedral_angle(true_global[p1], true_global[p2], true_global[p3], true_global[p4], eps=eps)
        sp = angle_to_sincos(ang_p)
        st = angle_to_sincos(ang_t)
        losses.append(torch.nn.functional.mse_loss(sp, st, reduction="none").mean(dim=-1))

    if psi_idx.numel() > 0:
        p1, p2, p3, p4 = [psi_idx[:, k] for k in range(4)]
        ang_p = dihedral_angle(pred_global[p1], pred_global[p2], pred_global[p3], pred_global[p4], eps=eps)
        ang_t = dihedral_angle(true_global[p1], true_global[p2], true_global[p3], true_global[p4], eps=eps)
        sp = angle_to_sincos(ang_p)
        st = angle_to_sincos(ang_t)
        losses.append(torch.nn.functional.mse_loss(sp, st, reduction="none").mean(dim=-1))

    if not losses:
        return pred_global.new_tensor(0.0)

    loss = torch.cat(losses, dim=0)
    return _clamp_loss(loss, max_loss_value).mean()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    spherical: torch.Tensor
    bead_distance: torch.Tensor
    bond_length: torch.Tensor
    bond_angle: torch.Tensor
    coulomb: torch.Tensor
    dipole: torch.Tensor
    rama: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "spherical": float(self.spherical.detach().cpu()),
            "bead_distance": float(self.bead_distance.detach().cpu()),
            "bond_length": float(self.bond_length.detach().cpu()),
            "bond_angle": float(self.bond_angle.detach().cpu()),
            "coulomb": float(self.coulomb.detach().cpu()),
            "dipole": float(self.dipole.detach().cpu()),
            "rama": float(self.rama.detach().cpu()),
        }

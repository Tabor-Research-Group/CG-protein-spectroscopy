from __future__ import annotations

"""Loss functions for oscillator-local backmapping.

This module is designed to be *production safe*:

- It refuses to silently compute a loss on an empty atom tensor (Na==0).
  (That situation is the classic reason for losses being printed as 0.0000.)
- It avoids NaNs/Infs by clamping unstable operations (e.g., 1/r) and by using
  non-exploding close-contact penalties.
- It returns a structured breakdown suitable for JSON logging and per-epoch plots.

Coordinate systems
------------------
The model predicts atom coordinates in the **local residue frame**.

- Denoising losses are computed in local coordinates.
- Geometry/physics losses are computed in global coordinates after transforming
  predicted and target local coordinates back with the residue frames.

The batch must be created by :func:`backmap.data.collate.collate_graph_samples`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from backmap.config import LossConfig
from backmap.geometry.dihedral import dihedral_angle, angle_to_sincos
from backmap.geometry.spherical import cartesian_to_spherical_sincos
from backmap.model.pipeline import atoms_local_to_global


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _safe_norm(v: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


def _clamp_term(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Clamp a loss term to avoid exploding gradients.

    Note: we clamp *after* nan_to_num so NaN does not silently become 0.
    """
    x = torch.nan_to_num(x, nan=max_val, posinf=max_val, neginf=max_val)
    return torch.clamp(x, -float(max_val), float(max_val))


def _angle_sincos(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float) -> torch.Tensor:
    """Return sin/cos encoding of angle ABC.

    Uses cross-product norm for sine to improve stability near 0 and pi.
    """
    v1 = a - b
    v2 = c - b
    v1u = v1 / _safe_norm(v1, eps).unsqueeze(-1)
    v2u = v2 / _safe_norm(v2, eps).unsqueeze(-1)

    cos = torch.clamp((v1u * v2u).sum(dim=-1), -1.0, 1.0)
    sin = torch.linalg.norm(torch.cross(v1u, v2u, dim=-1), dim=-1)
    return torch.stack([cos, sin], dim=-1)


# -----------------------------------------------------------------------------
# Loss breakdown
# -----------------------------------------------------------------------------


@dataclass
class LossBreakdown:
    """Scalar losses for logging plus a debug flag.

    Why the `bad` flag?
    -------------------
    If a batch contains NaNs/Infs (in inputs or intermediate geometry), PyTorch
    will happily propagate them through the model. One optimizer step with NaN
    gradients can permanently poison the weights, after which *every* batch will
    look bad.

    We therefore surface a `bad` flag so the training loop can **skip** such
    batches *before* backward/optimizer.step().
    """

    total: torch.Tensor

    denoise_cart: torch.Tensor
    denoise_sph: torch.Tensor

    bond: torch.Tensor
    angle: torch.Tensor
    dihedral: torch.Tensor
    dipole: torch.Tensor

    contact: torch.Tensor

    # Debug info
    bad: bool = False
    bad_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "total": float(self.total.detach().cpu()),
            "denoise_cart": float(self.denoise_cart.detach().cpu()),
            "denoise_sph": float(self.denoise_sph.detach().cpu()),
            "bond": float(self.bond.detach().cpu()),
            "angle": float(self.angle.detach().cpu()),
            "dihedral": float(self.dihedral.detach().cpu()),
            "dipole": float(self.dipole.detach().cpu()),
            "contact": float(self.contact.detach().cpu()),
            "bad": bool(self.bad),
            "bad_reason": str(self.bad_reason),
        }


# -----------------------------------------------------------------------------
# Individual terms
# -----------------------------------------------------------------------------


def denoise_cart_mse(pred_local: torch.Tensor, target_local: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_local, target_local)


def denoise_sph_mse(pred_local: torch.Tensor, target_local: torch.Tensor, eps: float) -> torch.Tensor:
    pred_sph = cartesian_to_spherical_sincos(pred_local, eps=eps)
    targ_sph = cartesian_to_spherical_sincos(target_local, eps=eps)
    return F.mse_loss(pred_sph, targ_sph)


def bond_length_loss(pred_global: torch.Tensor, target_global: torch.Tensor, bond_index: torch.Tensor, eps: float) -> torch.Tensor:
    if bond_index.numel() == 0:
        return pred_global.new_tensor(0.0)
    _require(bond_index.ndim == 2 and bond_index.shape[0] == 2, "bond_index must be [2,N]")
    i = bond_index[0]
    j = bond_index[1]
    d_pred = _safe_norm(pred_global[i] - pred_global[j], eps)
    d_true = _safe_norm(target_global[i] - target_global[j], eps)
    return F.smooth_l1_loss(d_pred, d_true)


def bond_angle_loss(pred_global: torch.Tensor, target_global: torch.Tensor, angle_index: torch.Tensor, eps: float) -> torch.Tensor:
    if angle_index.numel() == 0:
        return pred_global.new_tensor(0.0)
    _require(angle_index.ndim == 2 and angle_index.shape[0] == 3, "angle_index must be [3,N]")
    a, b, c = angle_index
    sc_pred = _angle_sincos(pred_global[a], pred_global[b], pred_global[c], eps)
    sc_true = _angle_sincos(target_global[a], target_global[b], target_global[c], eps)
    return F.mse_loss(sc_pred, sc_true)


def dihedral_loss(pred_global: torch.Tensor, target_global: torch.Tensor, dihedral_index: torch.Tensor, eps: float) -> torch.Tensor:
    if dihedral_index.numel() == 0:
        return pred_global.new_tensor(0.0)
    _require(dihedral_index.ndim == 2 and dihedral_index.shape[0] == 4, "dihedral_index must be [4,N]")
    a, b, c, d = dihedral_index
    ang_pred = dihedral_angle(pred_global[a], pred_global[b], pred_global[c], pred_global[d], eps=eps)
    ang_true = dihedral_angle(target_global[a], target_global[b], target_global[c], target_global[d], eps=eps)
    sc_pred = angle_to_sincos(ang_pred)
    sc_true = angle_to_sincos(ang_true)
    return F.mse_loss(sc_pred, sc_true)


def dipole_corr_loss(pred_global: torch.Tensor, target_global: torch.Tensor, dipole_index: torch.Tensor, eps: float) -> torch.Tensor:
    """1 - cosine similarity between dipole vectors.

    Dipole definition (backbone peptide dipole proxy):
        d = 0.665 * (O - C) + 0.258 * (N - C)

    If N is missing (N_idx=-1), we fall back to d = (O - C).
    """
    if dipole_index.numel() == 0:
        return pred_global.new_tensor(0.0)
    _require(dipole_index.ndim == 2 and dipole_index.shape[0] == 3, "dipole_index must be [3,N]")

    c_idx = dipole_index[0]
    o_idx = dipole_index[1]
    n_idx = dipole_index[2]

    CO_p = pred_global[o_idx] - pred_global[c_idx]
    CO_t = target_global[o_idx] - target_global[c_idx]

    # Optional CN term
    CN_p = torch.zeros_like(CO_p)
    CN_t = torch.zeros_like(CO_t)

    has_n = n_idx >= 0
    if bool(has_n.any()):
        n_valid = n_idx[has_n]
        CN_p[has_n] = pred_global[n_valid] - pred_global[c_idx[has_n]]
        CN_t[has_n] = target_global[n_valid] - target_global[c_idx[has_n]]

    # Use CN only where available
    coeff_cn = has_n.to(dtype=pred_global.dtype).unsqueeze(-1) * 0.258
    d_p = 0.665 * CO_p + coeff_cn * CN_p
    d_t = 0.665 * CO_t + coeff_cn * CN_t

    u_p = d_p / _safe_norm(d_p, eps).unsqueeze(-1)
    u_t = d_t / _safe_norm(d_t, eps).unsqueeze(-1)

    corr = torch.clamp((u_p * u_t).sum(dim=-1), -1.0, 1.0)
    return (1.0 - corr).mean()


def contact_loss(
    pred_global: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    *,
    r0: float,
    eps: float,
) -> torch.Tensor:
    """Penalize predicted non-bonded atom-atom contacts below r0.

    This is not meant to be a full force field. It is a **stability term** that
    discourages catastrophic self-intersections during training and sampling.

    Implementation notes
    --------------------
    - Operates *within each sample* using `atom_ptr` to avoid cross-sample pairs.
    - Excludes directly bonded pairs (from bond_index) from the contact penalty.
    - Uses a bounded quadratic hinge: relu(r0 - r)^2, which is stable and does
      not blow up to inf.
    """
    if "atom_ptr" not in batch:
        # This should never happen when using our collate function.
        return pred_global.new_tensor(0.0)

    atom_ptr = batch["atom_ptr"]
    bond_index = batch.get("bond_index")
    atom_batch = batch.get("atom_batch")

    B = int(atom_ptr.numel()) - 1
    if B <= 0:
        return pred_global.new_tensor(0.0)

    # Build per-sample bonded adjacency (tiny matrices; atoms per oscillator are small).
    bonded = [None for _ in range(B)]
    for b in range(B):
        n = int(atom_ptr[b + 1] - atom_ptr[b])
        if n <= 1:
            bonded[b] = None
        else:
            bonded[b] = torch.zeros((n, n), dtype=torch.bool, device=pred_global.device)

    if bond_index is not None and bond_index.numel() > 0 and atom_batch is not None:
        # bond_index is in global atom index space.
        for k in range(int(bond_index.shape[1])):
            i = int(bond_index[0, k].item())
            j = int(bond_index[1, k].item())
            b = int(atom_batch[i].item())
            # sanity: bonds should not connect samples
            if b != int(atom_batch[j].item()):
                continue
            a0 = int(atom_ptr[b].item())
            ii = i - a0
            jj = j - a0
            if bonded[b] is not None and 0 <= ii < bonded[b].shape[0] and 0 <= jj < bonded[b].shape[0]:
                bonded[b][ii, jj] = True
                bonded[b][jj, ii] = True

    total = pred_global.new_tensor(0.0)
    count = 0

    for b in range(B):
        a0 = int(atom_ptr[b].item())
        a1 = int(atom_ptr[b + 1].item())
        n = a1 - a0
        if n <= 1:
            continue

        X = pred_global[a0:a1]  # [n,3]
        # Full pairwise distances (n is tiny).
        diff = X[:, None, :] - X[None, :, :]
        dist = _safe_norm(diff, eps).clamp_min(eps)

        # Upper triangle i<j
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=X.device), diagonal=1)

        # Exclude bonded pairs
        if bonded[b] is not None:
            mask = mask & (~bonded[b])

        d = dist[mask]
        if d.numel() == 0:
            continue

        # Stable penalty (hinge)
        pen = F.relu(float(r0) - d) ** 2
        total = total + pen.sum()
        count += int(pen.numel())

    if count == 0:
        return pred_global.new_tensor(0.0)
    return total / float(count)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def compute_losses(
    *,
    pred_local: torch.Tensor,   # [Na,3]
    target_local: torch.Tensor, # [Na,3]
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> LossBreakdown:
    """Compute total loss and a breakdown.

    Parameters
    ----------
    pred_local:
        Predicted x0 in local residue coordinates.
    target_local:
        Ground truth x0 in local residue coordinates.
    batch:
        Collated batch dict.
    cfg:
        LossConfig with weights and stability settings.
    """
    _require(pred_local.shape == target_local.shape, "pred_local and target_local shape mismatch")

    Na = int(target_local.shape[0])
    _require(Na > 0, "Na==0 atoms in batch; this would produce a meaningless zero loss")

    # --- denoising losses (local) ---
    den_cart = denoise_cart_mse(pred_local, target_local)
    den_sph = pred_local.new_tensor(0.0)
    if float(cfg.w_denoise_sph) != 0.0:
        den_sph = denoise_sph_mse(pred_local, target_local, eps=cfg.eps)

    # --- convert to global for geometry losses ---
    pred_global = atoms_local_to_global(pred_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
    true_global = atoms_local_to_global(target_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])

    # --- geometry terms ---
    bond = pred_local.new_tensor(0.0)
    angle = pred_local.new_tensor(0.0)
    dihed = pred_local.new_tensor(0.0)
    dip = pred_local.new_tensor(0.0)
    cont = pred_local.new_tensor(0.0)

    if float(cfg.w_bond) != 0.0:
        bond = bond_length_loss(pred_global, true_global, batch.get("bond_index", pred_local.new_zeros((2, 0), dtype=torch.long)), eps=cfg.eps)

    if float(cfg.w_angle) != 0.0:
        angle = bond_angle_loss(pred_global, true_global, batch.get("angle_index", pred_local.new_zeros((3, 0), dtype=torch.long)), eps=cfg.eps)

    if float(cfg.w_dihedral) != 0.0:
        dihed = dihedral_loss(pred_global, true_global, batch.get("dihedral_index", pred_local.new_zeros((4, 0), dtype=torch.long)), eps=cfg.eps)

    if float(cfg.w_dipole) != 0.0:
        dip = dipole_corr_loss(pred_global, true_global, batch.get("dipole_index", pred_local.new_zeros((3, 0), dtype=torch.long)), eps=cfg.eps)

    if float(cfg.w_contact) != 0.0:
        cont = contact_loss(pred_global, batch, r0=float(cfg.contact_r0), eps=cfg.eps)

    # --- weighted sum ---
    total = (
        float(cfg.w_denoise_cart) * den_cart
        + float(cfg.w_denoise_sph) * den_sph
        + float(cfg.w_bond) * bond
        + float(cfg.w_angle) * angle
        + float(cfg.w_dihedral) * dihed
        + float(cfg.w_dipole) * dip
        + float(cfg.w_contact) * cont
    )


    # --- detect non-finite values BEFORE clamping ---
    #
    # Note: we still clamp for safety, but we must NOT allow a non-finite batch
    # to backprop / optimizer.step(), otherwise weights can become NaN.
    bad_terms = []

    def _flag_if_bad(name: str, t: torch.Tensor) -> None:
        try:
            if not torch.isfinite(t).all().item():
                bad_terms.append(name)
        except Exception:
            bad_terms.append(name)

    _flag_if_bad("pred_local", pred_local)
    _flag_if_bad("target_local", target_local)
    _flag_if_bad("bb_pos", batch["bb_pos"])
    _flag_if_bad("bb_frames", batch["bb_frames"])

    _flag_if_bad("denoise_cart", den_cart)
    _flag_if_bad("denoise_sph", den_sph)
    _flag_if_bad("bond", bond)
    _flag_if_bad("angle", angle)
    _flag_if_bad("dihedral", dihed)
    _flag_if_bad("dipole", dip)
    _flag_if_bad("contact", cont)
    _flag_if_bad("total", total)

    bad = len(bad_terms) > 0
    bad_reason = ",".join(bad_terms)

    # Clamp each term so logging and gradients don't blow up.
    maxv = float(cfg.max_term_value)
    total = _clamp_term(total, maxv)
    den_cart = _clamp_term(den_cart, maxv)
    den_sph = _clamp_term(den_sph, maxv)
    bond = _clamp_term(bond, maxv)
    angle = _clamp_term(angle, maxv)
    dihed = _clamp_term(dihed, maxv)
    dip = _clamp_term(dip, maxv)
    cont = _clamp_term(cont, maxv)

    return LossBreakdown(
        total=total,
        denoise_cart=den_cart,
        denoise_sph=den_sph,
        bond=bond,
        angle=angle,
        dihedral=dihed,
        dipole=dip,
        contact=cont,
        bad=bad,
        bad_reason=bad_reason,
    )

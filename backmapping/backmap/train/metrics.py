from __future__ import annotations

"""Per-batch metric extraction for plotting.

Loss scalars are necessary but not sufficient to debug backmapping quality.
This module extracts richer *distributions* from each batch so we can generate
per-epoch diagnostic plots:

- Dipole correlation histograms
- Bond length pred vs true scatter + error hist
- Angle/dihedral distributions (true vs pred)
- Radial distance distribution (|x| in local)
- Nonbonded minimum-distance and close-contact statistics
- A simple "repulsion energy" proxy based on close contacts

All computations are stable and avoid NaNs via clamping.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from backmap.config import LossConfig
from backmap.geometry.dihedral import dihedral_angle
from backmap.model.pipeline import atoms_local_to_global


@dataclass
class BatchMetrics:
    """A container for per-batch metric arrays.

    All tensors are 1D float tensors on CPU (ready to accumulate and plot).
    Some fields may be empty tensors if the batch has no corresponding indices.
    """

    dipole_cos: torch.Tensor

    bond_true: torch.Tensor
    bond_pred: torch.Tensor

    angle_true: torch.Tensor
    angle_pred: torch.Tensor

    dihedral_true: torch.Tensor
    dihedral_pred: torch.Tensor

    radial_true: torch.Tensor
    radial_pred: torch.Tensor

    nonbond_min_true: torch.Tensor
    nonbond_min_pred: torch.Tensor

    repulsion_energy_true: torch.Tensor
    repulsion_energy_pred: torch.Tensor


def _safe_norm(v: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


def _angle_value(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float) -> torch.Tensor:
    """Return angle ABC in radians."""
    v1 = a - b
    v2 = c - b
    n1 = _safe_norm(v1, eps=eps)
    n2 = _safe_norm(v2, eps=eps)
    v1u = v1 / n1.unsqueeze(-1)
    v2u = v2 / n2.unsqueeze(-1)
    cos = torch.clamp((v1u * v2u).sum(dim=-1), -1.0, 1.0)
    return torch.acos(cos)


def _dipole_cosine(
    pred_global: torch.Tensor,
    true_global: torch.Tensor,
    dipole_index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Cosine similarity between predicted and true dipole vectors.

    dipole_index columns are (C_idx, O_idx, N_idx) with N_idx=-1 if missing.
    """
    if dipole_index.numel() == 0:
        return pred_global.new_zeros((0,), dtype=torch.float32, device="cpu")

    c_idx = dipole_index[0]
    o_idx = dipole_index[1]
    n_idx = dipole_index[2]

    CO_p = pred_global[o_idx] - pred_global[c_idx]
    CO_t = true_global[o_idx] - true_global[c_idx]

    # CN is optional (N_idx may be -1)
    has_n = n_idx >= 0
    CN_p = torch.zeros_like(CO_p)
    CN_t = torch.zeros_like(CO_t)
    if bool(has_n.any()):
        CN_p[has_n] = pred_global[n_idx[has_n]] - pred_global[c_idx[has_n]]
        CN_t[has_n] = true_global[n_idx[has_n]] - true_global[c_idx[has_n]]

    # Dipole model used elsewhere in this repo
    d_p = 0.665 * CO_p + 0.258 * CN_p
    d_t = 0.665 * CO_t + 0.258 * CN_t

    u_p = d_p / _safe_norm(d_p, eps=eps).unsqueeze(-1)
    u_t = d_t / _safe_norm(d_t, eps=eps).unsqueeze(-1)
    cos = torch.clamp((u_p * u_t).sum(dim=-1), -1.0, 1.0)

    return cos.detach().to(device="cpu", dtype=torch.float32)


def _nonbond_min_and_repulsion(
    X: torch.Tensor,
    *,
    atom_ptr: torch.Tensor,
    bond_index: torch.Tensor,
    atom_batch: torch.Tensor,
    contact_r0: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample minimum nonbonded distance and a repulsion proxy.

    The repulsion proxy is the *sum* of ReLU(contact_r0 - r)^2 over nonbonded
    pairs (i<j), returned per sample.

    This is used for plots, not as a physical energy.
    """
    B = int(atom_ptr.numel() - 1)
    min_d = []
    repE = []

    # Build bonded adjacency sets per sample (local indices)
    bonded = [set() for _ in range(B)]
    if bond_index.numel() > 0:
        bi = bond_index[0]
        bj = bond_index[1]
        for k in range(int(bi.numel())):
            i = int(bi[k].item())
            j = int(bj[k].item())
            s = int(atom_batch[i].item())
            a0 = int(atom_ptr[s].item())
            li = i - a0
            lj = j - a0
            if li < 0 or lj < 0:
                continue
            if li == lj:
                continue
            bonded[s].add((min(li, lj), max(li, lj)))

    for s in range(B):
        a0 = int(atom_ptr[s].item())
        a1 = int(atom_ptr[s + 1].item())
        n = a1 - a0
        if n < 2:
            min_d.append(float("nan"))
            repE.append(0.0)
            continue

        coords = X[a0:a1]  # [n,3]
        # pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]
        dist = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=eps))  # [n,n]

        # upper triangle mask
        mask = torch.triu(torch.ones((n, n), device=dist.device, dtype=torch.bool), diagonal=1)

        # exclude bonded
        if bonded[s]:
            bonded_mask = torch.zeros((n, n), device=dist.device, dtype=torch.bool)
            for (i, j) in bonded[s]:
                if 0 <= i < n and 0 <= j < n:
                    bonded_mask[i, j] = True
                    bonded_mask[j, i] = True
            mask = mask & (~bonded_mask)

        d = dist[mask]
        if d.numel() == 0:
            min_d.append(float("nan"))
            repE.append(0.0)
            continue

        min_d.append(float(d.min().detach().cpu()))

        # Repulsion proxy
        pen = torch.relu(float(contact_r0) - d) ** 2
        repE.append(float(pen.sum().detach().cpu()))

    return (
        torch.tensor(min_d, dtype=torch.float32, device="cpu"),
        torch.tensor(repE, dtype=torch.float32, device="cpu"),
    )


@torch.no_grad()
def compute_batch_metrics(
    *,
    pred_local: torch.Tensor,
    target_local: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> BatchMetrics:
    """Compute per-batch metric distributions.

    Parameters
    ----------
    pred_local / target_local:
        [Na,3] local coordinates.
    batch:
        Collated batch dict (see :func:`backmap.data.collate.collate_graph_samples`).
    cfg:
        LossConfig for eps and contact threshold.
    """
    eps = float(cfg.eps)

    # Global conversion for geometry/dipoles
    pred_global = atoms_local_to_global(pred_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
    true_global = atoms_local_to_global(target_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])

    # --- radial distances ---
    radial_true = _safe_norm(target_local, eps=eps).detach().to(device="cpu", dtype=torch.float32)
    radial_pred = _safe_norm(pred_local, eps=eps).detach().to(device="cpu", dtype=torch.float32)

    # --- dipoles ---
    dipole_cos = _dipole_cosine(pred_global, true_global, batch["dipole_index"], eps=eps)

    # --- bonds ---
    if batch["bond_index"].numel() > 0:
        i = batch["bond_index"][0]
        j = batch["bond_index"][1]
        bond_true = _safe_norm(true_global[i] - true_global[j], eps=eps).detach().to(device="cpu", dtype=torch.float32)
        bond_pred = _safe_norm(pred_global[i] - pred_global[j], eps=eps).detach().to(device="cpu", dtype=torch.float32)
    else:
        bond_true = torch.zeros((0,), dtype=torch.float32, device="cpu")
        bond_pred = torch.zeros((0,), dtype=torch.float32, device="cpu")

    # --- angles ---
    if batch["angle_index"].numel() > 0:
        a, b, c = batch["angle_index"]
        angle_true = _angle_value(true_global[a], true_global[b], true_global[c], eps=eps).detach().to(device="cpu", dtype=torch.float32)
        angle_pred = _angle_value(pred_global[a], pred_global[b], pred_global[c], eps=eps).detach().to(device="cpu", dtype=torch.float32)
    else:
        angle_true = torch.zeros((0,), dtype=torch.float32, device="cpu")
        angle_pred = torch.zeros((0,), dtype=torch.float32, device="cpu")

    # --- dihedrals ---
    if batch["dihedral_index"].numel() > 0:
        a, b, c, d = batch["dihedral_index"]
        dihedral_true = dihedral_angle(true_global[a], true_global[b], true_global[c], true_global[d], eps=eps).detach().to(device="cpu", dtype=torch.float32)
        dihedral_pred = dihedral_angle(pred_global[a], pred_global[b], pred_global[c], pred_global[d], eps=eps).detach().to(device="cpu", dtype=torch.float32)
    else:
        dihedral_true = torch.zeros((0,), dtype=torch.float32, device="cpu")
        dihedral_pred = torch.zeros((0,), dtype=torch.float32, device="cpu")

    # --- nonbonded min distance + repulsion energy proxy (per sample) ---
    nonbond_min_true, repE_true = _nonbond_min_and_repulsion(
        true_global,
        atom_ptr=batch["atom_ptr"],
        bond_index=batch["bond_index"],
        atom_batch=batch["atom_batch"],
        contact_r0=float(cfg.contact_r0),
        eps=eps,
    )
    nonbond_min_pred, repE_pred = _nonbond_min_and_repulsion(
        pred_global,
        atom_ptr=batch["atom_ptr"],
        bond_index=batch["bond_index"],
        atom_batch=batch["atom_batch"],
        contact_r0=float(cfg.contact_r0),
        eps=eps,
    )

    return BatchMetrics(
        dipole_cos=dipole_cos,
        bond_true=bond_true,
        bond_pred=bond_pred,
        angle_true=angle_true,
        angle_pred=angle_pred,
        dihedral_true=dihedral_true,
        dihedral_pred=dihedral_pred,
        radial_true=radial_true,
        radial_pred=radial_pred,
        nonbond_min_true=nonbond_min_true,
        nonbond_min_pred=nonbond_min_pred,
        repulsion_energy_true=repE_true,
        repulsion_energy_pred=repE_pred,
    )


def merge_metrics(metrics: List[BatchMetrics]) -> Dict[str, np.ndarray]:
    """Merge a list of BatchMetrics into numpy arrays.

    The returned dict is designed for plotting routines.
    """
    if not metrics:
        return {
            "dipole_cos": np.zeros((0,), dtype=np.float32),
            "bond_true": np.zeros((0,), dtype=np.float32),
            "bond_pred": np.zeros((0,), dtype=np.float32),
            "angle_true": np.zeros((0,), dtype=np.float32),
            "angle_pred": np.zeros((0,), dtype=np.float32),
            "dihedral_true": np.zeros((0,), dtype=np.float32),
            "dihedral_pred": np.zeros((0,), dtype=np.float32),
            "radial_true": np.zeros((0,), dtype=np.float32),
            "radial_pred": np.zeros((0,), dtype=np.float32),
            "nonbond_min_true": np.zeros((0,), dtype=np.float32),
            "nonbond_min_pred": np.zeros((0,), dtype=np.float32),
            "repulsion_energy_true": np.zeros((0,), dtype=np.float32),
            "repulsion_energy_pred": np.zeros((0,), dtype=np.float32),
        }

    def _cat(field: str) -> np.ndarray:
        xs = [getattr(m, field).detach().cpu().numpy() for m in metrics]
        if not xs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0)

    return {
        "dipole_cos": _cat("dipole_cos"),
        "bond_true": _cat("bond_true"),
        "bond_pred": _cat("bond_pred"),
        "angle_true": _cat("angle_true"),
        "angle_pred": _cat("angle_pred"),
        "dihedral_true": _cat("dihedral_true"),
        "dihedral_pred": _cat("dihedral_pred"),
        "radial_true": _cat("radial_true"),
        "radial_pred": _cat("radial_pred"),
        "nonbond_min_true": _cat("nonbond_min_true"),
        "nonbond_min_pred": _cat("nonbond_min_pred"),
        "repulsion_energy_true": _cat("repulsion_energy_true"),
        "repulsion_energy_pred": _cat("repulsion_energy_pred"),
    }

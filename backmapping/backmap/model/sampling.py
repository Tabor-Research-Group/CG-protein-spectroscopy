from __future__ import annotations

from typing import Dict, Any, Tuple

import torch

from backmap.model.gnn import BackmapGNN, EdgeCutoffs
from backmap.model.diffusion import GaussianDiffusion
from backmap.model.pipeline import (
    atoms_local_to_global,
    build_node_pos,
    build_node_geom_sph,
    build_node_atom_group,
    build_atom_node_mask,
    clamp_local_atoms,
)


@torch.no_grad()
def sample_atoms_full(
    *,
    model: BackmapGNN,
    diffusion: GaussianDiffusion,
    batch: Dict[str, Any],
    timesteps: int,
    max_atom_radius: float,
    edge_cutoffs: EdgeCutoffs,
    eps: float = 1e-8,
    init: str = "gaussian",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full reverse diffusion in local space for all atoms in a collated batch.

    Returns:
      x_final_local: [Na,3]
      x_final_global: [Na,3]
    """
    device = batch["x0_local"].device
    Na = batch["x0_local"].shape[0]
    Nn = int(batch["num_nodes"])

    # init x_T in local frame (centered at BB)
    if init == "gaussian":
        xt_local = torch.randn((Na, 3), device=device, dtype=batch["x0_local"].dtype)
    elif init == "uniform_ball":
        v = torch.randn((Na, 3), device=device, dtype=batch["x0_local"].dtype)
        v = v / torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=eps))
        u = torch.rand((Na, 1), device=device, dtype=batch["x0_local"].dtype)
        r = float(max_atom_radius) * (u ** (1.0 / 3.0))
        xt_local = v * r
    else:
        raise ValueError(f"Unknown init: {init}")
    xt_local = clamp_local_atoms(xt_local, max_atom_radius, eps=eps)

    # fixed node attributes
    node_atom_group = build_node_atom_group(Nn, batch["atom_node_indices"], batch["atom_group"])
    atom_mask = build_atom_node_mask(Nn, batch["atom_node_indices"])

    for t in reversed(range(timesteps)):
        xt_global = atoms_local_to_global(xt_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
        node_pos = build_node_pos(
            num_nodes=Nn,
            bead_node_indices=batch["bead_node_indices"],
            bead_pos=batch["bead_pos"],
            atom_node_indices=batch["atom_node_indices"],
            atom_pos=xt_global,
        )
        node_geom = build_node_geom_sph(Nn, batch["atom_node_indices"], xt_local, eps=eps)
        t_node = torch.full((Nn,), int(t), device=device, dtype=torch.long)

        _, x0_pred_local = model(
            node_pos=node_pos,
            node_type=batch["node_type"],
            node_name=batch["node_name"],
            node_resname=batch["node_resname"],
            node_res=batch["node_res"],
            node_res_in_frame=batch["node_res_in_frame"],
            node_atom_group=node_atom_group,
            edge_index=batch["edge_index"],
            edge_type=batch["edge_type"],
            bb_frames=batch["bb_frames"],
            t_node=t_node,
            node_geom_sph=node_geom,
            atom_node_mask=atom_mask,
            edge_cutoffs=edge_cutoffs,
        )

        t_atom = torch.full((Na,), int(t), device=device, dtype=torch.long)
        mean, var, _ = diffusion.p_mean_variance(xt_local, t_atom, x0_pred_local)
        if t == 0:
            xt_local = mean
        else:
            noise = torch.randn_like(xt_local)
            xt_local = mean + torch.sqrt(torch.clamp(var, min=eps)).unsqueeze(-1) * noise
        xt_local = clamp_local_atoms(xt_local, max_atom_radius, eps=eps)

    x_final_local = xt_local
    x_final_global = atoms_local_to_global(x_final_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
    return x_final_local, x_final_global

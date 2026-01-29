from __future__ import annotations

from typing import Dict, Tuple

import torch

from backmap.geometry.frames import local_to_global, global_to_local, clamp_norm
from backmap.geometry.spherical import cartesian_to_spherical_sincos


def atoms_local_to_global(
    atom_local: torch.Tensor,   # [Na,3] in residue-local frame
    atom_res: torch.Tensor,     # [Na] global residue indices
    bb_pos: torch.Tensor,       # [Nres,3]
    bb_frames: torch.Tensor,    # [Nres,3,3]
) -> torch.Tensor:
    origin = bb_pos[atom_res]
    R = bb_frames[atom_res]
    return local_to_global(atom_local, origin, R)


def atoms_global_to_local(
    atom_global: torch.Tensor,  # [Na,3]
    atom_res: torch.Tensor,     # [Na]
    bb_pos: torch.Tensor,       # [Nres,3]
    bb_frames: torch.Tensor,    # [Nres,3,3]
) -> torch.Tensor:
    origin = bb_pos[atom_res]
    R = bb_frames[atom_res]
    return global_to_local(atom_global, origin, R)


def build_node_pos(
    num_nodes: int,
    bead_node_indices: torch.Tensor,  # [Nb_total]
    bead_pos: torch.Tensor,           # [Nb_total,3]
    atom_node_indices: torch.Tensor,  # [Na_total]
    atom_pos: torch.Tensor,           # [Na_total,3]
) -> torch.Tensor:
    x = bead_pos.new_zeros((int(num_nodes), 3))
    x[bead_node_indices] = bead_pos
    x[atom_node_indices] = atom_pos
    return x


def build_node_geom_sph(
    num_nodes: int,
    atom_node_indices: torch.Tensor,  # [Na]
    atom_local: torch.Tensor,         # [Na,3] (e.g., x_t in local frame)
    eps: float = 1e-8,
) -> torch.Tensor:
    sph = cartesian_to_spherical_sincos(atom_local, eps=eps)  # [Na,5]
    out = sph.new_zeros((int(num_nodes), 5))
    out[atom_node_indices] = sph
    return out


def build_node_atom_group(
    num_nodes: int,
    atom_node_indices: torch.Tensor,  # [Na]
    atom_group: torch.Tensor,         # [Na]
) -> torch.Tensor:
    out = atom_group.new_zeros((int(num_nodes),), dtype=torch.long)
    out[atom_node_indices] = atom_group.to(dtype=torch.long)
    return out


def build_atom_node_mask(
    num_nodes: int,
    atom_node_indices: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros((int(num_nodes),), dtype=torch.bool, device=atom_node_indices.device)
    mask[atom_node_indices] = True
    return mask


def clamp_local_atoms(atom_local: torch.Tensor, max_radius: float, eps: float = 1e-8) -> torch.Tensor:
    return clamp_norm(atom_local, float(max_radius), eps=eps)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from backmap.geometry.rbf import rbf_expand
from backmap.geometry.spherical import spherical_sincos_to_cartesian
from backmap.model.embeddings import SinusoidalEmbedding, TimeEmbedding


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class EdgeCutoffs:
    bead_bead: float = 10.0
    atom_any: float = 10.0


class BackmapGNN(nn.Module):
    """Graph network that predicts per-atom local spherical coordinates.

    Key design choice:
      - All geometric edge direction features are expressed in the *receiver node's*
        residue-local frame. This makes edge features invariant to global rotations,
        while the overall mapping from CG geometry to atom positions is SE(3)-equivariant
        because local frames rotate with the protein.
    """

    def __init__(
        self,
        *,
        num_resnames: int,
        num_names: int,
        num_node_types: int,
        num_atom_groups: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        pos_embed_dim: int = 64,
        rbf_num_centers: int = 16,
        rbf_max_dist: float = 20.0,
        edge_type_embed_dim: int = 32,
        max_atom_radius: float = 6.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rbf_num_centers = rbf_num_centers
        self.rbf_max_dist = rbf_max_dist
        self.max_atom_radius = float(max_atom_radius)
        self.eps = eps

        self.emb_node_type = nn.Embedding(num_node_types, 32)
        self.emb_name = nn.Embedding(num_names, 64)
        self.emb_resname = nn.Embedding(num_resnames, 64)
        self.emb_atom_group = nn.Embedding(num_atom_groups, 16)

        self.pos_emb = SinusoidalEmbedding(pos_embed_dim if pos_embed_dim % 2 == 0 else pos_embed_dim + 1)
        self.time_emb = TimeEmbedding(time_embed_dim if time_embed_dim % 2 == 0 else time_embed_dim + 1)

        # Project all scalar inputs to hidden_dim
        in0 = 32 + 64 + 64 + 16 + self.pos_emb.dim + time_embed_dim + 32  # + geom(5)->32
        self.geom_proj = nn.Linear(5, 32)
        self.input_proj = nn.Linear(in0, hidden_dim)

        # Edge-type embedding (edge_type is small int)
        self.emb_edge_type = nn.Embedding(8, edge_type_embed_dim)

        edge_in = 2 * hidden_dim + rbf_num_centers + 3 + edge_type_embed_dim
        self.edge_mlp = MLP(edge_in, hidden_dim, hidden_dim, dropout=dropout)
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # Output head: 5 numbers (r, cosθ, sinθ, cosφ, sinφ)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(
        self,
        *,
        node_pos: torch.Tensor,           # [Nn,3] global
        node_type: torch.Tensor,          # [Nn]
        node_name: torch.Tensor,          # [Nn]
        node_resname: torch.Tensor,       # [Nn]
        node_res: torch.Tensor,           # [Nn] global residue index into bb_frames
        node_res_in_frame: torch.Tensor,  # [Nn] residue index starting at 0 per frame (for pos encoding)
        node_atom_group: torch.Tensor,    # [Nn], 0 for beads, 0/1 for atoms
        edge_index: torch.Tensor,         # [2,E] (src,dst)
        edge_type: torch.Tensor,          # [E]
        bb_frames: torch.Tensor,          # [Nres_total,3,3]
        t_node: torch.Tensor,             # [Nn] integer timestep
        node_geom_sph: torch.Tensor,      # [Nn,5] (atoms: xt spherical in local frame; beads: zeros)
        atom_node_mask: torch.Tensor,     # [Nn] bool
        edge_cutoffs: EdgeCutoffs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pred_sph_atoms, pred_local_atoms).

        pred_local_atoms are in the *atom's residue-local frame* relative to its BB origin.
        """
        if node_pos.ndim != 2 or node_pos.shape[-1] != 3:
            raise ValueError(f"node_pos must be [N,3], got {tuple(node_pos.shape)}")
        Nn = node_pos.shape[0]
        if atom_node_mask.shape[0] != Nn:
            raise ValueError("atom_node_mask shape mismatch")

        # Initial node features
        h = torch.cat(
            [
                self.emb_node_type(node_type),
                self.emb_name(node_name),
                self.emb_resname(node_resname),
                self.emb_atom_group(node_atom_group),
                self.pos_emb(node_res_in_frame.to(dtype=torch.float32)),
                self.time_emb(t_node),
                self.geom_proj(node_geom_sph),
            ],
            dim=-1,
        )
        h = self.input_proj(h)

        src = edge_index[0]
        dst = edge_index[1]

        # Precompute receiver frames for edge direction in local coordinates
        R_dst = bb_frames[node_res[dst]]  # [E,3,3]
        rel = node_pos[src] - node_pos[dst]  # [E,3] vector from dst to src (global)
        dist = torch.sqrt(torch.clamp((rel * rel).sum(dim=-1), min=self.eps))  # [E]

        # Dynamic edge masking by type
        # type 0: BB-BB -> bead_bead cutoff
        # type 3,4: interactions -> atom_any cutoff
        # type 1,2,5: always keep
        t = edge_type
        keep = torch.ones_like(dist, dtype=torch.bool)
        is_bb = (t == 0)
        is_int = (t == 3) | (t == 4)
        keep = keep & (~is_bb | (dist <= float(edge_cutoffs.bead_bead)))
        keep = keep & (~is_int | (dist <= float(edge_cutoffs.atom_any)))

        # Filter edges (vectorized)
        src = src[keep]
        dst = dst[keep]
        rel = rel[keep]
        dist = dist[keep]
        t = t[keep]
        R_dst = R_dst[keep]

        # Edge geometric features in receiver local frame
        rel_local = torch.matmul(R_dst.transpose(-1, -2), rel.unsqueeze(-1)).squeeze(-1)  # [Ef,3]
        dir_local = rel_local / dist.unsqueeze(-1)  # [Ef,3]

        rbf = rbf_expand(dist, num_centers=self.rbf_num_centers, rbf_max_dist=self.rbf_max_dist, eps=self.eps)  # [Ef,K]
        et = self.emb_edge_type(torch.clamp(t, 0, self.emb_edge_type.num_embeddings - 1))
        # Build messages
        h_src = h[src]
        h_dst = h[dst]
        edge_in = torch.cat([h_src, h_dst, rbf, dir_local, et], dim=-1)
        m = self.edge_mlp(edge_in)

        # Aggregate by dst (sum)
        agg = torch.zeros((Nn, self.hidden_dim), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, m)

        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Repeat layers (recompute edge geometry each layer? Invariant features depend on node_pos which changes only via diffusion input,
        # not by the network. We keep them fixed within this forward for speed.)
        for _ in range(self.num_layers - 1):
            # messages based on updated h but same geometry
            h_src = h[src]
            h_dst = h[dst]
            edge_in = torch.cat([h_src, h_dst, rbf, dir_local, et], dim=-1)
            m = self.edge_mlp(edge_in)
            agg.zero_()
            agg.index_add_(0, dst, m)
            h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Output spherical parameters for atoms only
        h_atoms = h[atom_node_mask]
        raw = self.out_mlp(h_atoms)  # [Na,5]
        raw_r = raw[:, 0]
        raw_ct = raw[:, 1]
        raw_st = raw[:, 2]
        raw_cp = raw[:, 3]
        raw_sp = raw[:, 4]

        r = self.max_atom_radius * torch.sigmoid(raw_r)

        # Normalize trig pairs
        ct_st = torch.stack([raw_ct, raw_st.abs()], dim=-1)
        ct_st = ct_st / torch.sqrt(torch.clamp((ct_st * ct_st).sum(dim=-1, keepdim=True), min=self.eps))
        cp_sp = torch.stack([raw_cp, raw_sp], dim=-1)
        cp_sp = cp_sp / torch.sqrt(torch.clamp((cp_sp * cp_sp).sum(dim=-1, keepdim=True), min=self.eps))

        cos_theta = ct_st[:, 0]
        sin_theta = ct_st[:, 1]
        cos_phi = cp_sp[:, 0]
        sin_phi = cp_sp[:, 1]

        pred_sph = torch.stack([r, cos_theta, sin_theta, cos_phi, sin_phi], dim=-1)
        pred_local = spherical_sincos_to_cartesian(pred_sph, eps=self.eps)

        return pred_sph, pred_local

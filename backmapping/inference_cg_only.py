#!/usr/bin/env python3
"""
================================================================================
CG-ONLY INFERENCE: Backmapping from Coarse-Grained Trajectories
================================================================================

This script performs inference using ONLY coarse-grained (CG) bead positions.
No atomistic trajectory is required - the model predicts atomic positions
directly from CG beads.

Key differences from inference.py:
  - extract_atomistic is not needed and ignored
  - Atom topology (names, count) is determined from oscillator type + residue name
  - No placeholder atoms with fake coordinates - graph built directly from CG beads
  - Ground truth comparison metrics are not available (no atomistic data)

Usage:
    python inference_cg_only.py \\
        --config config_inference_traj.yaml \\
        --checkpoint path/to/checkpoint.pt

Output pickle format:
  - Each oscillator has 'predicted_atoms': Dict[str, np.ndarray]
  - If compute_analysis=true: 'predicted_rama_nnfs' and 'predicted_dipole'
  - No 'atoms' field (no ground truth)

================================================================================
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml

# Add backmap to path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backmap.config import Config
from backmap.data.collate import collate_graph_samples
from backmap.data.oscillator_dataset import build_default_vocab_from_pickle
from backmap.data.oscillator_graph import build_graph_from_oscillator
from backmap.model.diffusion import GaussianDiffusion, make_schedule
from backmap.model.gnn import BackmapGNN, EdgeCutoffs
from backmap.model.pipeline import (
    atoms_local_to_global,
    build_atom_node_mask,
    build_node_atom_group,
    build_node_geom_sph,
    build_node_pos,
    clamp_local_atoms,
)
from backmap.utils.checkpoint import load_checkpoint
from backmap.geometry.frames import compute_residue_local_frames
    
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Atom Topology Definitions
# ============================================================================

# Backbone oscillator atoms (peptide bond C=O between residue i and i+1)
BACKBONE_ATOMS = [
    ("N_prev", 0),    # from residue i
    ("CA_prev", 0),
    ("C_prev", 0),
    ("O_prev", 0),
    ("N_curr", 1),    # from residue i+1
    ("H_curr", 1),
    ("CA_curr", 1),
]

# Sidechain oscillator atoms by residue type
SIDECHAIN_ATOMS = {
    "GLN": [
        ("CA", 0),
        ("CB", 0),
        ("CG", 0),
        ("CD", 0),
        ("OE1", 0),
        ("NE2", 0),
        ("HE21", 0),
        ("HE22", 0),
    ],
    "ASN": [
        ("CA", 0),
        ("CB", 0),
        ("CG", 0),
        ("OD1", 0),
        ("ND2", 0),
        ("HD21", 0),
        ("HD22", 0),
    ],
}


def get_atom_topology(osc_type: str, residue_name: str) -> List[Tuple[str, int]]:
    """
    Get the atom topology (names and residue indices) for an oscillator.

    Args:
        osc_type: 'backbone' or 'sidechain'
        residue_name: e.g., 'ALA', 'GLN', 'ASN'

    Returns:
        List of (atom_name, local_residue_index) tuples
    """
    if osc_type == "backbone":
        return BACKBONE_ATOMS
    elif osc_type == "sidechain":
        # Strip '-SC' suffix if present
        base_resname = residue_name.replace("-SC", "")
        if base_resname in SIDECHAIN_ATOMS:
            return SIDECHAIN_ATOMS[base_resname]
        else:
            raise ValueError(f"Unknown sidechain residue: {base_resname}")
    else:
        raise ValueError(f"Unknown oscillator type: {osc_type}")


# ============================================================================
# Graph Building for CG-Only Inference
# ============================================================================

def _as_float_tensor(
    x: Any,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Convert to float tensor, handling various input types."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    arr = np.asarray(x, dtype=np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _tuple_reskey(x: Any) -> Optional[Tuple[int, str]]:
    """Convert residue key to (resid, resname) tuple."""
    if x is None:
        return None
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        return (int(x[0]), str(x[1]))
    return None


def build_graph_for_inference(
    osc: Mapping[str, Any],
    *,
    vocab: Any,
    all_oscillators: Optional[List[Dict[str, Any]]] = None,
    current_osc_idx: Optional[int] = None,
    max_sidechain_beads: int = 4,
    fully_connected_edges: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    Build a graph for inference from CG bead positions only.

    Unlike build_graph_from_oscillator, this function:
    - Does NOT require osc['atoms']
    - Determines atom topology from oscillator type and residue name
    - Uses placeholder positions at bead locations (will be replaced by model)

    Args:
        osc: Oscillator dictionary with CG bead positions
        vocab: GraphVocab for encoding names
        all_oscillators: List of all oscillators in the frame (for neighbor beads)
        current_osc_idx: Index of this oscillator in all_oscillators
        max_sidechain_beads: Maximum SC beads to include
        fully_connected_edges: Whether to use fully connected graph
        device: Torch device
        dtype: Torch dtype

    Returns:
        Graph dictionary ready for model input (compatible with collate_graph_samples)
    """
    osc_type = str(osc.get("oscillator_type", osc.get("type", "")))
    residue_name = str(osc.get("residue_name", "UNK"))

    if osc_type not in {"backbone", "sidechain"}:
        raise ValueError(f"Unknown oscillator_type: {osc_type}")

    # ----------------
    # Metadata
    # ----------------
    meta_folder = osc.get("folder")
    meta_frame = osc.get("frame")
    meta_oscillator_index = osc.get("oscillator_index")

    # ----------------
    # Residue BB beads (anchors)
    # ----------------
    residue_keys: List[Tuple[int, str]] = []
    bb_prev_raw: Optional[torch.Tensor] = None

    if osc_type == "backbone":
        bb_curr = _as_float_tensor(
            osc.get("bb_curr", osc.get("bb_curr_pos")), device, dtype
        )
        bb_next = _as_float_tensor(
            osc.get("bb_next", osc.get("bb_next_pos")), device, dtype
        )
        bb_pos = torch.stack([bb_curr, bb_next], dim=0)  # [2, 3]

        k0 = _tuple_reskey(osc.get("bb_curr_key")) or _tuple_reskey(osc.get("residue_key"))
        k1 = _tuple_reskey(osc.get("bb_next_key"))
        k0 = k0 or (-1, "UNK")
        k1 = k1 or (-1, "UNK")
        residue_keys = [k0, k1]
        Nres = 2
    else:
        bb_prev_raw = _as_float_tensor(
            osc.get("bb_prev", osc.get("bb_prev_pos")), device, dtype
        )
        bb_pos = bb_prev_raw.unsqueeze(0)  # [1, 3]

        k0 = _tuple_reskey(osc.get("bb_prev_key")) or _tuple_reskey(osc.get("residue_key"))
        k0 = k0 or (-1, "UNK")
        residue_keys = [k0]
        Nres = 1

    # Optional sidechain beads
    sc_beads: Mapping[str, Any] = osc.get("sc_beads", {}) or {}
    sc1_any = sc_beads.get("SC1")
    sc1_t: Optional[torch.Tensor] = None
    if sc1_any is not None:
        try:
            sc1_t = _as_float_tensor(sc1_any, device, dtype)
        except Exception:
            sc1_t = None

    # Choose anchor for local frame
    if osc_type == "sidechain" and sc1_t is not None and torch.isfinite(sc1_t).all().item():
        assert bb_prev_raw is not None
        bb_pos = sc1_t.unsqueeze(0)
        sc1_pos = bb_prev_raw.unsqueeze(0)
    else:
        if sc1_t is None:
            sc1_pos = None
        else:
            if Nres == 2:
                nan = torch.full_like(sc1_t, float("nan"))
                sc1_pos = torch.stack([sc1_t, nan], dim=0)
            else:
                sc1_pos = sc1_t.unsqueeze(0)

    bb_frames = compute_residue_local_frames(bb_pos, sc1_pos=sc1_pos)  # [Nres, 3, 3]

    # ----------------
    # Atom topology from oscillator type (NO atomistic data needed)
    # ----------------
    base_resname = residue_name.replace("-SC", "")
    atom_topology = get_atom_topology(osc_type, base_resname)

    atom_names: List[str] = []
    atom_names_pdb: List[str] = []
    atom_pos_global: List[torch.Tensor] = []
    atom_res_local: List[int] = []
    atom_group: List[int] = []

    # Create placeholder positions at bead locations
    # These will be replaced by the model during sampling
    for atom_name, res_idx in atom_topology:
        atom_names.append(atom_name)
        atom_names_pdb.append(atom_name.split("_")[0] if "_" in atom_name else atom_name)

        # Placeholder at the appropriate bead position
        if osc_type == "backbone":
            anchor = bb_pos[res_idx]
        else:
            anchor = bb_pos[0]
        atom_pos_global.append(anchor.clone())
        atom_res_local.append(res_idx)
        atom_group.append(vocab.atom_group_id(atom_name))

    Na = len(atom_names)
    if Na == 0:
        raise RuntimeError(f"No atoms defined for oscillator type={osc_type}, residue={residue_name}")

    atom_pos0 = torch.stack(atom_pos_global, dim=0)  # [Na, 3]
    atom_res_local_t = torch.tensor(atom_res_local, device=device, dtype=torch.long)
    atom_group_t = torch.tensor(atom_group, device=device, dtype=torch.long)

    # Convert global → local coordinates
    origin = bb_pos[atom_res_local_t]
    R = bb_frames[atom_res_local_t]

    # Local coords: rotate (pos - origin) by R^T
    diff = atom_pos0 - origin
    x0_local = torch.einsum("nij,nj->ni", R.transpose(-1, -2), diff)

    # ----------------
    # Bead nodes (CG)
    # ----------------
    bead_names: List[str] = []
    bead_pos: List[torch.Tensor] = []
    bead_res_local: List[int] = []

    # Add BB beads (use "BB" for both, differentiate by resid_local)
    if osc_type == "backbone":
        bead_names.extend(["BB", "BB"])
        bead_pos.append(bb_pos[0])
        bead_pos.append(bb_pos[1])
        bead_res_local.extend([0, 1])
    else:
        bead_names.append("BB")
        bead_pos.append(bb_prev_raw if bb_prev_raw is not None else bb_pos[0])
        bead_res_local.append(0)

    # Add SC beads
    sc_keys = sorted([k for k in sc_beads.keys() if k.startswith("SC")])
    for sc_key in sc_keys[:max_sidechain_beads]:
        bead_names.append(sc_key)
        bead_pos.append(_as_float_tensor(sc_beads[sc_key], device, dtype))
        bead_res_local.append(0)

    Nb = len(bead_names)
    bead_pos_t = torch.stack(bead_pos, dim=0) if bead_pos else torch.zeros(0, 3, device=device, dtype=dtype)
    bead_res_local_t = torch.tensor(bead_res_local, device=device, dtype=torch.long)

    # ----------------
    # Node indices
    # ----------------
    num_nodes = Nb + Na
    bead_node_indices = torch.arange(0, Nb, device=device, dtype=torch.long)
    atom_node_indices = torch.arange(Nb, Nb + Na, device=device, dtype=torch.long)

    # ----------------
    # Build edges
    # ----------------
    if fully_connected_edges:
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, device=device, dtype=torch.long)

    # Edge types: 0=bead-bead, 1=bead-atom, 2=atom-bead, 3=atom-atom
    edge_type = []
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        s_is_bead = s < Nb
        d_is_bead = d < Nb
        if s_is_bead and d_is_bead:
            edge_type.append(0)
        elif s_is_bead and not d_is_bead:
            edge_type.append(1)
        elif not s_is_bead and d_is_bead:
            edge_type.append(2)
        else:
            edge_type.append(3)
    edge_type = torch.tensor(edge_type, device=device, dtype=torch.long)

    # ----------------
    # Node features (must match collate expectations)
    # ----------------
    node_type = torch.empty((num_nodes,), device=device, dtype=torch.long)
    node_type[:Nb] = 0  # bead
    node_type[Nb:] = 1  # atom

    node_name = torch.empty((num_nodes,), device=device, dtype=torch.long)
    for i, nm in enumerate(bead_names):
        node_name[i] = vocab.name_id(nm)
    for i, nm in enumerate(atom_names):
        node_name[Nb + i] = vocab.name_id(nm)

    node_resname = torch.full((num_nodes,), vocab.resname_id(residue_name), device=device, dtype=torch.long)

    node_res = torch.empty((num_nodes,), device=device, dtype=torch.long)
    node_res[:Nb] = bead_res_local_t
    node_res[Nb:] = atom_res_local_t

    # node_res_in_frame is local per sample (same as node_res for single oscillator)
    node_res_in_frame = node_res.clone()

    # ----------------
    # Topology indices (empty for inference - not needed for prediction)
    # ----------------
    bond_index = torch.zeros((2, 0), device=device, dtype=torch.long)
    angle_index = torch.zeros((3, 0), device=device, dtype=torch.long)
    dihedral_index = torch.zeros((4, 0), device=device, dtype=torch.long)
    dipole_index = torch.zeros((3, 0), device=device, dtype=torch.long)

    # ----------------
    # Build output dict (compatible with collate_graph_samples)
    # ----------------
    return {
        # Atom data
        "x0_local": x0_local,
        "atom_pos0": atom_pos0,
        "atom_res": atom_res_local_t,
        "atom_group": atom_group_t,

        # Residue data
        "bb_pos": bb_pos,
        "bb_frames": bb_frames,

        # Bead data
        "bead_pos": bead_pos_t,
        "bead_node_indices": bead_node_indices,
        "atom_node_indices": atom_node_indices,

        # Node attributes
        "node_type": node_type,
        "node_name": node_name,
        "node_resname": node_resname,
        "node_res": node_res,
        "node_res_in_frame": node_res_in_frame,

        # Edges
        "edge_index": edge_index,
        "edge_type": edge_type,

        # Counts
        "num_nodes": num_nodes,

        # Topology (empty for inference)
        "bond_index": bond_index,
        "angle_index": angle_index,
        "dihedral_index": dihedral_index,
        "dipole_index": dipole_index,

        # Metadata
        "oscillator_type": osc_type,
        "residue_name": residue_name,
        "meta_folder": meta_folder,
        "meta_frame": meta_frame,
        "meta_oscillator_index": meta_oscillator_index,
        "meta_residue_keys": residue_keys,
        "meta_atom_names": atom_names,
        "meta_atom_names_pdb": atom_names_pdb,
    }


# ============================================================================
# DDIM Sampling (Fast Inference)
# ============================================================================

def ddim_sample_atoms(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Dict[str, Any],
    ddim_steps: int,
    max_atom_radius: float,
    edge_cutoffs: EdgeCutoffs,
    eps: float = 1e-8,
    init: str = "uniform_ball",
    eta: float = 0.0,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDIM sampling in local space.

    Returns:
      x_final_local: [Na,3]
      x_final_global: [Na,3]
    """
    device = batch["x0_local"].device
    Na = batch["x0_local"].shape[0]
    Nn = int(batch["num_nodes"])

    # init x_T
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

    node_atom_group = build_node_atom_group(Nn, batch["atom_node_indices"], batch["atom_group"])
    atom_mask = build_atom_node_mask(Nn, batch["atom_node_indices"])

    T = int(diffusion.timesteps)
    ddim_steps = int(ddim_steps)
    if ddim_steps <= 0:
        raise ValueError("ddim_steps must be > 0")
    timestep_sequence = np.linspace(T - 1, 0, ddim_steps, dtype=int)
    alpha_bars = diffusion.schedule.alpha_bars.to(device=device, dtype=torch.float64)

    for i, t in enumerate(timestep_sequence):
        t_curr = int(t)
        t_next = int(timestep_sequence[i + 1]) if i < len(timestep_sequence) - 1 else -1

        with torch.cuda.amp.autocast(enabled=use_amp):
            xt_global = atoms_local_to_global(xt_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
            node_pos = build_node_pos(
                num_nodes=Nn,
                bead_node_indices=batch["bead_node_indices"],
                bead_pos=batch["bead_pos"],
                atom_node_indices=batch["atom_node_indices"],
                atom_pos=xt_global,
            )
            node_geom = build_node_geom_sph(Nn, batch["atom_node_indices"], xt_local, eps=eps)
            t_node = torch.full((Nn,), t_curr, device=device, dtype=torch.long)

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

        x0_pred_local = clamp_local_atoms(x0_pred_local, max_atom_radius, eps=eps)

        # DDIM update
        a_t = alpha_bars[t_curr]
        if t_next >= 0:
            a_next = alpha_bars[t_next]
        else:
            a_next = torch.tensor(1.0, device=device, dtype=torch.float64)

        sqrt_a_t = torch.sqrt(torch.clamp(a_t, min=eps)).to(dtype=torch.float32)
        sqrt_one_minus_a_t = torch.sqrt(torch.clamp(1.0 - a_t, min=eps)).to(dtype=torch.float32)

        eps_pred = (xt_local - sqrt_a_t * x0_pred_local) / torch.clamp(sqrt_one_minus_a_t, min=eps)

        sqrt_a_next = torch.sqrt(torch.clamp(a_next, min=eps)).to(dtype=torch.float32)
        sqrt_one_minus_a_next = torch.sqrt(torch.clamp(1.0 - a_next, min=eps)).to(dtype=torch.float32)

        if eta != 0.0 and t_next >= 0:
            # stochasticity
            sigma_t = (
                eta
                * torch.sqrt((1 - a_next) / (1 - a_t))
                * torch.sqrt(torch.clamp(1 - a_t / a_next, min=0.0))
            ).to(dtype=torch.float32)
            noise = torch.randn_like(xt_local)
            xt_local = sqrt_a_next * x0_pred_local + sqrt_one_minus_a_next * eps_pred + sigma_t * noise
        else:
            xt_local = sqrt_a_next * x0_pred_local + sqrt_one_minus_a_next * eps_pred

        xt_local = clamp_local_atoms(xt_local, max_atom_radius, eps=eps)

    x_final_local = xt_local
    x_final_global = atoms_local_to_global(x_final_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])
    return x_final_local, x_final_global


# ============================================================================
# CG PDB Parsing
# ============================================================================

def parse_cg_frame_structure(frame_content: str) -> Tuple[List, Dict, List]:
    """Parse single CG frame to extract bead info."""
    lines = frame_content.splitlines()

    bead_sequence = []
    residue_beads = {}

    for line in lines:
        if not line.startswith('ATOM'):
            continue

        resname = line[17:20].strip()
        resseq = int(line[22:26].strip())
        bead_type = line[12:16].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        position = np.array([x, y, z], dtype=np.float32)

        bead_sequence.append((resseq, resname, bead_type, position))

        key = (resseq, resname)
        if key not in residue_beads:
            residue_beads[key] = {}
        residue_beads[key][bead_type] = position

    seen_residues = set()
    ordered_residues = []
    for resseq, resname, _, _ in bead_sequence:
        if (resseq, resname) not in seen_residues:
            ordered_residues.append((resseq, resname))
            seen_residues.add((resseq, resname))

    return bead_sequence, residue_beads, ordered_residues


def extract_oscillators_from_cg_frame(bead_sequence: List) -> List[Dict]:
    """Extract oscillator list from bead sequence."""
    bb_indices = [i for i, (_, _, bead_type, _) in enumerate(bead_sequence)
                  if bead_type == 'BB']

    oscillators = []

    # Backbone oscillators
    for i in range(len(bb_indices) - 1):
        bb_curr_idx = bb_indices[i]
        bb_next_idx = bb_indices[i + 1]

        resseq_curr, resname_curr, _, pos_curr = bead_sequence[bb_curr_idx]
        resseq_next, resname_next, _, pos_next = bead_sequence[bb_next_idx]

        sc_beads_between = {}
        for j in range(bb_curr_idx + 1, bb_next_idx):
            _, _, bead_type, pos = bead_sequence[j]
            if bead_type.startswith('SC'):
                sc_beads_between[bead_type] = pos

        oscillators.append({
            'type': 'backbone',
            'oscillator_type': 'backbone',
            'residue_key': (resseq_curr, resname_curr),
            'residue_name': resname_curr,
            'bead_type': 'BB',
            'residue_index': i,
            'bb_curr_key': (resseq_curr, resname_curr),
            'bb_next_key': (resseq_next, resname_next),
            'bb_curr': pos_curr,
            'bb_next': pos_next,
            'bb_curr_pos': pos_curr,
            'bb_next_pos': pos_next,
            'sc_beads': sc_beads_between,
        })

    # Sidechain oscillators (GLN/ASN with SC1)
    for i, (resseq, resname, bead_type, pos) in enumerate(bead_sequence):
        if resname in ['GLN', 'ASN'] and bead_type == 'SC1':
            bb_prev_idx = None
            for j in range(i - 1, -1, -1):
                if bead_sequence[j][2] == 'BB':
                    bb_prev_idx = j
                    break

            sc_beads_dict = {bead_type: pos}
            for j in range(i + 1, len(bead_sequence)):
                _, _, next_bead_type, next_pos = bead_sequence[j]
                if next_bead_type == 'BB':
                    break
                if next_bead_type.startswith('SC'):
                    sc_beads_dict[next_bead_type] = next_pos

            bb_prev_key = None
            bb_prev_pos = None
            if bb_prev_idx is not None:
                resseq_bb, resname_bb, _, bb_prev_pos = bead_sequence[bb_prev_idx]
                bb_prev_key = (resseq_bb, resname_bb)

            oscillators.append({
                'type': 'sidechain',
                'oscillator_type': 'sidechain',
                'residue_key': (resseq, resname),
                'residue_name': f"{resname}-SC",
                'bead_type': 'SC1',
                'bb_prev_key': bb_prev_key,
                'bb_prev': bb_prev_pos,
                'bb_prev_pos': bb_prev_pos,
                'sc_beads': sc_beads_dict,
            })

    return oscillators


def parse_cg_pdb_frames(
    pdb_file: Path,
    frame_indices: List[int]
) -> List[Dict]:
    """Parse multiple CG PDB frames."""
    with open(pdb_file, 'r') as f:
        content = f.read()

    frame_contents = content.split('ENDMDL')
    frame_contents = [f.strip() + '\nENDMDL' for f in frame_contents if f.strip()]

    frames_data = []

    for frame_idx in frame_indices:
        if frame_idx >= len(frame_contents):
            raise IndexError(f"Requested frame {frame_idx}, but CG PDB has only {len(frame_contents)} frames")

        bead_sequence, residue_beads, ordered_res = parse_cg_frame_structure(
            frame_contents[frame_idx]
        )

        oscillators = extract_oscillators_from_cg_frame(bead_sequence)

        frames_data.append({
            'frame_idx': frame_idx,
            'oscillators': oscillators,
        })

    return frames_data


# ============================================================================
# Hamiltonian Extraction
# ============================================================================

def find_hamiltonian_file(protein_folder: Path, filename: str = 'diagonal_hamiltonian.txt') -> Optional[Path]:
    """Find the Hamiltonian file inside a protein folder."""
    candidates = [
        protein_folder / filename,
        protein_folder / 'hamiltonian' / filename,
        protein_folder / 'hamiltonians' / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    found = list(protein_folder.rglob(filename))
    return found[0] if found else None


def load_hamiltonian_lines(hamiltonian_path: Path) -> List[str]:
    """Load all lines from the Hamiltonian file."""
    with open(hamiltonian_path, 'r') as f:
        return f.readlines()


def extract_hamiltonians_from_lines(ham_lines: List[str], frame_idx: int) -> Optional[np.ndarray]:
    """Extract Hamiltonian values for a given frame."""
    if frame_idx < 0 or frame_idx >= len(ham_lines):
        return None

    ham_line = ham_lines[frame_idx].strip()
    if not ham_line:
        return None

    try:
        if ',' in ham_line:
            ham_vals = np.array([float(val) for val in ham_line.split(',')], dtype=np.float32)
        else:
            ham_vals = np.array(ham_line.split(), dtype=np.float32)
        if ham_vals.size >= 1:
            ham_vals = ham_vals[1:]  # Skip frame number
        return ham_vals
    except Exception:
        return None


# ============================================================================
# Frame Selection
# ============================================================================

def _compute_selection_hash(frame_indices: List[int]) -> str:
    """Stable hash for frame selection."""
    h = hashlib.sha1()
    h.update(','.join(str(i) for i in frame_indices).encode('utf-8'))
    return h.hexdigest()


def select_frame_indices(total_frames: int, config_infer: Dict[str, Any]) -> Tuple[List[int], str]:
    """Select which trajectory frame indices to process."""
    max_frames = config_infer.get('max_frames', None)
    frame_selection = config_infer.get('frame_selection', 'first')
    seed = int(config_infer.get('random_seed', 0))
    sort_random = bool(config_infer.get('sort_random_frames', True))

    if total_frames <= 0:
        return [], _compute_selection_hash([])

    if max_frames is None or max_frames <= 0:
        k = total_frames
    else:
        k = min(int(max_frames), total_frames)

    if frame_selection == 'all':
        frame_indices = list(range(total_frames))
    elif frame_selection == 'first':
        frame_indices = list(range(k))
    elif frame_selection == 'last':
        frame_indices = list(range(max(0, total_frames - k), total_frames))
    elif frame_selection == 'random':
        rng = random.Random(seed)
        frame_indices = rng.sample(range(total_frames), k=k)
        if sort_random:
            frame_indices.sort()
    else:
        raise ValueError(f"Unknown frame_selection: {frame_selection}")

    return frame_indices, _compute_selection_hash(frame_indices)


# ============================================================================
# Dipole and Ramachandran Calculation
# ============================================================================

def compute_dipole_from_atoms(atoms: Dict[str, np.ndarray], osc_type: str) -> Optional[np.ndarray]:
    """Compute dipole vector from atom positions."""
    if osc_type == 'backbone':
        c_pos = atoms.get('C_prev')
        o_pos = atoms.get('O_prev')
        if c_pos is not None and o_pos is not None:
            return (o_pos - c_pos).astype(np.float32)
    elif osc_type == 'sidechain':
        # For GLN: CD -> OE1, for ASN: CG -> OD1
        cd = atoms.get('CD')
        oe1 = atoms.get('OE1')
        if cd is not None and oe1 is not None:
            return (oe1 - cd).astype(np.float32)
        cg = atoms.get('CG')
        od1 = atoms.get('OD1')
        if cg is not None and od1 is not None:
            return (od1 - cg).astype(np.float32)
    return None


def add_dipole_vectors_to_oscillators(
    oscillators: List[Dict],
    source: str = 'predicted_atoms',
    target_field: str = 'predicted_dipole'
):
    """Add dipole vectors to oscillators."""
    for osc in oscillators:
        atoms = osc.get(source, {})
        osc_type = osc.get('oscillator_type', osc.get('type', ''))
        dipole = compute_dipole_from_atoms(atoms, osc_type)
        osc[target_field] = dipole


try:
    from MDAnalysis.lib.distances import calc_dihedrals
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


def add_rama_angles_to_oscillators(
    oscillators: List[Dict],
    source: str = 'predicted_atoms',
    target_field: str = 'predicted_rama_nnfs'
):
    """Add Ramachandran angles to oscillators (backbone only)."""
    if not HAS_MDANALYSIS:
        for osc in oscillators:
            osc[target_field] = {}
        return

    # Group backbone oscillators by frame
    frame_groups = {}
    for osc in oscillators:
        if osc.get('oscillator_type', osc.get('type')) != 'backbone':
            osc[target_field] = {}
            continue
        frame = osc.get('frame', 0)
        if frame not in frame_groups:
            frame_groups[frame] = []
        frame_groups[frame].append(osc)

    for frame, frame_oscs in frame_groups.items():
        # Sort by residue index
        frame_oscs.sort(key=lambda o: o.get('bb_curr_key', (0,))[0])

        # Build coordinate arrays
        for i, osc in enumerate(frame_oscs):
            atoms = osc.get(source, {})
            rama = {}

            # Phi (C-1, N, CA, C): need previous oscillator's C
            if i > 0:
                prev_atoms = frame_oscs[i-1].get(source, {})
                c_prev = prev_atoms.get('C_prev')
                n = atoms.get('N_prev')
                ca = atoms.get('CA_prev')
                c = atoms.get('C_prev')
                if all(x is not None for x in [c_prev, n, ca, c]):
                    try:
                        phi = calc_dihedrals(
                            c_prev.reshape(1, 3),
                            n.reshape(1, 3),
                            ca.reshape(1, 3),
                            c.reshape(1, 3)
                        )[0]
                        rama['phi_N'] = float(np.degrees(phi))
                    except:
                        pass

            # Psi (N, CA, C, N+1): need current atoms
            n = atoms.get('N_prev')
            ca = atoms.get('CA_prev')
            c = atoms.get('C_prev')
            n_next = atoms.get('N_curr')
            if all(x is not None for x in [n, ca, c, n_next]):
                try:
                    psi = calc_dihedrals(
                        n.reshape(1, 3),
                        ca.reshape(1, 3),
                        c.reshape(1, 3),
                        n_next.reshape(1, 3)
                    )[0]
                    rama['psi_N'] = float(np.degrees(psi))
                except:
                    pass

            osc[target_field] = rama


# ============================================================================
# Model Loading
# ============================================================================

def load_config_from_yaml(yaml_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model_and_diffusion(
    config: Dict,
    vocab: Any,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[BackmapGNN, GaussianDiffusion]:
    """Build model and diffusion from config and checkpoint."""
    cfg = Config.from_dict(config)

    model = BackmapGNN(
        num_resnames=vocab.num_resnames,
        num_names=vocab.num_names,
        num_node_types=vocab.num_node_types,
        num_atom_groups=vocab.num_atom_groups,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=0.0,  # No dropout at inference
        time_embed_dim=cfg.model.time_embed_dim,
        pos_embed_dim=cfg.model.pos_embed_dim,
        rbf_num_centers=cfg.model.rbf_num_centers,
        rbf_max_dist=cfg.model.rbf_max_dist,
        max_atom_radius=cfg.model.max_atom_radius,
        eps=cfg.loss.eps,
    ).to(device)

    # Load checkpoint (load_checkpoint takes model as second arg and loads weights into it)
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        map_location=device,
    )
    model.eval()

    schedule = make_schedule(
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        device=device,
    )

    diffusion = GaussianDiffusion(
        schedule=schedule,
        timesteps=cfg.diffusion.timesteps,
        max_radius=cfg.model.max_atom_radius,
        clip_each_step=cfg.diffusion.clip_radius_each_step,
        eps=cfg.loss.eps,
    )

    return model, diffusion


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def run_inference_on_oscillators(
    oscillators: List[Dict],
    model: BackmapGNN,
    diffusion: GaussianDiffusion,
    vocab: Any,
    config: Dict,
    device: torch.device,
    ddim_steps: int = 50,
    use_fp16: bool = True,
) -> List[Dict]:
    """
    Run inference on a batch of oscillators using CG-only graph building.

    Returns oscillators with 'predicted_atoms' field added.
    """
    edge_cutoffs = EdgeCutoffs(
        bead_bead=config['model'].get('bead_edge_cutoff', 10.0),
        atom_any=config['model'].get('atom_edge_cutoff', 10.0),
    )

    graphs = []
    kept_oscillators = []
    na_list = []
    atom_names_list = []

    for local_idx, osc in enumerate(oscillators):
        try:
            # Validate bead positions
            osc_type = osc.get('oscillator_type', osc.get('type', ''))

            if osc_type == 'backbone':
                bb_curr = np.asarray(osc.get('bb_curr', osc.get('bb_curr_pos')), dtype=np.float32)
                bb_next = np.asarray(osc.get('bb_next', osc.get('bb_next_pos')), dtype=np.float32)
                if not np.all(np.isfinite(bb_curr)) or not np.all(np.isfinite(bb_next)):
                    raise ValueError("Non-finite BB positions")
                if float(np.linalg.norm(bb_next - bb_curr)) < 1e-3:
                    raise ValueError("Degenerate BB positions (bb_next ≈ bb_curr)")
            elif osc_type == 'sidechain':
                sc_beads = osc.get('sc_beads', {})
                sc1 = sc_beads.get('SC1') if isinstance(sc_beads, dict) else None
                if sc1 is None:
                    sc1 = osc.get('bb_prev', osc.get('bb_prev_pos'))
                if sc1 is None:
                    raise ValueError("Missing sidechain anchor")
                sc1 = np.asarray(sc1, dtype=np.float32)
                if not np.all(np.isfinite(sc1)):
                    raise ValueError("Non-finite sidechain anchor")

            # Build graph from CG only
            graph = build_graph_from_oscillator(
                osc,
                vocab=vocab,
                all_oscillators=oscillators,
                current_osc_idx=local_idx,
                max_sidechain_beads=config['data']['max_sidechain_beads'],
                fully_connected_edges=config['data']['fully_connected_edges'],
                device=torch.device('cpu'),
                dtype=torch.float32,
            )

            graphs.append(graph)
            kept_oscillators.append(osc)
            na_list.append(int(graph['x0_local'].shape[0]))
            atom_names_list.append(list(graph['meta_atom_names']))

        except Exception as e:
            print(f"Warning: Failed to build graph for oscillator "
                  f"(frame={osc.get('frame')}, idx={osc.get('oscillator_index')}): {e}")
            continue

    if not graphs:
        return []

    # Collate
    batch = collate_graph_samples(graphs)

    # Move to device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    # Sample
    x_final_local, x_final_global = ddim_sample_atoms(
        model=model,
        diffusion=diffusion,
        batch=batch,
        ddim_steps=ddim_steps,
        max_atom_radius=config['model']['max_atom_radius'],
        edge_cutoffs=edge_cutoffs,
        eps=config['loss']['eps'],
        init=config['diffusion'].get('sample_init', 'uniform_ball'),
        eta=0.0,
        use_amp=use_fp16 and device.type == 'cuda',
    )

    # Convert to numpy
    x_final_global_np = x_final_global.detach().cpu().numpy().astype(np.float32)

    # Split predictions by atom counts
    splits = np.cumsum([0] + na_list)

    for k, osc in enumerate(kept_oscillators):
        start = int(splits[k])
        end = int(splits[k + 1])

        pred_coords = x_final_global_np[start:end]
        atom_names = atom_names_list[k]

        atom_dict = {}
        for i, name in enumerate(atom_names):
            if i < len(pred_coords):
                atom_dict[name] = pred_coords[i]

        osc['predicted_atoms'] = atom_dict

    return kept_oscillators


# ============================================================================
# Main Pipeline
# ============================================================================

def process_trajectory(
    config: Dict,
    checkpoint_path: str,
    output_pkl_path: str,
):
    """Main trajectory processing pipeline for CG-only inference."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    vocab_pkl = Path(os.path.dirname(__file__)) / 'tiny.pkl'
    print(f"Loading vocabulary from {vocab_pkl}...")
    vocab = build_default_vocab_from_pickle(vocab_pkl)

    # Build model
    print(f"Loading model from {checkpoint_path}...")
    model, diffusion = build_model_and_diffusion(config, vocab, checkpoint_path, device)

    # Get protein info
    protein_name = config['infer']['protein_name']
    protein_folder = Path(config['infer']['protein_folder'])

    # Get CG trajectory file
    cg_pdb_pattern = config['infer']['cg_pdb_pattern']
    cg_pdb_file = protein_folder / cg_pdb_pattern.format(folder=protein_name)

    if not cg_pdb_file.exists():
        raise FileNotFoundError(f"CG PDB file not found: {cg_pdb_file}")

    print(f"\nProcessing protein: {protein_name}")
    print(f"  CG PDB: {cg_pdb_file}")
    print(f"  Mode: CG-only (no atomistic extraction)")

    # Count frames
    with open(cg_pdb_file, 'r') as f:
        total_frames = f.read().count('ENDMDL')
    print(f"  Total frames: {total_frames}")

    # Select frames
    frame_indices, _ = select_frame_indices(total_frames, config['infer'])
    print(f"  Selected frames: {len(frame_indices)}")

    # Setup output
    output_dir = Path(output_pkl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hamiltonian file (optional)
    ham_filename = config['infer'].get('hamiltonian_file', 'diagonal_hamiltonian.txt')
    hamiltonian_path = find_hamiltonian_file(protein_folder, ham_filename)
    ham_lines = None
    if hamiltonian_path is not None:
        ham_lines = load_hamiltonian_lines(hamiltonian_path)
        print(f"  Hamiltonian file: {hamiltonian_path}")

    # Process in chunks
    chunk_size = config['infer'].get('chunk_size', 100)
    all_oscillators = []

    print(f"\nProcessing in chunks of {chunk_size}...")

    for chunk_start in range(0, len(frame_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(frame_indices))
        chunk_frames = frame_indices[chunk_start:chunk_end]

        print(f"\n{'='*60}")
        print(f"Chunk: frames {chunk_start}-{chunk_end-1}")

        t0 = time.time()

        # Parse CG frames
        print("  Parsing CG data...")
        frames_data = parse_cg_pdb_frames(cg_pdb_file, chunk_frames)

        # Process each frame
        print("  Running inference...")
        chunk_oscillators = []

        for frame_data in frames_data:
            frame_idx = frame_data['frame_idx']
            oscillators = frame_data['oscillators']

            # Add metadata
            for osc_idx, osc in enumerate(oscillators):
                osc['folder'] = protein_name
                osc['frame'] = frame_idx
                osc['oscillator_index'] = osc_idx

                # Hamiltonian
                if ham_lines is not None:
                    ham_vals = extract_hamiltonians_from_lines(ham_lines, frame_idx)
                    if ham_vals is not None and osc_idx < len(ham_vals):
                        osc['hamiltonian'] = float(ham_vals[osc_idx])
                    else:
                        osc['hamiltonian'] = None
                else:
                    osc['hamiltonian'] = None

            # Run inference
            oscillators = run_inference_on_oscillators(
                oscillators,
                model,
                diffusion,
                vocab,
                config,
                device,
                ddim_steps=config['infer'].get('ddim_steps', 50),
                use_fp16=config['infer'].get('use_fp16', True),
            )

            chunk_oscillators.extend(oscillators)

        print(f"  Inference complete: {len(chunk_oscillators)} oscillators")

        # Compute analysis
        if config['infer'].get('compute_analysis', True):
            print("  Computing Rama angles & dipoles...")
            add_rama_angles_to_oscillators(
                chunk_oscillators,
                source='predicted_atoms',
                target_field='predicted_rama_nnfs'
            )
            add_dipole_vectors_to_oscillators(
                chunk_oscillators,
                source='predicted_atoms',
                target_field='predicted_dipole'
            )

        all_oscillators.extend(chunk_oscillators)

        elapsed = time.time() - t0
        print(f"  Chunk time: {elapsed:.1f}s")

    # Organize by amino acid type
    print(f"\n{'='*60}")
    print("Organizing data...")

    amino_acid_baskets = {}
    for osc in all_oscillators:
        resname = osc.get('residue_name', 'UNK')
        if resname not in amino_acid_baskets:
            amino_acid_baskets[resname] = []
        amino_acid_baskets[resname].append(osc)

    # Save
    print(f"Saving to {output_pkl_path}...")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(amino_acid_baskets, f)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"Total oscillators: {len(all_oscillators)}")
    for aa, oscs in sorted(amino_acid_baskets.items()):
        print(f"  {aa}: {len(oscs)}")
    print(f"\nOutput: {output_pkl_path}")

    return amino_acid_baskets


def main():
    parser = argparse.ArgumentParser(
        description="CG-only inference: backmap from coarse-grained trajectory"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to inference config YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output path from config'
    )

    args = parser.parse_args()

    config = load_config_from_yaml(args.config)

    if args.output:
        config['infer']['output'] = args.output

    output_path = config['infer']['output']

    process_trajectory(config, args.checkpoint, output_path)


if __name__ == "__main__":
    main()

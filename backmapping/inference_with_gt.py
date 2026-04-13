#!/usr/bin/env python3
"""
================================================================================
COMPLETE TRAJECTORY-BASED INFERENCE WITH RAMA & DIPOLE ANALYSIS
================================================================================

Features:
  ✓ Fast vectorized trajectory extraction (CG + optional atomistic)
  ✓ DDIM sampling with FP16/AMP for maximum speed
  ✓ Ramachandran angle calculation (predicted + atomistic if enabled)
  ✓ Dipole vector calculation (predicted + atomistic if enabled)
  ✓ Checkpoint/resume capability
  ✓ Chunked processing with progress tracking
  ✓ Compatible output format with existing pickle structure

Usage:
    python infer_from_trajectory_complete.py \\
        --config config_inference_traj.yaml \\
        --checkpoint path/to/checkpoint.pt

The config file controls all parameters including:
  - extract_atomistic: true/false (enables ground truth comparison)
  - compute_analysis: true/false (rama + dipole for predictions)
  - compute_atomistic_analysis: true/false (rama + dipole for atomistic)
  
Output pickle format matches amino_acid_baskets structure with added fields:
  - Each oscillator has 'predicted_atoms': Dict[str, np.ndarray]
  - If compute_analysis=true: 'predicted_rama_nnfs' and 'predicted_dipole'
  - If extract_atomistic=true: 'atoms' field with atomistic coords
  - If compute_atomistic_analysis=true: 'atomistic_dipole' field

================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MDAnalysis as mda
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from MDAnalysis.lib.distances import calc_dihedrals
except ImportError:
    print("ERROR: MDAnalysis required. Install: pip install MDAnalysis")
    sys.exit(1)

# Add backmap to path
_SCRIPT_DIR = Path(__file__).parent
_BACKMAP_DIR = _SCRIPT_DIR / "backmap_diffusion_production_2_2_3"
if _BACKMAP_DIR.exists():
    sys.path.insert(0, str(_BACKMAP_DIR))

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
from backmap.model.sampling import sample_atoms_full
from backmap.utils.checkpoint import load_checkpoint


warnings.filterwarnings("ignore", category=UserWarning)


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
# Configuration
# ============================================================================

def load_config_from_yaml(yaml_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class InferenceCheckpoint:
    """Tracks inference progress for resume capability."""
    
    protein_name: str
    total_frames_in_traj: int
    selected_frames: List[int]
    selected_frames_hash: str
    frames_processed: List[int]
    last_chunk_pos: int
    
    def save(self, checkpoint_path: Path):
        """Save checkpoint to JSON."""
        data = {
            'protein_name': self.protein_name,
            'total_frames_in_traj': self.total_frames_in_traj,
            'selected_frames': self.selected_frames,
            'selected_frames_hash': self.selected_frames_hash,
            'frames_processed': self.frames_processed,
            'last_chunk_pos': self.last_chunk_pos,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, checkpoint_path: Path) -> Optional['InferenceCheckpoint']:
        """Load checkpoint from JSON. Supports legacy checkpoints."""
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Legacy compatibility
        if 'selected_frames' not in data:
            total_frames = int(data.get('total_frames', data.get('total_frames_in_traj', 0)))
            selected_frames = list(range(total_frames))
            selected_hash = _compute_selection_hash(selected_frames)
            frames_processed = data.get('frames_processed', [])
            last_chunk_pos = int(data.get('last_chunk_end', data.get('last_chunk_pos', 0)))
            protein_name = data.get('protein_name', '')
            return cls(
                protein_name=protein_name,
                total_frames_in_traj=total_frames,
                selected_frames=selected_frames,
                selected_frames_hash=selected_hash,
                frames_processed=frames_processed,
                last_chunk_pos=last_chunk_pos,
            )
        
        return cls(
            protein_name=data['protein_name'],
            total_frames_in_traj=int(data.get('total_frames_in_traj', data.get('total_frames', 0))),
            selected_frames=[int(x) for x in data['selected_frames']],
            selected_frames_hash=str(data.get('selected_frames_hash', _compute_selection_hash([int(x) for x in data['selected_frames']]))),
            frames_processed=[int(x) for x in data.get('frames_processed', [])],
            last_chunk_pos=int(data.get('last_chunk_pos', data.get('last_chunk_end', 0))),
        )
    
    @classmethod
    def initialize(
        cls,
        protein_name: str,
        total_frames_in_traj: int,
        selected_frames: List[int],
        selected_frames_hash: str,
    ) -> 'InferenceCheckpoint':
        """Create new checkpoint."""
        return cls(
            protein_name=protein_name,
            total_frames_in_traj=total_frames_in_traj,
            selected_frames=selected_frames,
            selected_frames_hash=selected_frames_hash,
            frames_processed=[],
            last_chunk_pos=0,
        )

# ============================================================================
# CG PDB Parsing (Vectorized)
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
    
    # Get unique ordered residues
    seen_residues = set()
    ordered_residues = []
    for resseq, resname, _, _ in bead_sequence:
        if (resseq, resname) not in seen_residues:
            ordered_residues.append((resseq, resname))
            seen_residues.add((resseq, resname))
    
    return bead_sequence, residue_beads, ordered_residues


def extract_oscillators_from_cg_frame(bead_sequence: List) -> List[Dict]:
    """Extract oscillator list from bead sequence."""
    # Find BB bead indices
    bb_indices = [i for i, (_, _, bead_type, _) in enumerate(bead_sequence)
                  if bead_type == 'BB']
    
    oscillators = []
    
    # Backbone oscillators
    for i in range(len(bb_indices) - 1):
        bb_curr_idx = bb_indices[i]
        bb_next_idx = bb_indices[i + 1]
        
        resseq_curr, resname_curr, _, pos_curr = bead_sequence[bb_curr_idx]
        resseq_next, resname_next, _, pos_next = bead_sequence[bb_next_idx]
        
        # Collect SC beads between BB beads
        sc_beads_between = {}
        for j in range(bb_curr_idx + 1, bb_next_idx):
            _, _, bead_type, pos = bead_sequence[j]
            if bead_type.startswith('SC'):
                sc_beads_between[bead_type] = pos
        
        oscillators.append({
            'type': 'backbone',
            'residue_key': (resseq_curr, resname_curr),
            'bead_type': 'BB',
            'residue_index': i,
            'bb_curr_key': (resseq_curr, resname_curr),
            'bb_next_key': (resseq_next, resname_next),
            'bb_curr_pos': pos_curr,
            'bb_next_pos': pos_next,
            'sc_beads': sc_beads_between,
        })
    
    # Sidechain oscillators (GLN/ASN with SC1)
    for i, (resseq, resname, bead_type, pos) in enumerate(bead_sequence):
        if resname in ['GLN', 'ASN'] and bead_type == 'SC1':
            # Find previous BB
            bb_prev_idx = None
            for j in range(i - 1, -1, -1):
                if bead_sequence[j][2] == 'BB':
                    bb_prev_idx = j
                    break
            
            # Collect subsequent SC beads
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
                'residue_key': (resseq, resname),
                'bead_type': 'SC1',
                'bb_prev_key': bb_prev_key,
                'bb_prev_pos': bb_prev_pos,
                'sc_beads': sc_beads_dict,
            })
    
    return oscillators


def parse_cg_pdb_frames_vectorized(
    pdb_file: Path,
    frame_indices: List[int]
) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """Parse multiple CG PDB frames efficiently."""
    with open(pdb_file, 'r') as f:
        content = f.read()
    
    # Split into frames
    frame_contents = content.split('ENDMDL')
    frame_contents = [f.strip() + '\nENDMDL' for f in frame_contents if f.strip()]
    
    frames_data = []
    ordered_residues = None
    
    for frame_idx in frame_indices:
        if frame_idx >= len(frame_contents):
            raise IndexError(f"Requested frame {frame_idx}, but CG PDB has only {len(frame_contents)} frames")
        
        bead_sequence, residue_beads, ordered_res = parse_cg_frame_structure(
            frame_contents[frame_idx]
        )
        
        if ordered_residues is None:
            ordered_residues = ordered_res
        
        oscillators = extract_oscillators_from_cg_frame(bead_sequence)
        
        frames_data.append({
            'bead_sequence': bead_sequence,
            'residue_beads': residue_beads,
            'oscillators': oscillators,
        })
    
    return frames_data, ordered_residues


# ============================================================================
# Atomistic Extraction (Optional)
# ============================================================================

def extract_atomistic_frames_vectorized(
    universe: mda.Universe,
    frame_indices: List[int]
) -> List[Dict[Tuple[int, str], Dict[str, np.ndarray]]]:
    """Extract atomistic coordinates for multiple frames efficiently."""
    protein = universe.select_atoms('protein')
    frames_data = []
    
    for frame_idx in frame_indices:
        universe.trajectory[frame_idx]
        
        residue_data = {}
        for residue in protein.residues:
            atom_coords = {}
            for atom in residue.atoms:
                atom_coords[atom.name.strip()] = atom.position.astype(np.float32)
            
            key = (residue.resid, residue.resname)
            residue_data[key] = atom_coords
        
        frames_data.append(residue_data)
    
    return frames_data

# ============================================================================
# Hamiltonian Extraction
# ============================================================================

def _compute_selection_hash(frame_indices: List[int]) -> str:
    """Stable hash for a list of frame indices (for checkpoint validation)."""
    h = hashlib.sha1()
    # comma-separated ints is stable and human debuggable
    h.update(','.join(str(i) for i in frame_indices).encode('utf-8'))
    return h.hexdigest()

def find_hamiltonian_file(protein_folder: Path, filename: str = 'diagonal_hamiltonian.txt') -> Optional[Path]:
    """
    Find the Hamiltonian file inside a protein folder.

    Preference order:
      1) <protein_folder>/<filename>
      2) <protein_folder>/hamiltonian/<filename> (common variants)
      3) recursive glob for the filename and common diagonal_hamiltonian*.txt variants

    Returns None if not found.
    """
    candidates = [
        protein_folder / filename,
        protein_folder / 'hamiltonian' / filename,
        protein_folder / 'hamiltonians' / filename,
        protein_folder / 'Hamiltonian' / filename,
        protein_folder / 'Hamiltonians' / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Recursive search (bounded by folder size; typically small per protein)
    patterns = [filename, 'diagonal_hamiltonian.txt', 'diagonal_hamiltonian*.txt', '*hamiltonian*.txt']
    found: List[Path] = []
    for pat in patterns:
        found.extend(sorted(protein_folder.rglob(pat)))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in found:
        if p.is_file() and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[0] if uniq else None

def load_hamiltonian_lines(hamiltonian_path: Path) -> List[str]:
    """Load all lines from the Hamiltonian file."""
    with open(hamiltonian_path, 'r') as f:
        return f.readlines()

def extract_hamiltonians_from_lines(ham_lines: List[str], frame_idx: int) -> Optional[np.ndarray]:
    """
    Extract Hamiltonian values for a given frame from already-loaded lines.

    Expected format per line:
      - either comma-separated or whitespace-separated floats
      - first column is the frame index (skipped)
      - remaining columns are per-oscillator Hamiltonians in oscillator order
    """
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
        # First column is frame number, skip it
        if ham_vals.size >= 1:
            ham_vals = ham_vals[1:]
        return ham_vals
    except Exception:
        return None

def select_frame_indices(total_frames_in_traj: int, config_infer: Dict[str, Any]) -> Tuple[List[int], str]:
    """Select which trajectory frame indices to process."""
    max_frames = config_infer.get('max_frames', None)
    frame_selection = config_infer.get('frame_selection', 'first')  # 'first' | 'random' | 'all'
    seed = int(config_infer.get('random_seed', 0))
    sort_random = bool(config_infer.get('sort_random_frames', True))

    if total_frames_in_traj <= 0:
        return [], _compute_selection_hash([])

    if frame_selection not in ('first', 'random', 'all', 'last'):
        raise ValueError(f"infer.frame_selection must be one of ['first','random','all'], got: {frame_selection}")

    if max_frames is None or max_frames <= 0:
        k = total_frames_in_traj
    else:
        k = min(int(max_frames), total_frames_in_traj)

    if frame_selection in ('all',):
        print('Using all frames')
        frame_indices = list(range(total_frames_in_traj))
    elif frame_selection == 'first':
        print(f'Using first {k} frames')
        frame_indices = list(range(k))
    elif frame_selection == 'last':
        print(f'Using last {k} frames')
        frame_indices = list(range(total_frames_in_traj-k, total_frames_in_traj))
    else:  # random
        rng = random.Random(seed)
        frame_indices = rng.sample(range(total_frames_in_traj), k=k)
        print(f'Using random {k} frames')
        if sort_random:
            frame_indices.sort()

    return frame_indices, _compute_selection_hash(frame_indices)



def add_backbone_atoms(osc: Dict, atomistic_data: Dict):
    """Add atomistic coordinates to backbone oscillator."""
    bb_curr_key = osc['bb_curr_key']
    bb_next_key = osc['bb_next_key']
    
    atoms_curr = atomistic_data.get(bb_curr_key, {})
    atoms_next = atomistic_data.get(bb_next_key, {})
    
    atom_dict = {
        'C_prev': atoms_curr.get('C', np.zeros(3, dtype=np.float32)),
        'O_prev': atoms_curr.get('O', np.zeros(3, dtype=np.float32)),
        'CA_prev': atoms_curr.get('CA', np.zeros(3, dtype=np.float32)),
        'N_prev': atoms_curr.get('N', np.zeros(3, dtype=np.float32)),
        'N_curr': atoms_next.get('N', np.zeros(3, dtype=np.float32)),
        'H_curr': atoms_next.get('H', np.zeros(3, dtype=np.float32)),
        'CA_curr': atoms_next.get('CA', np.zeros(3, dtype=np.float32)),
    }
    
    osc['atoms'] = atom_dict


def add_sidechain_atoms(osc: Dict, atomistic_data: Dict):
    """Add atomistic coordinates to sidechain oscillator."""
    residue_key = osc['residue_key']
    resname = residue_key[1]
    
    atoms = atomistic_data.get(residue_key, {})
    
    sidechain_atoms = {
        'CA': atoms.get('CA', np.zeros(3, dtype=np.float32)),
        'CB': atoms.get('CB', np.zeros(3, dtype=np.float32)),
    }
    
    if resname == 'GLN':
        sidechain_atoms.update({
            'CG': atoms.get('CG', np.zeros(3, dtype=np.float32)),
            'CD': atoms.get('CD', np.zeros(3, dtype=np.float32)),
            'OE1': atoms.get('OE1', np.zeros(3, dtype=np.float32)),
            'NE2': atoms.get('NE2', np.zeros(3, dtype=np.float32)),
            'HE21': atoms.get('HE21', np.zeros(3, dtype=np.float32)),
            'HE22': atoms.get('HE22', np.zeros(3, dtype=np.float32)),
        })
    elif resname == 'ASN':
        sidechain_atoms.update({
            'CG': atoms.get('CG', np.zeros(3, dtype=np.float32)),
            'OD1': atoms.get('OD1', np.zeros(3, dtype=np.float32)),
            'ND2': atoms.get('ND2', np.zeros(3, dtype=np.float32)),
            'HD21': atoms.get('HD21', np.zeros(3, dtype=np.float32)),
            'HD22': atoms.get('HD22', np.zeros(3, dtype=np.float32)),
        })
    
    osc['atoms'] = sidechain_atoms


def _sanitize_vec3(v: Any, fallback: np.ndarray) -> np.ndarray:
    """Return a finite float32 (3,) vector; replace NaN/Inf with fallback."""
    a = np.asarray(v, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(a)):
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return a


def _anchors_for_oscillator(osc: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute robust anchor positions for placeholder atoms / sanitization.

    Returns:
        (anchor_prev, anchor_next, anchor_sc)
    """
    z = np.zeros(3, dtype=np.float32)

    bb_curr = osc.get('bb_curr', osc.get('bb_curr_pos', None))
    bb_next = osc.get('bb_next', osc.get('bb_next_pos', None))

    if bb_curr is None:
        bb_curr = z
    bb_curr = _sanitize_vec3(bb_curr, z)

    if bb_next is None:
        bb_next = bb_curr
    bb_next = _sanitize_vec3(bb_next, bb_curr)

    # Sidechain anchor: prefer SC1, then cg_bead, then bb_prev
    sc1 = None
    sc_beads = osc.get('sc_beads', None)
    if isinstance(sc_beads, dict):
        sc1 = sc_beads.get('SC1', None)
    if sc1 is None:
        sc1 = osc.get('cg_bead', None)
    if sc1 is None:
        sc1 = osc.get('bb_prev', osc.get('bb_prev_pos', None))
    if sc1 is None:
        sc1 = bb_curr
    sc1 = _sanitize_vec3(sc1, bb_curr)

    return bb_curr, bb_next, sc1


def sanitize_oscillator_inplace(osc: Dict) -> None:
    """
    Make oscillator geometry safe for graph building:
      - ensure bead coordinates are finite
      - ensure sc_beads entries are finite
      - ensure atoms are finite (or created later)
    """
    anchor_prev, anchor_next, anchor_sc = _anchors_for_oscillator(osc)

    osc_type = osc.get('oscillator_type', osc.get('type', ''))

    if osc_type == 'backbone':
        osc['bb_curr'] = _sanitize_vec3(osc.get('bb_curr', osc.get('bb_curr_pos', anchor_prev)), anchor_prev)
        osc['bb_next'] = _sanitize_vec3(osc.get('bb_next', osc.get('bb_next_pos', anchor_next)), anchor_next)
        # Keep legacy *_pos fields consistent for downstream code that uses them
        osc['bb_curr_pos'] = osc.get('bb_curr_pos', osc['bb_curr'])
        osc['bb_next_pos'] = osc.get('bb_next_pos', osc['bb_next'])
    elif osc_type == 'sidechain':
        if osc.get('bb_prev', None) is not None or osc.get('bb_prev_pos', None) is not None:
            osc['bb_prev'] = _sanitize_vec3(osc.get('bb_prev', osc.get('bb_prev_pos', anchor_prev)), anchor_prev)
            osc['bb_prev_pos'] = osc.get('bb_prev_pos', osc['bb_prev'])

    # Sanitize sc_beads
    sc_beads = osc.get('sc_beads', None)
    if isinstance(sc_beads, dict):
        clean = {}
        for k, v in sc_beads.items():
            clean[k] = _sanitize_vec3(v, anchor_sc)
        osc['sc_beads'] = clean

    # Sanitize existing atoms (if any)
    if isinstance(osc.get('atoms', None), dict) and osc['atoms']:
        clean_atoms = {}
        for name, v in osc['atoms'].items():
            # Backbone atoms associated with next residue should fall back to anchor_next
            if osc_type == 'backbone' and name in ('N_curr', 'H_curr', 'CA_curr'):
                clean_atoms[name] = _sanitize_vec3(v, anchor_next)
            else:
                clean_atoms[name] = _sanitize_vec3(v, anchor_prev if osc_type == 'backbone' else anchor_sc)
        osc['atoms'] = clean_atoms


def _ensure_atoms_for_inference(osc: Dict):
    """
    Ensure oscillator has a valid 'atoms' field for graph building.

    IMPORTANT: Atomistic extraction is OPTIONAL for comparison/metrics, but NOT
    required for prediction. We always construct a full placeholder atom set
    (with finite, non-zero coordinates anchored on the relevant bead(s)) when
    ground-truth atoms are unavailable.

    This guarantees:
      - Na (atom count) is correct for the oscillator topology
      - no NaNs/Infs are introduced
      - drop_zero_atoms won't delete everything due to all-zero placeholders
    """
    sanitize_oscillator_inplace(osc)
    if 'atoms' in osc and isinstance(osc['atoms'], dict) and osc['atoms']:
        return

    osc_type = osc.get('oscillator_type', osc.get('type', ''))
    anchor_prev, anchor_next, anchor_sc = _anchors_for_oscillator(osc)

    if osc_type == 'backbone':
        # Place prev-res atoms at bb_curr, and next-res atoms at bb_next
        osc['atoms'] = {
            'C_prev': anchor_prev.copy(),
            'O_prev': anchor_prev.copy(),
            'CA_prev': anchor_prev.copy(),
            'N_prev': anchor_prev.copy(),
            'N_curr': anchor_next.copy(),
            'H_curr': anchor_next.copy(),
            'CA_curr': anchor_next.copy(),
        }
        return

    if osc_type == 'sidechain':
        # Default minimal sidechain scaffold
        res_key = osc.get('residue_key', (0, 'UNK'))
        resname = res_key[1] if isinstance(res_key, (tuple, list)) and len(res_key) > 1 else 'UNK'

        atoms = {
            'CA': anchor_sc.copy(),
            'CB': anchor_sc.copy(),
        }
        if resname == 'GLN':
            atoms.update({
                'CG': anchor_sc.copy(),
                'CD': anchor_sc.copy(),
                'OE1': anchor_sc.copy(),
                'NE2': anchor_sc.copy(),
                'HE21': anchor_sc.copy(),
                'HE22': anchor_sc.copy(),
            })
        elif resname == 'ASN':
            atoms.update({
                'CG': anchor_sc.copy(),
                'OD1': anchor_sc.copy(),
                'ND2': anchor_sc.copy(),
                'HD21': anchor_sc.copy(),
                'HD22': anchor_sc.copy(),
            })
        osc['atoms'] = atoms
        return

    # Fallback: unknown oscillator type
    osc['atoms'] = {}


# ============================================================================
# Ramachandran Angle Calculation
# ============================================================================

def compute_phi_psi_from_backbone_oscillators(
    oscillators: List[Dict],
    source: str = 'atoms'  # 'atoms' or 'predicted_atoms'
) -> Tuple[Dict[int, float], Dict[int, float], Optional[int], Optional[int]]:
    """
    Compute φ/ψ angles from backbone oscillators using MDAnalysis-consistent dihedrals.
    
    Args:
        oscillators: List of backbone oscillators
        source: Which atom coordinates to use ('atoms' for atomistic, 'predicted_atoms' for predicted)
    
    Returns:
        phi_by_resid, psi_by_resid, first_resid, last_resid
    """
    # Build mapping from resid to oscillator
    bond_by_resid = {}
    all_resids = set()
    
    for osc in oscillators:
        if osc['type'] != 'backbone':
            continue
        
        bb_curr_key = osc.get('bb_curr_key')
        bb_next_key = osc.get('bb_next_key')
        
        if not bb_curr_key or not bb_next_key:
            continue
        
        i = int(bb_curr_key[0])
        j = int(bb_next_key[0])
        
        bond_by_resid[i] = osc
        all_resids.add(i)
        all_resids.add(j)
    
    if not all_resids:
        return {}, {}, None, None
    
    all_resids = sorted(all_resids)
    first_resid = all_resids[0]
    last_resid = all_resids[-1]
    
    phi_by_resid = {}
    psi_by_resid = {}
    
    # Internal residues only
    internal_resids = [r for r in all_resids if first_resid < r < last_resid]
    
    # Compute φ(i) = dihedral(C_{i-1}, N_i, CA_i, C_i)
    phi_resids = []
    phi_coords1, phi_coords2, phi_coords3, phi_coords4 = [], [], [], []
    
    for i in internal_resids:
        osc_i = bond_by_resid.get(i)
        osc_im1 = bond_by_resid.get(i - 1)
        
        if osc_i is None or osc_im1 is None:
            continue
        
        atoms_i = osc_i.get(source, {})
        atoms_im1 = osc_im1.get(source, {})
        
        try:
            C_im1 = np.asarray(atoms_im1['C_prev'], dtype=np.float64)
            N_i = np.asarray(atoms_i['N_prev'], dtype=np.float64)
            CA_i = np.asarray(atoms_i['CA_prev'], dtype=np.float64)
            C_i = np.asarray(atoms_i['C_prev'], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        
        # Skip if any coordinate is zero
        if (np.allclose(C_im1, 0.0) or np.allclose(N_i, 0.0) or 
            np.allclose(CA_i, 0.0) or np.allclose(C_i, 0.0)):
            continue
        
        phi_resids.append(i)
        phi_coords1.append(C_im1)
        phi_coords2.append(N_i)
        phi_coords3.append(CA_i)
        phi_coords4.append(C_i)
    
    if phi_resids:
        phi_angles = np.rad2deg(
            calc_dihedrals(
                np.stack(phi_coords1),
                np.stack(phi_coords2),
                np.stack(phi_coords3),
                np.stack(phi_coords4),
            )
        )
        for resid, angle in zip(phi_resids, phi_angles):
            phi_by_resid[resid] = float(angle)
    
    # Compute ψ(i) = dihedral(N_i, CA_i, C_i, N_{i+1})
    psi_resids = []
    psi_coords1, psi_coords2, psi_coords3, psi_coords4 = [], [], [], []
    
    for i in internal_resids:
        osc_i = bond_by_resid.get(i)
        
        if osc_i is None:
            continue
        
        atoms_i = osc_i.get(source, {})
        
        try:
            N_i = np.asarray(atoms_i['N_prev'], dtype=np.float64)
            CA_i = np.asarray(atoms_i['CA_prev'], dtype=np.float64)
            C_i = np.asarray(atoms_i['C_prev'], dtype=np.float64)
            N_ip1 = np.asarray(atoms_i['N_curr'], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        
        # Skip if any coordinate is zero
        if (np.allclose(N_i, 0.0) or np.allclose(CA_i, 0.0) or 
            np.allclose(C_i, 0.0) or np.allclose(N_ip1, 0.0)):
            continue
        
        psi_resids.append(i)
        psi_coords1.append(N_i)
        psi_coords2.append(CA_i)
        psi_coords3.append(C_i)
        psi_coords4.append(N_ip1)
    
    if psi_resids:
        psi_angles = np.rad2deg(
            calc_dihedrals(
                np.stack(psi_coords1),
                np.stack(psi_coords2),
                np.stack(psi_coords3),
                np.stack(psi_coords4),
            )
        )
        for resid, angle in zip(psi_resids, psi_angles):
            psi_by_resid[resid] = float(angle)
    
    # Apply boundary convention: φ=ψ=0.0 for first and last residue
    if first_resid is not None:
        phi_by_resid[first_resid] = 0.0
        psi_by_resid[first_resid] = 0.0
    if last_resid is not None and last_resid != first_resid:
        phi_by_resid[last_resid] = 0.0
        psi_by_resid[last_resid] = 0.0
    
    return phi_by_resid, psi_by_resid, first_resid, last_resid


def compute_nnfs_angles_for_oscillator(
    osc: Dict,
    phi_by_resid: Dict[int, float],
    psi_by_resid: Dict[int, float]
) -> Dict[str, Optional[float]]:
    """Compute NNFS angles for one oscillator."""
    res_key = osc.get('residue_key')
    if not res_key:
        return {key: None for key in ['phi_N', 'psi_N', 'phi_C', 'psi_C']}
    
    resid = int(res_key[0])
    
    phi_i = phi_by_resid.get(resid)
    psi_i = psi_by_resid.get(resid)
    phi_prev = phi_by_resid.get(resid - 1)
    psi_prev = psi_by_resid.get(resid - 1)
    phi_next = phi_by_resid.get(resid + 1)
    psi_next = psi_by_resid.get(resid + 1)
    
    return {
        'phi_N': phi_i,
        'psi_N': psi_prev,
        'phi_C': phi_next,
        'psi_C': psi_i,
    }


def add_rama_angles_to_oscillators(
    oscillators: List[Dict],
    source: str = 'atoms',
    target_field: str = 'rama_nnfs'
):
    """Add Ramachandran NNFS angles to all oscillators."""
    backbone_oscs = [o for o in oscillators if o['type'] == 'backbone']

    frames = set([osc['frame'] for osc in backbone_oscs])
    for t in frames:
        oscs_t = [o for o in backbone_oscs if o['frame'] == t]
        
        # Compute φ/ψ
        phi_by_resid, psi_by_resid, first_resid, last_resid = \
            compute_phi_psi_from_backbone_oscillators(oscs_t, source=source)
        
        # Add NNFS angles to each oscillator
        for osc in oscs_t:
            osc[target_field] = compute_nnfs_angles_for_oscillator(osc, phi_by_resid, psi_by_resid)
    
    sidechain_oscs = [o for o in oscillators if o['type'] != 'backbone']
    for osc in sidechain_oscs:
        osc[target_field] = {'phi_N': None, 'psi_N': None, 'phi_C': None, 'psi_C': None}


# ============================================================================
# Dipole Vector Calculation
# ============================================================================

def compute_dipole_vector(atoms: Dict[str, np.ndarray], osc_type: str, resname: str) -> Optional[np.ndarray]:
    """
    Compute dipole unit vector for an oscillator.
    
    For backbone: C=O bond direction (O - C)
    For sidechain GLN: CD=OE1 bond direction (OE1 - CD)
    For sidechain ASN: CG=OD1 bond direction (OD1 - CG)
    
    Returns:
        Unit vector as numpy array, or None if atoms missing
    """
    if osc_type == 'backbone':
        C = atoms.get('C_prev')
        O = atoms.get('O_prev')
        N = atoms.get('N_curr')
        
        if C is None or O is None or N is None:
            return None
        
        C = np.asarray(C, dtype=np.float64)
        O = np.asarray(O, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        
        # Skip if zero
        if np.allclose(C, 0.0) or np.allclose(O, 0.0) or np.allclose(N, 0.0):
            return None
        
        # C=O dipole direction
        CO = O - C
        
        # Skip if CO vector is zero 
        if np.linalg.norm(CO) < 1e-6:
            return None
            
        # CN vector (if N is available)
        if N is not None:
            CN = N - C
            # Check if CN vector is meaningful
            if np.linalg.norm(CN) < 1e-6:
                CN = None
        else:
            CN = None
        
    elif osc_type == 'sidechain':
        if resname == 'GLN' or resname == 'GLN-SC':
            C = atoms.get('CD')
            O = atoms.get('OE1')
            N = atoms.get('NE2')
        elif resname == 'ASN' or resname == 'ASN-SC':
            C = atoms.get('CG')
            O = atoms.get('OD1')
            N = atoms.get('ND2')
        else:
            return None

        if C is None or O is None:
            return None

        C = np.asarray(C, dtype=np.float64)
        O = np.asarray(O, dtype=np.float64)

        # CO vector
        CO = O - C

        # Skip if CO vector is zero (placeholder atoms)
        if np.linalg.norm(CO) < 1e-6:
            return None

        # CN vector (if N is available)
        if N is not None:
            N = np.asarray(N, dtype=np.float64)
            CN = N - C
            # Check if CN vector is meaningful
            if np.linalg.norm(CN) < 1e-6:
                CN = None
        else:
            CN = None

    else:
        return None
    
    # Normalize CO vector 
    CO_norm = np.linalg.norm(CO)
    CO_unit = CO / CO_norm

    # Compute s vector: s = 0.665*CO + 0.258*CN
    if CN is not None:
        s = 0.665 * CO + 0.258 * CN
    else:
        # Fallback if N is missing: s = CO
        s = CO

    # Compute Torii dipole using the formula
    # μ = 0.276 * (s - ((CO·s) + √(|s|² - (CO·s)²)/tan(10°)) * CO)

    s_norm_sq = np.dot(s, s)
    CO_dot_s = np.dot(CO_unit, s)

    # tan(10°) ≈ 0.17633
    tan_10_deg = np.tan(np.deg2rad(10.0))

    # Compute the square root term
    sqrt_term_arg = s_norm_sq - CO_dot_s**2
    if sqrt_term_arg < 0:
        sqrt_term_arg = 0.0  # Numerical safety
    sqrt_term = np.sqrt(sqrt_term_arg)

    # Compute the coefficient
    coeff = CO_dot_s + sqrt_term / tan_10_deg

    # Compute dipole: μ = 0.276 * (s - coeff * CO_unit)
    dipole = 0.276 * (s - coeff * CO_unit)

    # Return dipole in Debye (already in correct units due to 0.276 prefactor)
    return dipole.astype(np.float32)

def add_dipole_vectors_to_oscillators(
    oscillators: List[Dict],
    source: str = 'atoms',
    target_field: str = 'dipole'
):
    """Add dipole unit vectors to all oscillators."""
    for osc in oscillators:
        atoms = osc.get(source, {})
        osc_type = osc['type']
        resname = osc.get('residue_name', osc['residue_key'][1])
        
        dipole = compute_dipole_vector(atoms, osc_type, resname)
        osc[target_field] = dipole


# ============================================================================
# Model Setup
# ============================================================================

def build_model_and_diffusion(
    config: Dict,
    vocab: Any,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[BackmapGNN, GaussianDiffusion]:
    """Build model and diffusion from config."""
    cfg = Config(config)
    
    # Build model
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
    
    # Load checkpoint
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        map_location=device,
    )
    
    model.eval()
    
    # Build diffusion
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
    use_ddim: bool = True,
    ddim_steps: int = 50,
    use_fp16: bool = True,
) -> List[Dict]:
    """
    Run inference on a batch of oscillators.
    
    Returns oscillators with 'predicted_atoms' field added.
    """
    # Build edge cutoffs
    edge_cutoffs = EdgeCutoffs(
        bead_bead=config['model'].get('bead_edge_cutoff', 10.0),
        atom_any=config['model'].get('atom_edge_cutoff', 10.0),
    )
    
    # Build graphs on CPU (matches training)
    graphs = []
    kept_oscillators = []
    na_list = []  # Atom counts per oscillator
    atom_names_list = []  # Atom names per oscillator
    
    for local_idx, osc in enumerate(oscillators):
        try:
            # Make oscillator safe/complete for prediction (atomistic not required)
            _ensure_atoms_for_inference(osc)

            # Skip degenerate anchors (prevents NaNs in local frames / direction features)
            osc_type = osc.get('oscillator_type', osc.get('type', ''))
            if osc_type == 'backbone':
                bb_curr = np.asarray(osc.get('bb_curr', osc.get('bb_curr_pos', np.zeros(3))), dtype=np.float32).reshape(3)
                bb_next = np.asarray(osc.get('bb_next', osc.get('bb_next_pos', bb_curr)), dtype=np.float32).reshape(3)
                if not np.all(np.isfinite(bb_curr)) or not np.all(np.isfinite(bb_next)):
                    raise ValueError("Non-finite BB anchors")
                if float(np.linalg.norm(bb_next - bb_curr)) < 1.0e-3:
                    raise ValueError("Degenerate BB anchors (bb_next≈bb_curr)")
            elif osc_type == 'sidechain':
                sc1 = None
                if isinstance(osc.get('sc_beads', None), dict):
                    sc1 = osc['sc_beads'].get('SC1', None)
                if sc1 is None:
                    sc1 = osc.get('cg_bead', None)
                if sc1 is None:
                    sc1 = osc.get('bb_prev', osc.get('bb_prev_pos', None))
                if sc1 is None:
                    raise ValueError("Missing sidechain anchor (SC1/cg_bead/bb_prev)")
                sc1 = np.asarray(sc1, dtype=np.float32).reshape(3)
                if not np.all(np.isfinite(sc1)):
                    raise ValueError("Non-finite sidechain anchor")

            graph = build_graph_from_oscillator(
                osc,
                vocab=vocab,
                all_oscillators=oscillators,
                current_osc_idx=local_idx,
                drop_zero_atoms=config['data']['drop_zero_atoms'],
                max_sidechain_beads=config['data']['max_sidechain_beads'],
                fully_connected_edges=config['data']['fully_connected_edges'],
                device=torch.device('cpu'),  # Always build graphs on CPU
                dtype=torch.float32,
            )

            graphs.append(graph)
            kept_oscillators.append(osc)
            na_list.append(int(graph['x0_local'].shape[0]))
            atom_names_list.append([str(x) for x in graph.get('meta_atom_names', [])])

        except Exception as e:
            print(f"Warning: Failed to build graph for oscillator (frame={osc.get('frame')}, idx={osc.get('oscillator_index')}): {e}")
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
    if use_ddim:
        x_final_local, x_final_global = ddim_sample_atoms(
            model=model,
            diffusion=diffusion,
            batch=batch,
            ddim_steps=ddim_steps,
            max_atom_radius=config['model']['max_atom_radius'],
            edge_cutoffs=edge_cutoffs,
            eps=config['loss']['eps'],
            init=config['diffusion']['sample_init'],
            eta=0.0,
            use_amp=use_fp16 and device.type == 'cuda',
        )
    else:
        x_final_local, x_final_global = sample_atoms_full(
            model=model,
            diffusion=diffusion,
            batch=batch,
            timesteps=diffusion.timesteps,
            max_atom_radius=config['model']['max_atom_radius'],
            edge_cutoffs=edge_cutoffs,
            eps=config['loss']['eps'],
            init=config['diffusion']['sample_init'],
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
        
        # Build atom dictionary
        atom_dict = {}
        for i, name in enumerate(atom_names):
            if i < len(pred_coords):
                atom_dict[name] = pred_coords[i]
        
        osc['predicted_atoms'] = atom_dict
    
    return kept_oscillators


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def process_trajectory(
    config: Dict,
    checkpoint_path: str,
    output_pkl_path: str,
):
    """Main trajectory processing pipeline."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary from pickle
    vocab_pkl = Path(os.path.dirname(__file__)) / 'tiny.pkl'
    print(f"Loading vocabulary from {vocab_pkl}...")
    vocab = build_default_vocab_from_pickle(vocab_pkl)
    
    # Build model
    print(f"Loading model from {checkpoint_path}...")
    model, diffusion = build_model_and_diffusion(config, vocab, checkpoint_path, device)
    
    # Get protein info
    protein_name = config['infer']['protein_name']
    protein_folder = Path(config['infer']['protein_folder'])
    
    # Get trajectory files
    extract_atomistic = config['infer'].get('extract_atomistic', False)
    if extract_atomistic:
        if 'xtc_filename' in config['infer'].keys():
            xtc_file = protein_folder / config['infer']['xtc_filename']
        else:
            raise FileNotFoundError(f"Missing xtc file for {protein_name}")
        if not xtc_file.exists():
            raise FileNotFoundError(f"Missing xtc file for {protein_name}")
        if 'tpr_filename' in config['infer'].keys():
            tpr_file = protein_folder / config['infer']['tpr_filename']
        else:
            raise FileNotFoundError(f"Missing tpr file for {protein_name}")
        if not tpr_file.exists():
            raise FileNotFoundError(f"Missing tpr file for {protein_name}")
    else:
        xtc_file = None
        tpr_file = None
    cg_pdb_pattern = config['infer']['cg_pdb_pattern']
    cg_pdb_file = protein_folder / cg_pdb_pattern.format(folder=protein_name)
    if not cg_pdb_file.exists():
        raise FileNotFoundError(f"Missing CG trajectory file for {protein_name}")
    
    print(f"\nProcessing protein: {protein_name}")
    print(f"  XTC: {xtc_file}")
    print(f"  TPR: {tpr_file}")
    print(f"  CG PDB: {cg_pdb_file}")
    
    # Count frames
    with open(cg_pdb_file, 'r') as f:
        total_frames_in_traj = f.read().count('ENDMDL')
    print(f"  Total frames in trajectory: {total_frames_in_traj}")
    
    # Determine which frames to process (first N / random N / all)
    frame_indices, frame_indices_hash = select_frame_indices(total_frames_in_traj, config['infer'])
    frame_selection = config['infer'].get('frame_selection', 'first')
    print(f"  Frame selection: {frame_selection}")
    print(f"  Selected frames: {len(frame_indices)}")
    if len(frame_indices) > 0:
        preview = frame_indices[:10]
        print(f"  First selected indices: {preview}{' ...' if len(frame_indices) > 10 else ''}")
    
    # Setup checkpoint
    checkpoint_dir = Path(output_pkl_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{protein_name}_checkpoint.json"
    
    checkpoint = InferenceCheckpoint.load(checkpoint_file)
    if checkpoint is not None:
        # Validate that the checkpoint matches the current frame selection
        if checkpoint.protein_name != protein_name:
            raise ValueError(
                f"Checkpoint protein_name={checkpoint.protein_name} does not match current protein_name={protein_name}"
            )
        if checkpoint.selected_frames_hash != frame_indices_hash or checkpoint.selected_frames != frame_indices:
            raise ValueError(
                "Checkpoint frame selection does not match current selection. "
                "Delete the checkpoint file (and optionally the output pkl) and rerun."
            )
        print(f"\n✓ Resuming from checkpoint: {len(checkpoint.frames_processed)} frames done")
        start_pos = checkpoint.last_chunk_pos
    else:
        checkpoint = InferenceCheckpoint.initialize(
            protein_name=protein_name,
            total_frames_in_traj=total_frames_in_traj,
            selected_frames=frame_indices,
            selected_frames_hash=frame_indices_hash,
        )
        start_pos = 0
    
    # Setup atomistic universe if needed
    u = None
    if extract_atomistic:
        print("\n✓ Extracting atomistic data (SLOW)")
        u = mda.Universe(str(tpr_file), str(xtc_file))
    
    # Hamiltonian file setup
    ham_filename = config['infer'].get('hamiltonian_file', 'diagonal_hamiltonian.txt')
    require_hamiltonians = bool(config['infer'].get('require_hamiltonians', False))
    strict_ham_validation = bool(config['infer'].get('strict_hamiltonian_validation', True))
    hamiltonian_path = find_hamiltonian_file(protein_folder, ham_filename)
    ham_lines: Optional[List[str]] = None
    if hamiltonian_path is None:
        msg = f"  Hamiltonian file not found (looked for '{ham_filename}' in {protein_folder})"
        if require_hamiltonians:
            raise FileNotFoundError(msg)
        print(msg + "; proceeding with hamiltonian=None")
    else:
        ham_lines = load_hamiltonian_lines(hamiltonian_path)
        print(f"  Hamiltonian file: {hamiltonian_path} ({len(ham_lines)} lines)")
    
    # Process in chunks
    chunk_size = config['infer']['chunk_size']
    save_intermediate = bool(config['infer'].get('save_intermediate', True))
    all_oscillators_data: List[Dict[str, Any]] = []
    
    # If resuming, load already-saved oscillators so final output is complete
    if start_pos > 0 and Path(output_pkl_path).exists():
        try:
            with open(output_pkl_path, 'rb') as f:
                existing = pickle.load(f)
            # Flatten existing amino_acid_baskets dict into a list
            for _, oscs in existing.items():
                if isinstance(oscs, list):
                    all_oscillators_data.extend(oscs)
            print(f"  ✓ Loaded {len(all_oscillators_data)} oscillators from existing output for resume")
        except Exception as e:
            print(f"  ⚠ Could not load existing output pkl for resume ({type(e).__name__}: {e}). Continuing without it.")
            all_oscillators_data = []

    
    print(f"\nProcessing in chunks of {chunk_size}...")
    
    for chunk_start in range(start_pos, len(frame_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(frame_indices))
        chunk_frames = frame_indices[chunk_start:chunk_end]
        
        print(f"\n{'='*80}")
        print(f"Chunk {chunk_start//chunk_size + 1}: Frames {chunk_start}-{chunk_end-1}")
        print(f"{'='*80}")
        
        t0 = time.time()
        
        # Extract CG data
        print("  1/5 Extracting CG data...")
        cg_frames, ordered_residues = parse_cg_pdb_frames_vectorized(
            cg_pdb_file, chunk_frames
        )
        print(f"      ✓ Extracted {len(cg_frames)} frames")
        
        # Extract atomistic if enabled
        atomistic_frames = None
        if extract_atomistic:
            print("  2/5 Extracting atomistic data...")
            atomistic_frames = extract_atomistic_frames_vectorized(u, chunk_frames)
            print(f"      ✓ Extracted {len(atomistic_frames)} frames")
        else:
            print("  2/5 Skipping atomistic extraction (disabled)")
        
        # Process each frame
        print("  3/5 Running model inference...")
        chunk_oscillators = []
        
        for frame_num, (cg_frame, frame_idx) in enumerate(zip(cg_frames, chunk_frames)):
            oscillators = cg_frame['oscillators']
            
            # Add atomistic if available
            if atomistic_frames is not None:
                atomistic_data = atomistic_frames[frame_num]
                for osc in oscillators:
                    if osc['type'] == 'backbone':
                        add_backbone_atoms(osc, atomistic_data)
                    elif osc['type'] == 'sidechain':
                        add_sidechain_atoms(osc, atomistic_data)
            
            # Extract Hamiltonians for this frame (optional)
            ham_vals: Optional[np.ndarray] = None
            if ham_lines is not None:
                ham_vals = extract_hamiltonians_from_lines(ham_lines, frame_idx)
            
            if ham_vals is None and require_hamiltonians:
                raise ValueError(f"Frame {frame_idx}: Hamiltonian data missing/unreadable")
            if ham_vals is not None and strict_ham_validation and len(ham_vals) != len(oscillators):
                raise ValueError(
                    f"Frame {frame_idx}: oscillator count {len(oscillators)} != Hamiltonian count {len(ham_vals)} (excluding frame column)"
                )

            # Add metadata
            for osc_idx, osc in enumerate(oscillators):
                osc['folder'] = protein_name
                osc['frame'] = frame_idx
                osc['oscillator_index'] = osc_idx
                osc['oscillator_type'] = osc['type']
                # Hamiltonian (one value per oscillator, aligned by oscillator_index)
                if ham_vals is not None and osc_idx < len(ham_vals):
                    osc['hamiltonian'] = float(ham_vals[osc_idx])
                else:
                    osc['hamiltonian'] = None
                
                
                # Ensure bb_curr/bb_next fields (some code expects these names)
                if osc['type'] == 'backbone':
                    if 'bb_curr' not in osc and 'bb_curr_pos' in osc:
                        osc['bb_curr'] = osc['bb_curr_pos']
                    if 'bb_next' not in osc and 'bb_next_pos' in osc:
                        osc['bb_next'] = osc['bb_next_pos']
                elif osc['type'] == 'sidechain':
                    if 'bb_prev' not in osc and 'bb_prev_pos' in osc:
                        osc['bb_prev'] = osc['bb_prev_pos']
                
                # residue_name field
                if osc['type'] == 'backbone':
                    osc['residue_name'] = osc['residue_key'][1]
                else:
                    osc['residue_name'] = f"{osc['residue_key'][1]}-SC"
                
                # Legacy fields
                osc['cg_bead'] = (osc.get('bb_curr_pos') if osc['type'] == 'backbone' 
                                 else osc.get('sc_beads', {}).get('SC1'))
                osc['cg_bead_type'] = 'BB' if osc['type'] == 'backbone' else 'SC1'
                
                # Ensure atoms field exists for graph building
                _ensure_atoms_for_inference(osc)
            
            # Run inference
            oscillators = run_inference_on_oscillators(
                oscillators,
                model,
                diffusion,
                vocab,
                config,
                device,
                use_ddim=True,
                ddim_steps=config['infer'].get('ddim_steps', 50),
                use_fp16=config['infer'].get('use_fp16', True),
            )
            
            chunk_oscillators.extend(oscillators)
        
        print(f"      ✓ Inference complete: {len(chunk_oscillators)} oscillators")
        
        # Compute analysis
        compute_analysis = config['infer'].get('compute_analysis', True)
        compute_atomistic_analysis = config['infer'].get('compute_atomistic_analysis', False)
        
        if compute_analysis:
            print("  4/5 Computing predicted Rama angles & dipoles...")
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
            print("      ✓ Analysis complete")
        else:
            print("  4/5 Skipping predicted analysis (disabled)")
        
        if extract_atomistic and compute_atomistic_analysis:
            print("  5/5 Computing atomistic Rama angles & dipoles...")
            add_rama_angles_to_oscillators(
                chunk_oscillators,
                source='atoms',
                target_field='rama_nnfs'
            )
            add_dipole_vectors_to_oscillators(
                chunk_oscillators,
                source='atoms',
                target_field='atomistic_dipole'
            )
            print("      ✓ Atomistic analysis complete")
        else:
            print("  5/5 Skipping atomistic analysis")
        
        # Add to global list
        all_oscillators_data.extend(chunk_oscillators)
        
        # Optional: save intermediate output (makes resume truly safe)
        if save_intermediate:
            amino_acid_baskets_tmp: Dict[str, List[Dict[str, Any]]] = {}
            for osc in all_oscillators_data:
                resname = osc.get('residue_name', 'UNK')
                if resname not in amino_acid_baskets_tmp:
                    amino_acid_baskets_tmp[resname] = []
                amino_acid_baskets_tmp[resname].append(osc)
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(amino_acid_baskets_tmp, f)
            print(f"      ✓ Intermediate output saved: {output_pkl_path}")

        # Update checkpoint
        checkpoint.frames_processed.extend(chunk_frames)
        checkpoint.last_chunk_pos = chunk_end
        checkpoint.save(checkpoint_file)
        
        elapsed = time.time() - t0
        print(f"\n  Chunk completed in {elapsed:.1f}s ({elapsed/len(chunk_frames):.2f}s/frame)")
    
    # Organize by amino acid type (same format as original pickle)
    print(f"\n{'='*80}")
    print("Organizing data by amino acid type...")
    amino_acid_baskets = {}
    
    for osc in all_oscillators_data:
        resname = osc['residue_name']
        
        if resname not in amino_acid_baskets:
            amino_acid_baskets[resname] = []
        
        amino_acid_baskets[resname].append(osc)
    
    # Save pickle
    print(f"Saving to {output_pkl_path}...")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(amino_acid_baskets, f)
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Total oscillators: {len(all_oscillators_data)}")
    print(f"Amino acid types: {len(amino_acid_baskets)}")
    for aa, oscs in sorted(amino_acid_baskets.items()):
        print(f"  {aa}: {len(oscs)} oscillators")
    print(f"\nOutput saved to: {output_pkl_path}")
    
    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return amino_acid_baskets


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete trajectory-based inference with Rama & dipole analysis"
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
    
    # Load config
    config = load_config_from_yaml(args.config)
    
    # Override output if specified
    if args.output:
        config['infer']['output'] = args.output
    
    output_path = config['infer']['output']
    
    # Run inference
    process_trajectory(config, args.checkpoint, output_path)


if __name__ == '__main__':
    main()



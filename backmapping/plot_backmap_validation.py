#!/usr/bin/env python3
"""
================================================================================
COMPLETE BACKMAPPING ANALYSIS & VISUALIZATION SUITE
================================================================================

Comprehensive plotting and analysis for CG-to-atomistic protein backmapping.

CORE PLOTS (8):
1. Ramachandran scatter (4-panel density with metrics)
2. Ramachandran angle distributions (overlaid histograms)
3. Ramachandran maps (φ vs ψ traditional plots, 4-panel)
4. Bond length distributions (C-O, C-N, N-CA, CA-C)
5. Dipole orientation analysis (3-panel: backbone/sidechain/combined)
6. Dipole component correlations (x, y, z)
7. Per-frame all-atom RMSD (complete structure evaluation)
8. Summary metrics table

ADVANCED METRICS (7):
9. TM-score distribution (structure similarity)
10. GDT-TS distribution (distance-based quality)
11. lDDT per-residue and distribution (local distance difference)
12. Contact map comparison (native contacts vs predicted)
13. Secondary structure Q3 accuracy
14. Ramachandran quality assessment (core/allowed/disallowed)
15. Clash analysis (steric violations)

All plots use:
- Consistent font sizing (base + relative adjustments)
- All-atom backbone coordinates where applicable
- Clean, publication-ready layouts
- Proper statistical reporting

Author: Backmapping Project
Date: 2025
================================================================================
"""

import argparse
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.spatial.distance import cdist, euclidean

warnings.filterwarnings("ignore")


# =============================================================================
# PUBLICATION STYLING WITH CONSISTENT FONT SIZES
# =============================================================================

def set_publication_style(base_font_size: int = 10):
    """Set publication-quality matplotlib style with CONSISTENT font sizing.
    
    All font sizes are defined relative to base_font_size:
    - axis labels: base
    - title: base + 2
    - tick labels: base - 1  
    - legend: base - 1
    - figure title: base + 3
    
    This ensures font size parameter is actually used throughout.
    """
    plt.rcParams.update({
        # Font settings - ALL relative to base
        "font.size": base_font_size,
        "axes.labelsize": base_font_size,
        "axes.titlesize": base_font_size + 2,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 3,
        
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        
        # Line widths
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        
        # Grid and layout
        "axes.grid": False,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "0.8",
    })


# Colorblind-friendly palette
COLORS = {
    'blue': '#0173B2',      # Ground truth / primary
    'orange': '#DE8F05',    # Predicted / secondary
    'green': '#029E73',     # Good / acceptable
    'red': '#CC78BC',       # Errors / outliers
    'purple': '#7E2F8E',    # Additional category
    'cyan': '#56B4E9',      # Highlights
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BackmappingData:
    """Container for backmapping evaluation data."""
    oscillators: List[Dict]
    n_oscillators: int
    n_files: int
    sources: List[Path]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pickle_files(root: Path, splits: List[str], pattern: str = "*.pkl") -> BackmappingData:
    """Load and aggregate all pickle files from specified splits."""
    all_oscillators = []
    sources = []
    
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[WARNING] Split directory not found: {split_dir}")
            continue
        
        pkl_files = sorted(split_dir.rglob(pattern))
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle different data structures
                if isinstance(data, dict):
                    for aa_type, oscs in data.items():
                        if isinstance(oscs, list):
                            all_oscillators.extend(oscs)
                elif isinstance(data, list):
                    all_oscillators.extend(data)
                
                sources.append(pkl_file)
                
            except Exception as e:
                print(f"[ERROR] Failed to load {pkl_file}: {e}")
                continue
    
    return BackmappingData(
        oscillators=all_oscillators,
        n_oscillators=len(all_oscillators),
        n_files=len(sources),
        sources=sources
    )


# =============================================================================
# RMSD CALCULATION WITH KABSCH ALIGNMENT
# =============================================================================

def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets after optimal superposition.
    
    Uses Kabsch algorithm for optimal rotation.
    
    Args:
        coords1: Nx3 array of reference coordinates
        coords2: Nx3 array of coordinates to align
        
    Returns:
        RMSD value in Angstroms
    """
    if coords1.shape != coords2.shape or len(coords1) == 0:
        return np.nan
    
    # Center both structures
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Kabsch algorithm for optimal rotation
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and compute RMSD
    c2_rotated = c2 @ R
    diff = c1 - c2_rotated
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


# =============================================================================
# EXTRACT ALL BACKBONE + SIDECHAIN ATOMS
# =============================================================================

def extract_all_atoms_from_oscillator(osc: Dict, source: str = 'atoms') -> Tuple[np.ndarray, np.ndarray]:
    """Extract ALL atoms (backbone + sidechain) from an oscillator.
    
    For backbone oscillators:
    - Backbone: N_prev, CA_prev, C_prev, O_prev, N_curr, CA_curr (6 atoms)
    
    For sidechain oscillators (GLN/ASN):
    - All sidechain atoms available
    
    Args:
        osc: Oscillator dictionary
        source: 'atoms' for GT or 'predicted_atoms' for predictions
        
    Returns:
        Tuple of (gt_coords, pred_coords) as Nx3 arrays
    """
    atoms_dict = osc.get(source if source != 'predicted' else 'predicted_atoms')
    
    if not isinstance(atoms_dict, dict):
        return np.array([]), np.array([])
    
    coords = []
    
    if osc.get('oscillator_type') == 'backbone':
        # Backbone atoms in order
        backbone_atoms = ['N_prev', 'CA_prev', 'C_prev', 'O_prev', 
                         'N_curr', 'CA_curr']
        
        for atom_name in backbone_atoms:
            pos = atoms_dict.get(atom_name)
            if pos is not None:
                pos_arr = np.asarray(pos, dtype=float)
                if pos_arr.shape == (3,) and not np.allclose(pos_arr, 0) and np.all(np.isfinite(pos_arr)):
                    coords.append(pos_arr)
    
    elif osc.get('oscillator_type') == 'sidechain':
        # All sidechain atoms
        for atom_name, pos in atoms_dict.items():
            if pos is not None:
                pos_arr = np.asarray(pos, dtype=float)
                if pos_arr.shape == (3,) and not np.allclose(pos_arr, 0) and np.all(np.isfinite(pos_arr)):
                    coords.append(pos_arr)
    
    return np.array(coords) if len(coords) > 0 else np.array([])


# =============================================================================
# PER-FRAME ALL-ATOM RMSD EXTRACTION
# =============================================================================

def extract_per_frame_all_atom_rmsd(oscillators: List[Dict]) -> Tuple[List[float], List[Tuple]]:
    """Extract per-frame ALL-ATOM RMSD (backbone + sidechain).
    
    Groups oscillators by (folder, frame) and computes RMSD over ALL atoms
    in the entire structure, not just backbone.
    
    Returns:
        rmsd_values: List of RMSD values (one per frame)
        frame_labels: List of (folder, frame) tuples
    """
    frames = {}
    
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        
        if folder is None or frame is None:
            continue
        
        key = (folder, frame)
        
        # Get ALL atoms from this oscillator
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if atoms_gt.shape != atoms_pred.shape:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    # Compute RMSD for each complete frame
    rmsd_values = []
    frame_labels = []
    
    for (folder, frame), coords in sorted(frames.items()):
        # Concatenate all atoms from all oscillators in this frame
        try:
            all_gt = np.vstack(coords['gt'])
            all_pred = np.vstack(coords['pred'])
            
            if all_gt.shape != all_pred.shape or len(all_gt) < 3:
                continue
            
            rmsd = compute_rmsd(all_gt, all_pred)
            
            if np.isfinite(rmsd):
                rmsd_values.append(rmsd)
                frame_labels.append((folder, frame))
        except:
            continue
    
    return rmsd_values, frame_labels


# =============================================================================
# RAMACHANDRAN ANGLE EXTRACTION
# =============================================================================

def extract_rama_angles(oscillators: List[Dict], 
                       source_field: str,
                       exclude_zero: bool = True) -> Tuple[np.ndarray, ...]:
    """Extract Ramachandran angles from oscillators.
    
    Args:
        oscillators: List of oscillator dictionaries
        source_field: 'rama_nnfs' for GT or 'predicted_rama_nnfs' for predictions
        exclude_zero: If True, exclude terminal residues (angles = 0)
        
    Returns:
        Tuple of (phi_N, psi_N, phi_C, psi_C) arrays
    """
    phi_N_list, psi_N_list = [], []
    phi_C_list, psi_C_list = [], []
    
    for osc in oscillators:
        if osc.get("oscillator_type") != "backbone":
            continue
        
        rama = osc.get(source_field)
        if not isinstance(rama, dict):
            continue
        
        phi_N = rama.get("phi_N")
        psi_N = rama.get("psi_N")
        phi_C = rama.get("phi_C")
        psi_C = rama.get("psi_C")
        
        if any(v is None for v in [phi_N, psi_N, phi_C, psi_C]):
            continue
        
        angles = np.array([phi_N, psi_N, phi_C, psi_C], dtype=float)
        if not np.all(np.isfinite(angles)):
            continue
        
        if exclude_zero and np.any(np.abs(angles) < 1e-6):
            continue
        
        phi_N_list.append(phi_N)
        psi_N_list.append(psi_N)
        phi_C_list.append(phi_C)
        psi_C_list.append(psi_C)
    
    return (
        np.array(phi_N_list),
        np.array(psi_N_list),
        np.array(phi_C_list),
        np.array(psi_C_list)
    )


def compute_angle_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive angle prediction metrics.
    
    Uses circular statistics appropriate for angles.
    """
    # Circular difference
    diff = (gt - pred + 180) % 360 - 180
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    mask = np.isfinite(gt) & np.isfinite(pred)
    if np.sum(mask) > 2:
        pearson_r, _ = stats.pearsonr(gt[mask], pred[mask])
        spearman_r, _ = stats.spearmanr(gt[mask], pred[mask])
        
        # Circular correlation coefficient
        sin_diff = np.sin(np.deg2rad(diff[mask]))
        cos_diff = np.cos(np.deg2rad(diff[mask]))
        circ_corr = np.sqrt(1 - np.mean(sin_diff**2) - np.mean((cos_diff - 1)**2))
    else:
        pearson_r = spearman_r = circ_corr = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'circular_corr': circ_corr,
        'n': len(gt)
    }


# =============================================================================
# BOND LENGTH EXTRACTION
# =============================================================================

def extract_bond_lengths(oscillators: List[Dict], source: str = 'ground_truth') -> Dict[str, np.ndarray]:
    """Extract peptide bond lengths from oscillators.
    
    Standard peptide bonds:
    - C-O: peptide carbonyl (1.23 Å)
    - C-N: peptide bond (1.33 Å)
    - N-CA: backbone (1.46 Å)
    - CA-C: backbone (1.52 Å)
    """
    bonds = {
        'C-O': [],
        'C-N': [],
        'N-CA': [],
        'CA-C': []
    }
    
    atoms_field = 'atoms' if source == 'ground_truth' else 'predicted_atoms'
    
    for osc in oscillators:
        if osc.get('oscillator_type') != 'backbone':
            continue
        
        atoms = osc.get(atoms_field)
        if not isinstance(atoms, dict):
            continue
        
        try:
            # Get atom positions
            C_prev = np.asarray(atoms.get('C_prev'), dtype=float)
            O_prev = np.asarray(atoms.get('O_prev'), dtype=float)
            N_curr = np.asarray(atoms.get('N_curr'), dtype=float)
            CA_curr = np.asarray(atoms.get('CA_curr'), dtype=float)
            CA_prev = np.asarray(atoms.get('CA_prev'), dtype=float)
            
            # Validate
            if any(x.shape != (3,) for x in [C_prev, O_prev, N_curr, CA_curr, CA_prev]):
                continue
            if any(np.allclose(x, 0) or not np.all(np.isfinite(x)) 
                   for x in [C_prev, O_prev, N_curr, CA_curr, CA_prev]):
                continue
            
            # Calculate bond lengths
            bonds['C-O'].append(np.linalg.norm(C_prev - O_prev))
            bonds['C-N'].append(np.linalg.norm(C_prev - N_curr))
            bonds['N-CA'].append(np.linalg.norm(N_curr - CA_curr))
            bonds['CA-C'].append(np.linalg.norm(CA_prev - C_prev))
            
        except:
            continue
    
    return {k: np.array(v) for k, v in bonds.items()}


# =============================================================================
# DIPOLE EXTRACTION
# =============================================================================

def extract_dipoles(oscillators: List[Dict], 
                   dipole_field: str,
                   oscillator_type: Optional[str] = None) -> np.ndarray:
    """Extract dipole vectors from oscillators.
    
    Args:
        oscillators: List of oscillators
        dipole_field: 'atomistic_dipole' or 'predicted_dipole'
        oscillator_type: 'backbone', 'sidechain', or None for all
        
    Returns:
        Nx3 array of normalized dipole vectors
    """
    dipoles = []
    
    for osc in oscillators:
        if oscillator_type is not None:
            if osc.get('oscillator_type') != oscillator_type:
                continue
        
        dip = osc.get(dipole_field)
        if dip is None:
            continue
        
        try:
            dip_arr = np.asarray(dip, dtype=float)
            if dip_arr.shape != (3,):
                continue
            
            # Normalize
            norm = np.linalg.norm(dip_arr)
            if norm > 1e-10:
                dipoles.append(dip_arr / norm)
        except:
            continue
    
    return np.array(dipoles) if len(dipoles) > 0 else np.array([])


# TO BE CONTINUED IN PART 2...
# =============================================================================
# PLOT 1: RAMACHANDRAN SCATTER (4-PANEL DENSITY)
# =============================================================================

def plot_ramachandran_scatter(gt_angles: Tuple, pred_angles: Tuple, 
                              outpath: Path, dpi: int = 300):
    """4-panel Ramachandran scatter with 2D density coloring."""
    from matplotlib.ticker import ScalarFormatter
    
    base_fs = plt.rcParams['font.size']
    
    gt_angles = gt_angles[:2]
    pred_angles = pred_angles[:2]
    
    fig = plt.figure(figsize=(12, 5.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.08, wspace=0.75,
                  left=0.08, right=0.95, top=0.90, bottom=0.15)
    
    from matplotlib.colors import LinearSegmentedColormap
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap_white = LinearSegmentedColormap.from_list('viridis_white', colors)
    
    angle_names = [
        (r'$\phi_\mathrm{true}$', r'$\phi_\mathrm{pred}$'),
        (r'$\psi_\mathrm{true}$', r'$\psi_\mathrm{pred}$')
    ]
    
    for idx, (gt, pred, (x_label, y_label)) in enumerate(zip(gt_angles, pred_angles, angle_names)):
        ax = fig.add_subplot(gs[0, idx])
        
        mask = np.isfinite(gt) & np.isfinite(pred)
        gt_f = gt[mask]
        pred_f = pred[mask]
        
        if len(gt_f) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=base_fs)
            continue
        
        h, xedges, yedges = np.histogram2d(gt_f, pred_f, bins=90, 
                                           range=[[-180, 180], [-180, 180]])
        
        h_display = h.T.copy()
        h_display[h_display == 0] = np.nan
        
        im = ax.imshow(h_display, origin='lower', aspect='auto',
                      extent=[-180, 180, -180, 180],
                      cmap=cmap_white, interpolation='nearest', vmin=1)
        
        ax.plot([-180, 180], [-180, 180], 'r--', lw=1.5, alpha=0.8)
        
        metrics = compute_angle_metrics(gt_f, pred_f)
        
        stats_text = (
            f"$\\rho = {metrics['pearson']:.3f}$\n"
            f"MAE $= {metrics['mae']:.1f}°$"
        )
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=base_fs-1, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='gray', linewidth=0.5))
        
        ax.set_xlabel(f'{x_label} (°)', fontsize=base_fs)
        ax.set_ylabel(f'{y_label} (°)', fontsize=base_fs)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', rotation=270, labelpad=15, fontsize=base_fs-1)
        
        # Apply ScalarFormatter with offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
        cbar.ax.tick_params(labelsize=base_fs-2)
        cbar.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# PLOT 2: RAMACHANDRAN DISTRIBUTIONS (OVERLAID HISTOGRAMS)
# =============================================================================

def plot_ramachandran_distributions(gt_angles: Tuple, pred_angles: Tuple,
                                   outpath: Path, dpi: int = 300):
    """4-panel overlaid histograms for angle distributions."""
    base_fs = plt.rcParams['font.size']
    
    # Only use first 2 angles (N-side only)
    gt_angles = gt_angles[:2]
    pred_angles = pred_angles[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()
    
    angle_names = [r'$\phi$', r'$\psi$']
    
    for idx, (gt, pred, name) in enumerate(zip(gt_angles, pred_angles, angle_names)):
        ax = axes[idx]
        
        mask_gt = np.isfinite(gt)
        mask_pred = np.isfinite(pred)
        
        if not mask_gt.any() or not mask_pred.any():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=base_fs)
            continue
        
        bins = np.linspace(-180, 180, 73)
        
        ax.hist(gt[mask_gt], bins=bins, alpha=0.6, color=COLORS['blue'],
               label='True', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(pred[mask_pred], bins=bins, alpha=0.6, color=COLORS['orange'],
               label='Pred', density=True, edgecolor='black', linewidth=0.5)
        
        # KS test
        if len(gt[mask_gt]) > 0 and len(pred[mask_pred]) > 0:
            ks_stat, ks_pval = stats.ks_2samp(gt[mask_gt], pred[mask_pred])
            ax.text(0.02, 0.98, f'KS = {ks_stat:.3f}',
                   transform=ax.transAxes, fontsize=base_fs-2,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'{name} (°)', fontsize=base_fs)
        ax.set_ylabel('Density', fontsize=base_fs)
        ax.legend(loc='upper right', fontsize=base_fs-1, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=base_fs-1)
    
    plt.tight_layout(w_pad=4.0)
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")

# =============================================================================
# PLOT 3: RAMACHANDRAN MAPS (φ vs ψ, 4-PANEL)
# =============================================================================

def plot_ramachandran_maps(gt_angles: Tuple, pred_angles: Tuple,
                          outpath: Path, dpi: int = 300):
    """Traditional Ramachandran φ vs ψ maps (2-panel, square aspect)."""
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    base_fs = plt.rcParams['font.size']
    
    gt_angles = gt_angles[:2]
    pred_angles = pred_angles[:2]
    
    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.08, wspace=0.75,
                  left=0.08, right=0.95, top=0.92, bottom=0.12)
    
    from matplotlib.colors import LinearSegmentedColormap
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap_white = LinearSegmentedColormap.from_list('viridis_white', colors)
    
    # N-side GT
    ax1 = fig.add_subplot(gs[0, 0])
    phi_N_gt, psi_N_gt = gt_angles[0], gt_angles[1]
    mask_gt_n = np.isfinite(phi_N_gt) & np.isfinite(psi_N_gt)
    
    h1, _, _ = np.histogram2d(phi_N_gt[mask_gt_n], psi_N_gt[mask_gt_n], 
                              bins=90, range=[[-180, 180], [-180, 180]])
    h1_display = h1.T.copy()
    h1_display[h1_display == 0] = np.nan
    
    im1 = ax1.imshow(h1_display, origin='lower', extent=[-180, 180, -180, 180],
                    cmap=cmap_white, aspect='equal', interpolation='nearest', vmin=1)
    ax1.set_xlabel(r'$\phi$ (°)', fontsize=base_fs)
    ax1.set_ylabel(r'$\psi$ (°)', fontsize=base_fs)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.tick_params(labelsize=base_fs-1)
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="4%", pad=0.10)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Count', rotation=270, labelpad=15, fontsize=base_fs-1)
    
    # Apply ScalarFormatter with offset
    formatter1 = ScalarFormatter(useMathText=True)
    formatter1.set_powerlimits((0, 0))
    cbar1.ax.yaxis.set_major_formatter(formatter1)
    cbar1.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
    cbar1.ax.tick_params(labelsize=base_fs-2)
    cbar1.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    # N-side Pred
    ax2 = fig.add_subplot(gs[0, 1])
    phi_N_pred, psi_N_pred = pred_angles[0], pred_angles[1]
    mask_pred_n = np.isfinite(phi_N_pred) & np.isfinite(psi_N_pred)
    
    h2, _, _ = np.histogram2d(phi_N_pred[mask_pred_n], psi_N_pred[mask_pred_n],
                              bins=90, range=[[-180, 180], [-180, 180]])
    h2_display = h2.T.copy()
    h2_display[h2_display == 0] = np.nan
    
    im2 = ax2.imshow(h2_display, origin='lower', extent=[-180, 180, -180, 180],
                    cmap=cmap_white, aspect='equal', interpolation='nearest', vmin=1)
    ax2.set_xlabel(r'$\phi$ (°)', fontsize=base_fs)
    ax2.set_ylabel(r'$\psi$ (°)', fontsize=base_fs)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-180, 180)
    ax2.tick_params(labelsize=base_fs-1)
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="4%", pad=0.10)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Count', rotation=270, labelpad=15, fontsize=base_fs-1)
    
    # Apply ScalarFormatter with offset
    formatter2 = ScalarFormatter(useMathText=True)
    formatter2.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_major_formatter(formatter2)
    cbar2.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
    cbar2.ax.tick_params(labelsize=base_fs-2)
    cbar2.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    fig.suptitle(f'True (left) vs Pred (right) — N = {np.sum(mask_gt_n):.2e}',
                fontsize=base_fs+2, y=0.98)
    
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


def plot_ramachandran_maps_contour(gt_angles: Tuple, pred_angles: Tuple,
                                   outpath: Path, dpi: int = 300):
    """Ramachandran φ vs ψ maps with smooth contours (2-panel, publication quality)."""
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage import gaussian_filter
    
    base_fs = plt.rcParams['font.size']
    
    gt_angles = gt_angles[:2]
    pred_angles = pred_angles[:2]
    
    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.08, wspace=0.75,
                  left=0.08, right=0.95, top=0.92, bottom=0.12)
    
    # N-side GT
    ax1 = fig.add_subplot(gs[0, 0])
    phi_N_gt, psi_N_gt = gt_angles[0], gt_angles[1]
    mask_gt_n = np.isfinite(phi_N_gt) & np.isfinite(psi_N_gt)
    
    h1, xedges, yedges = np.histogram2d(phi_N_gt[mask_gt_n], psi_N_gt[mask_gt_n], 
                                        bins=90, range=[[-180, 180], [-180, 180]])
    
    h1_smooth = gaussian_filter(h1.T, sigma=1.5)
    
    max_val = h1_smooth.max()
    levels = np.logspace(np.log10(max_val/100), np.log10(max_val), 10)
    
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    
    contourf1 = ax1.contourf(x_centers, y_centers, h1_smooth, 
                             levels=levels, cmap='viridis', alpha=0.8)
    ax1.contour(x_centers, y_centers, h1_smooth, 
                levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    
    ax1.set_xlabel(r'$\phi$ (°)', fontsize=base_fs)
    ax1.set_ylabel(r'$\psi$ (°)', fontsize=base_fs)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=base_fs-1)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="4%", pad=0.10)
    cbar1 = plt.colorbar(contourf1, cax=cax1)
    cbar1.set_label('Count', rotation=270, labelpad=15, fontsize=base_fs-1)
    
    # Apply ScalarFormatter with offset
    formatter1 = ScalarFormatter(useMathText=True)
    formatter1.set_powerlimits((0, 0))
    cbar1.ax.yaxis.set_major_formatter(formatter1)
    cbar1.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
    cbar1.ax.tick_params(labelsize=base_fs-2)
    cbar1.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    # N-side Pred
    ax2 = fig.add_subplot(gs[0, 1])
    phi_N_pred, psi_N_pred = pred_angles[0], pred_angles[1]
    mask_pred_n = np.isfinite(phi_N_pred) & np.isfinite(psi_N_pred)
    
    h2, xedges, yedges = np.histogram2d(phi_N_pred[mask_pred_n], psi_N_pred[mask_pred_n],
                                        bins=90, range=[[-180, 180], [-180, 180]])
    
    h2_smooth = gaussian_filter(h2.T, sigma=1.5)
    
    max_val2 = h2_smooth.max()
    levels2 = np.logspace(np.log10(max_val2/100), np.log10(max_val2), 10)
    
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    
    contourf2 = ax2.contourf(x_centers, y_centers, h2_smooth, 
                             levels=levels2, cmap='viridis', alpha=0.8)
    ax2.contour(x_centers, y_centers, h2_smooth, 
                levels=levels2, colors='black', linewidths=0.5, alpha=0.3)
    
    ax2.set_xlabel(r'$\phi$ (°)', fontsize=base_fs)
    ax2.set_ylabel(r'$\psi$ (°)', fontsize=base_fs)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-180, 180)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=base_fs-1)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="4%", pad=0.10)
    cbar2 = plt.colorbar(contourf2, cax=cax2)
    cbar2.set_label('Count', rotation=270, labelpad=15, fontsize=base_fs-1)
    
    # Apply ScalarFormatter with offset
    formatter2 = ScalarFormatter(useMathText=True)
    formatter2.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_major_formatter(formatter2)
    cbar2.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
    cbar2.ax.tick_params(labelsize=base_fs-2)
    cbar2.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    fig.suptitle(f'True (left) vs Pred (right) — N = {np.sum(mask_gt_n):.2e}',
                fontsize=base_fs+2, y=0.98)
    
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# TO BE CONTINUED...
# =============================================================================
# PLOT 4: BOND LENGTH DISTRIBUTIONS
# =============================================================================

def plot_bond_length_distributions(oscillators: List[Dict], 
                                  outpath: Path, dpi: int = 300):
    """4-panel bond length distributions with standard values."""
    base_fs = plt.rcParams['font.size']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    bonds_gt = extract_bond_lengths(oscillators, source='ground_truth')
    bonds_pred = extract_bond_lengths(oscillators, source='predicted')
    
    # Standard bond lengths (Angstroms)
    standard_bonds = {
        'C-O': 1.23,
        'C-N': 1.33,
        'N-CA': 1.46,
        'CA-C': 1.52
    }
    
    for idx, bond_name in enumerate(['C-O', 'C-N', 'N-CA', 'CA-C']):
        ax = axes[idx]
        
        gt_bonds = bonds_gt.get(bond_name, np.array([]))
        pred_bonds = bonds_pred.get(bond_name, np.array([]))
        
        if len(gt_bonds) == 0 and len(pred_bonds) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=base_fs)
            continue
        
        all_bonds = np.concatenate([gt_bonds, pred_bonds]) if len(pred_bonds) > 0 else gt_bonds
        bins = np.linspace(max(0.5, all_bonds.min() - 0.1), 
                          all_bonds.max() + 0.1, 50)
        
        if len(gt_bonds) > 0:
            gt_mean = gt_bonds.mean()
            gt_std = gt_bonds.std()
            ax.hist(gt_bonds, bins=bins, alpha=0.6, color=COLORS['blue'],
                    label=f'GT (σ={gt_std:.3f} Å)', 
                    density=True, edgecolor='black', linewidth=0.5)
        if len(pred_bonds) > 0:
            pred_mean = pred_bonds.mean()
            pred_std = pred_bonds.std()
            ax.hist(pred_bonds, bins=bins, alpha=0.6, color=COLORS['orange'],
                    label=f'Pred (σ={pred_std:.3f} Å)', 
                    density=True, edgecolor='black', linewidth=0.5)
        # Reference value line (NOT standard deviation!)
        standard = standard_bonds[bond_name]
        ax.axvline(standard, color='red', linestyle='--', linewidth=2,
                   label=f'Reference ({standard:.2f} Å)')
        
        ax.set_xlabel(f'{bond_name} bond length (Å)', fontsize=base_fs)
        ax.set_ylabel('Density', fontsize=base_fs)
        ax.legend(loc='best', fontsize=base_fs-1, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=base_fs-1)
    
    plt.tight_layout(w_pad=3.5, h_pad=2.5)
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# PLOT 5: DIPOLE ORIENTATION ANALYSIS
# =============================================================================

def plot_dipole_analysis(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Dipole orientation histogram with KDE (combined only, log scale)."""
    from matplotlib.ticker import LogFormatterSciNotation
    
    base_fs = plt.rcParams['font.size']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Combined dipoles only
    gt_dip = extract_dipoles(oscillators, 'atomistic_dipole', None)
    pred_dip = extract_dipoles(oscillators, 'predicted_dipole', None)
    
    if len(gt_dip) == 0 or len(pred_dip) == 0:
        ax.text(0.5, 0.5, 'No data', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=base_fs)
    else:
        n = min(len(gt_dip), len(pred_dip))
        dots = np.sum(gt_dip[:n] * pred_dip[:n], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.rad2deg(np.arccos(dots))
        
        bins = np.linspace(0, 180, 91)
        
        # CRITICAL FIX: Set log scale BEFORE creating histogram
        ax.set_yscale('log')
        
        # Create histogram (it will automatically handle log scale)
        counts, bin_edges, patches = ax.hist(angles, bins=bins, color=COLORS['blue'], 
                                             alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(angles)
        x_kde = np.linspace(0, 180, 500)
        y_kde = kde(x_kde)
        
        # Scale KDE to match histogram (convert density to counts)
        bin_width = bin_edges[1] - bin_edges[0]
        y_kde_scaled = y_kde * len(angles) * bin_width
        
        ax.plot(x_kde, y_kde_scaled, color='red', linewidth=2, alpha=0.8)
        
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        
        ax.axvline(mean_angle, color='darkred', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_angle:.1f}°')
        ax.axvline(median_angle, color='orange', linestyle='--', linewidth=2,
                  label=f'Median = {median_angle:.1f}°')
        
        ax.set_xlabel(r'Deviation between $\vec{\mu}_\mathrm{true}$ and $\vec{\mu}_\mathrm{pred}$ (°)', 
                     fontsize=base_fs)
        ax.set_ylabel('Count (log scale)', fontsize=base_fs)
        
        # Use LogFormatterSciNotation for cleaner log scale display
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        
        ax.set_title(f'N = {n:.2e}', fontsize=base_fs+1, pad=10)
        ax.legend(loc='upper right', fontsize=base_fs-1, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=base_fs-1)
        ax.set_xlim(0, 180)
        
        # Set y-axis limits to avoid empty space at bottom
        ax.set_ylim(bottom=max(1, counts[counts > 0].min() * 0.5))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")

def plot_dipole_analysis_linear(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Dipole orientation histogram with KDE (combined only, linear scale)."""
    
    base_fs = plt.rcParams['font.size']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Combined dipoles only
    gt_dip = extract_dipoles(oscillators, 'atomistic_dipole', None)
    pred_dip = extract_dipoles(oscillators, 'predicted_dipole', None)
    
    if len(gt_dip) == 0 or len(pred_dip) == 0:
        ax.text(0.5, 0.5, 'No data', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=base_fs)
    else:
        n = min(len(gt_dip), len(pred_dip))
        dots = np.sum(gt_dip[:n] * pred_dip[:n], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.rad2deg(np.arccos(dots))
        
        bins = np.linspace(0, 180, 91)
        
        # Create histogram (LINEAR SCALE - no set_yscale call)
        counts, bin_edges, patches = ax.hist(angles, bins=bins, color=COLORS['blue'], 
                                             alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(angles)
        x_kde = np.linspace(0, 180, 500)
        y_kde = kde(x_kde)
        
        # Scale KDE to match histogram (convert density to counts)
        bin_width = bin_edges[1] - bin_edges[0]
        y_kde_scaled = y_kde * len(angles) * bin_width
        
        ax.plot(x_kde, y_kde_scaled, color='red', linewidth=2, alpha=0.8)
        
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        
        ax.axvline(mean_angle, color='darkred', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_angle:.1f}°')
        ax.axvline(median_angle, color='orange', linestyle='--', linewidth=2,
                  label=f'Median = {median_angle:.1f}°')
        
        ax.set_xlabel(r'Deviation between $\vec{\mu}_\mathrm{true}$ and $\vec{\mu}_\mathrm{pred}$ (°)', 
                     fontsize=base_fs)
        ax.set_ylabel('Count', fontsize=base_fs)
                
        ax.set_title(f'N = {n:,}', fontsize=base_fs+1, pad=10)  # Using comma format instead of scientific
        ax.legend(loc='upper right', fontsize=base_fs-1, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=base_fs-1)
        ax.set_xlim(0, 180)
        
        # Optional: Set y-axis to start at 0 for linear scale
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")

# =============================================================================
# PLOT 6: DIPOLE COMPONENT CORRELATIONS
# =============================================================================

def plot_dipole_components(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """3-panel x/y/z dipole component correlations."""
    from matplotlib.ticker import ScalarFormatter
    
    base_fs = plt.rcParams['font.size']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    from matplotlib.colors import LinearSegmentedColormap
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap_white = LinearSegmentedColormap.from_list('viridis_white', colors)
    
    gt_dip = extract_dipoles(oscillators, 'atomistic_dipole', None)
    pred_dip = extract_dipoles(oscillators, 'predicted_dipole', None)
    
    if len(gt_dip) == 0 or len(pred_dip) == 0:
        print("[WARNING] No dipole data for component plot")
        return
    
    n = min(len(gt_dip), len(pred_dip))
    gt_dip = gt_dip[:n]
    pred_dip = pred_dip[:n]
    
    components = ['x', 'y', 'z']
    
    for idx, (ax, comp) in enumerate(zip(axes, components)):
        x = gt_dip[:, idx]
        y = pred_dip[:, idx]
        
        h, xedges, yedges = np.histogram2d(x, y, bins=80, 
                                           range=[[-1, 1], [-1, 1]])
        
        h_display = h.T.copy()
        h_display[h_display == 0] = np.nan
        
        im = ax.imshow(h_display, origin='lower', extent=[-1, 1, -1, 1],
                      cmap=cmap_white, aspect='auto', interpolation='nearest', vmin=1)
        
        ax.plot([-1, 1], [-1, 1], 'r--', lw=1.5, alpha=0.8)
        
        # Calculate Pearson correlation for title
        if len(x) > 2:
            pearson_r, _ = stats.pearsonr(x, y)
            ax.set_title(f'$\\rho$ = {pearson_r:.3f}', fontsize=base_fs, pad=8)
        
        ax.set_xlabel(f'$\\mu_{{{comp}}}^\\mathrm{{true}}$', fontsize=base_fs)
        ax.set_ylabel(f'$\\mu_{{{comp}}}^\\mathrm{{pred}}$', fontsize=base_fs)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=base_fs-1)
        
        cbar = plt.colorbar(im, ax=ax, label='Count', fraction=0.046, pad=0.04)
        
        # CHANGE: Use ScalarFormatter with offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.ticklabel_format(style='plain', axis='y', useOffset=True)
        cbar.ax.tick_params(labelsize=base_fs-2)
        cbar.ax.yaxis.get_offset_text().set_fontsize(base_fs-2)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")

# =============================================================================
# PLOT 7: PER-FRAME ALL-ATOM RMSD
# =============================================================================

def plot_rmsd_per_frame(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Per-frame all-atom RMSD distribution (clean, publication-ready)."""
    base_fs = plt.rcParams['font.size']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    rmsd_values, frame_labels = extract_per_frame_all_atom_rmsd(oscillators)
    
    if len(rmsd_values) == 0:
        print("[WARNING] No per-frame RMSD data available")
        return
    
    rmsd_arr = np.array(rmsd_values)
    
    # Use 99th percentile for x-axis
    q99 = np.percentile(rmsd_arr, 99)
    max_val = min(q99 * 1.1, rmsd_arr.max())
    
    n_bins = 60
    bins = np.linspace(0, max_val, n_bins + 1)
    
    ax.hist(rmsd_arr[rmsd_arr <= max_val], bins=bins, 
           color=COLORS['blue'], alpha=0.75,
           edgecolor='black', linewidth=0.5)
    
    mean_rmsd = np.mean(rmsd_arr)
    median_rmsd = np.median(rmsd_arr)
    std_rmsd = np.std(rmsd_arr)
    p99 = np.percentile(rmsd_arr, 99)
    
    # Compact legend with shorter dashes
    ax.axvline(mean_rmsd, color='red', linestyle='--', linewidth=2,
              alpha=0.8, label=f'Mean: {mean_rmsd:.3f} Å | Median: {median_rmsd:.3f} Å', 
              zorder=10)
    ax.axvline(median_rmsd, color='orange', linestyle='--', linewidth=2,
              alpha=0.8, zorder=10)
    
    ax.set_xlabel('All-Atom RMSD (Å)', fontsize=base_fs)
    ax.set_ylabel('Count', fontsize=base_fs)
    ax.set_title(f'All-Atom RMSD Distribution (N = {len(rmsd_values):.2e} frames)',
                fontsize=base_fs+2, pad=12)
    
    # Single legend entry with both mean and median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_rmsd:.3f} Å'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_rmsd:.3f} Å')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, 
             fontsize=base_fs-1)
    
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(labelsize=base_fs-1)
    
    ax.set_xlim(rmsd_arr.min(), rmsd_arr.max())
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")
    
    print(f"\nPer-Frame All-Atom RMSD Summary:")
    print(f"  Frames analyzed: {len(rmsd_values)}")
    print(f"  Mean RMSD:   {mean_rmsd:.3f} Å")
    print(f"  Median RMSD: {median_rmsd:.3f} Å")
    print(f"  Std RMSD:    {std_rmsd:.3f} Å")
    print(f"  99th percentile: {p99:.3f} Å")


# ADVANCED METRICS START HERE...
# =============================================================================
# ADVANCED METRIC: TM-SCORE
# =============================================================================

def compute_tm_score(coords_gt: np.ndarray, coords_pred: np.ndarray) -> float:
    """Compute TM-score (template modeling score).
    
    TM-score ∈ [0, 1], where >0.5 indicates same fold.
    """
    if coords_gt.shape != coords_pred.shape or len(coords_gt) < 3:
        return np.nan
    
    L = len(coords_gt)
    d0 = 1.24 * (L - 15)**(1.0/3.0) - 1.8 if L > 15 else 0.5
    
    # Superimpose
    rmsd_val = compute_rmsd(coords_gt, coords_pred)
    
    # Center
    c1 = coords_gt - coords_gt.mean(axis=0)
    c2 = coords_pred - coords_pred.mean(axis=0)
    
    # Kabsch
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    c2_rot = c2 @ R
    
    # TM-score
    distances = np.sqrt(np.sum((c1 - c2_rot)**2, axis=1))
    tm = np.sum(1.0 / (1.0 + (distances / d0)**2)) / L
    
    return tm


def extract_per_frame_tm_scores(oscillators: List[Dict]) -> List[float]:
    """Extract TM-scores for all frames."""
    frames = {}
    
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        
        if folder is None or frame is None:
            continue
        
        key = (folder, frame)
        
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    tm_scores = []
    
    for coords in frames.values():
        try:
            all_gt = np.vstack(coords['gt'])
            all_pred = np.vstack(coords['pred'])
            
            if all_gt.shape != all_pred.shape or len(all_gt) < 3:
                continue
            
            tm = compute_tm_score(all_gt, all_pred)
            if np.isfinite(tm):
                tm_scores.append(tm)
        except:
            continue
    
    return tm_scores


def plot_tm_score_distribution(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Plot TM-score distribution."""
    base_fs = plt.rcParams['font.size']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    tm_scores = extract_per_frame_tm_scores(oscillators)
    
    if len(tm_scores) == 0:
        print("[WARNING] No TM-score data available")
        return
    
    tm_arr = np.array(tm_scores)
    
    bins = np.linspace(0, 1, 51)
    ax.hist(tm_arr, bins=bins, color=COLORS['blue'], alpha=0.75,
           edgecolor='black', linewidth=0.5)
    
    mean_tm = np.mean(tm_arr)
    median_tm = np.median(tm_arr)
    
    ax.axvline(mean_tm, color='red', linestyle='--', linewidth=2,
              alpha=0.8, label='Mean:   {mean_tm:.3f}')
    ax.axvline(median_tm, color='orange', linestyle='--', linewidth=2,
              alpha=0.8, label='Median: {median_tm:.3f}')
    ax.axvline(0.5, color='green', linestyle=':', linewidth=2,
              alpha=0.7, label='Same fold threshold')
    
    ax.set_xlabel('TM-score', fontsize=base_fs)
    ax.set_ylabel('Count', fontsize=base_fs)
    ax.set_title(f'TM-Score Distribution (N = {len(tm_scores):,} frames)',
                fontsize=base_fs+2, pad=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=base_fs-1)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(labelsize=base_fs-1)
    
    pct_above = 100 * np.sum(tm_arr > 0.5) / len(tm_arr)
    
    stats_text = (
        f"Mean: {mean_tm:.3f}\n"
        f"Median: {median_tm:.3f}\n"
        f"Std: {np.std(tm_arr):.3f}\n"
        f">0.5 (same fold): {pct_above:.1f}%"
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=base_fs-1, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='gray', linewidth=0.8))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")

# =============================================================================
# ADVANCED METRICS COMPLETE IMPLEMENTATION
# =============================================================================
# Add these functions to plot_backmapping_complete.py before the main() function
# These implement the remaining 6 advanced metrics (TM-score already done)

# =============================================================================
# ADVANCED METRIC 2: GDT-TS (Global Distance Test - Total Score)
# =============================================================================

def compute_gdt_ts(coords_gt: np.ndarray, coords_pred: np.ndarray) -> float:
    """Compute GDT-TS score (CASP standard).
    
    GDT-TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8) / 4
    where GDT_Pn = % of residues within n Angstroms after superposition.
    
    Returns:
        GDT-TS score ∈ [0, 100], where >50 is good, >75 is excellent.
    """
    if coords_gt.shape != coords_pred.shape or len(coords_gt) < 3:
        return np.nan
    
    # Superimpose using Kabsch
    c1 = coords_gt - coords_gt.mean(axis=0)
    c2 = coords_pred - coords_pred.mean(axis=0)
    
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    c2_rot = c2 @ R
    
    # Compute distances
    distances = np.sqrt(np.sum((c1 - c2_rot)**2, axis=1))
    
    # Count residues within cutoffs
    n_total = len(distances)
    gdt_p1 = 100.0 * np.sum(distances < 1.0) / n_total
    gdt_p2 = 100.0 * np.sum(distances < 2.0) / n_total
    gdt_p4 = 100.0 * np.sum(distances < 4.0) / n_total
    gdt_p8 = 100.0 * np.sum(distances < 8.0) / n_total
    
    gdt_ts = (gdt_p1 + gdt_p2 + gdt_p4 + gdt_p8) / 4.0
    
    return gdt_ts


def extract_per_frame_gdt_ts(oscillators: List[Dict]) -> List[float]:
    """Extract GDT-TS scores for all frames."""
    frames = {}
    
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        
        if folder is None or frame is None:
            continue
        
        key = (folder, frame)
        
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    gdt_scores = []
    
    for coords in frames.values():
        try:
            all_gt = np.vstack(coords['gt'])
            all_pred = np.vstack(coords['pred'])
            
            if all_gt.shape != all_pred.shape or len(all_gt) < 3:
                continue
            
            gdt = compute_gdt_ts(all_gt, all_pred)
            if np.isfinite(gdt):
                gdt_scores.append(gdt)
        except:
            continue
    
    return gdt_scores


def plot_gdt_ts_distribution(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Plot GDT-TS distribution."""
    base_fs = plt.rcParams['font.size']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    gdt_scores = extract_per_frame_gdt_ts(oscillators)
    
    if len(gdt_scores) == 0:
        print("[WARNING] No GDT-TS data available")
        return
    
    gdt_arr = np.array(gdt_scores)
    
    bins = np.linspace(0, 100, 51)
    ax.hist(gdt_arr, bins=bins, color=COLORS['blue'], alpha=0.75,
           edgecolor='black', linewidth=0.5)
    
    mean_gdt = np.mean(gdt_arr)
    median_gdt = np.median(gdt_arr)
    
    ax.axvline(mean_gdt, color='red', linestyle='--', linewidth=2,
              alpha=0.8, label='Mean')
    ax.axvline(median_gdt, color='orange', linestyle='--', linewidth=2,
              alpha=0.8, label='Median')
    ax.axvline(50, color='green', linestyle=':', linewidth=2,
              alpha=0.7, label='Good quality (>50)')
    ax.axvline(75, color='purple', linestyle=':', linewidth=2,
              alpha=0.7, label='Excellent quality (>75)')
    
    ax.set_xlabel('GDT-TS Score', fontsize=base_fs)
    ax.set_ylabel('Count', fontsize=base_fs)
    ax.set_title(f'GDT-TS Distribution (N = {len(gdt_scores):,} frames)',
                fontsize=base_fs+2, pad=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=base_fs-1)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(labelsize=base_fs-1)
    
    pct_good = 100 * np.sum(gdt_arr > 50) / len(gdt_arr)
    pct_excellent = 100 * np.sum(gdt_arr > 75) / len(gdt_arr)
    
    stats_text = (
        f"Mean: {mean_gdt:.1f}\n"
        f"Median: {median_gdt:.1f}\n"
        f"Std: {np.std(gdt_arr):.1f}\n"
        f">50 (good): {pct_good:.1f}%\n"
        f">75 (excellent): {pct_excellent:.1f}%"
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=base_fs-1, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='gray', linewidth=0.8))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# ADVANCED METRIC 3: lDDT (Local Distance Difference Test)
# =============================================================================

def compute_lddt(coords_gt: np.ndarray, coords_pred: np.ndarray, 
                 r0: float = 15.0) -> Tuple[float, np.ndarray]:
    """Compute lDDT (local distance difference test).
    
    lDDT measures preservation of local interactions.
    
    Args:
        coords_gt: Ground truth coordinates (Nx3)
        coords_pred: Predicted coordinates (Nx3)
        r0: Inclusion radius (default 15 Å)
        
    Returns:
        global_lddt: Overall lDDT score ∈ [0, 1]
        per_residue_lddt: Per-residue lDDT scores
    """
    if coords_gt.shape != coords_pred.shape or len(coords_gt) < 2:
        return np.nan, np.array([])
    
    n = len(coords_gt)
    
    # Distance matrices
    dist_gt = cdist(coords_gt, coords_gt)
    dist_pred = cdist(coords_pred, coords_pred)
    
    # Thresholds for preserved distances
    thresholds = [0.5, 1.0, 2.0, 4.0]
    
    per_residue_lddt = np.zeros(n)
    
    for i in range(n):
        # Find neighbors within r0 in ground truth
        neighbors = np.where((dist_gt[i, :] < r0) & (dist_gt[i, :] > 0))[0]
        
        if len(neighbors) == 0:
            per_residue_lddt[i] = np.nan
            continue
        
        # Check distance preservation
        preserved = 0
        for thr in thresholds:
            diff = np.abs(dist_gt[i, neighbors] - dist_pred[i, neighbors])
            preserved += np.sum(diff < thr)
        
        per_residue_lddt[i] = preserved / (4 * len(neighbors))
    
    # Global lDDT (mean over valid residues)
    valid_mask = np.isfinite(per_residue_lddt)
    global_lddt = np.mean(per_residue_lddt[valid_mask]) if valid_mask.any() else np.nan
    
    return global_lddt, per_residue_lddt


def extract_per_frame_lddt(oscillators: List[Dict]) -> List[float]:
    """Extract lDDT scores for all frames."""
    frames = {}
    
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        
        if folder is None or frame is None:
            continue
        
        key = (folder, frame)
        
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    lddt_scores = []
    
    for coords in frames.values():
        try:
            all_gt = np.vstack(coords['gt'])
            all_pred = np.vstack(coords['pred'])
            
            if all_gt.shape != all_pred.shape or len(all_gt) < 2:
                continue
            
            lddt, _ = compute_lddt(all_gt, all_pred)
            if np.isfinite(lddt):
                lddt_scores.append(lddt)
        except:
            continue
    
    return lddt_scores


def plot_lddt_distribution(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Plot lDDT distribution."""
    base_fs = plt.rcParams['font.size']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    lddt_scores = extract_per_frame_lddt(oscillators)
    
    if len(lddt_scores) == 0:
        print("[WARNING] No lDDT data available")
        return
    
    lddt_arr = np.array(lddt_scores)
    
    bins = np.linspace(0, 1, 51)
    ax.hist(lddt_arr, bins=bins, color=COLORS['blue'], alpha=0.75,
           edgecolor='black', linewidth=0.5)
    
    mean_lddt = np.mean(lddt_arr)
    median_lddt = np.median(lddt_arr)
    
    ax.axvline(mean_lddt, color='red', linestyle='--', linewidth=2,
              alpha=0.8, label='Mean')
    ax.axvline(median_lddt, color='orange', linestyle='--', linewidth=2,
              alpha=0.8, label='Median')
    ax.axvline(0.7, color='green', linestyle=':', linewidth=2,
              alpha=0.7, label='Confident (>0.7)')
    ax.axvline(0.9, color='purple', linestyle=':', linewidth=2,
              alpha=0.7, label='Very high (>0.9)')
    
    ax.set_xlabel('lDDT Score', fontsize=base_fs)
    ax.set_ylabel('Count', fontsize=base_fs)
    ax.set_title(f'lDDT Distribution (N = {len(lddt_scores):,} frames)',
                fontsize=base_fs+2, pad=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=base_fs-1)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(labelsize=base_fs-1)
    
    pct_confident = 100 * np.sum(lddt_arr > 0.7) / len(lddt_arr)
    pct_high = 100 * np.sum(lddt_arr > 0.9) / len(lddt_arr)
    
    stats_text = (
        f"Mean: {mean_lddt:.3f}\n"
        f"Median: {median_lddt:.3f}\n"
        f"Std: {np.std(lddt_arr):.3f}\n"
        f">0.7 (confident): {pct_confident:.1f}%\n"
        f">0.9 (very high): {pct_high:.1f}%"
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=base_fs-1, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='gray', linewidth=0.8))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# ADVANCED METRIC 4: CONTACT MAP ANALYSIS
# =============================================================================

def compute_contact_map(coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Compute contact map (binary matrix of contacts within cutoff)."""
    dist_matrix = cdist(coords, coords)
    contacts = (dist_matrix < cutoff).astype(int)
    # Remove diagonal and upper triangle (symmetric)
    contacts = np.tril(contacts, k=-1)
    return contacts


def compute_contact_metrics(contacts_gt: np.ndarray, contacts_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1 for contact prediction."""
    # Flatten lower triangle only
    gt_flat = contacts_gt[np.tril_indices_from(contacts_gt, k=-1)]
    pred_flat = contacts_pred[np.tril_indices_from(contacts_pred, k=-1)]
    
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def plot_contact_map_comparison(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Plot contact map comparison for one representative frame."""
    base_fs = plt.rcParams['font.size']
    
    # Get one representative frame
    frames = {}
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        if folder is None or frame is None:
            continue
        key = (folder, frame)
        
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    if not frames:
        print("[WARNING] No contact map data available")
        return
    
    # Take first frame
    first_key = list(frames.keys())[0]
    coords_dict = frames[first_key]
    
    try:
        all_gt = np.vstack(coords_dict['gt'])
        all_pred = np.vstack(coords_dict['pred'])
        
        contacts_gt = compute_contact_map(all_gt, cutoff=8.0)
        contacts_pred = compute_contact_map(all_pred, cutoff=8.0)
        
        metrics = compute_contact_metrics(contacts_gt, contacts_pred)
        
        # Plot side-by-side contact maps
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # GT
        im1 = axes[0].imshow(contacts_gt, cmap='Blues', aspect='auto')
        axes[0].set_title('Ground Truth Contacts', fontsize=base_fs+1)
        axes[0].set_xlabel('Residue', fontsize=base_fs)
        axes[0].set_ylabel('Residue', fontsize=base_fs)
        axes[0].tick_params(labelsize=base_fs-1)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Pred
        im2 = axes[1].imshow(contacts_pred, cmap='Blues', aspect='auto')
        axes[1].set_title('Predicted Contacts', fontsize=base_fs+1)
        axes[1].set_xlabel('Residue', fontsize=base_fs)
        axes[1].set_ylabel('Residue', fontsize=base_fs)
        axes[1].tick_params(labelsize=base_fs-1)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference (TP, FP, FN)
        diff_map = np.zeros_like(contacts_gt, dtype=float)
        diff_map[(contacts_gt == 1) & (contacts_pred == 1)] = 1  # TP: green
        diff_map[(contacts_gt == 0) & (contacts_pred == 1)] = 2  # FP: red
        diff_map[(contacts_gt == 1) & (contacts_pred == 0)] = 3  # FN: yellow
        
        cmap = plt.cm.colors.ListedColormap(['white', 'green', 'red', 'yellow'])
        im3 = axes[2].imshow(diff_map, cmap=cmap, aspect='auto', vmin=0, vmax=3)
        axes[2].set_title('Comparison', fontsize=base_fs+1)
        axes[2].set_xlabel('Residue', fontsize=base_fs)
        axes[2].set_ylabel('Residue', fontsize=base_fs)
        axes[2].tick_params(labelsize=base_fs-1)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label=f'TP: {metrics["tp"]}'),
            Patch(facecolor='red', label=f'FP: {metrics["fp"]}'),
            Patch(facecolor='yellow', label=f'FN: {metrics["fn"]}')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', 
                      fontsize=base_fs-1, framealpha=0.9)
        
        fig.suptitle(f'Contact Map Analysis (cutoff=8Å) — '
                    f'Precision={metrics["precision"]:.3f}, Recall={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}',
                    fontsize=base_fs+2, y=0.98)
        
        plt.tight_layout()
        plt.savefig(outpath, dpi=dpi)
        plt.close()
        print(f"✓ Saved: {outpath}")
        
    except Exception as e:
        print(f"[WARNING] Contact map plotting failed: {e}")


# =============================================================================
# ADVANCED METRIC 5: RAMACHANDRAN QUALITY ASSESSMENT
# =============================================================================

def classify_ramachandran_region(phi: float, psi: float) -> str:
    """Classify φ/ψ into favored/allowed/disallowed regions."""
    # Favored regions (from Richardson et al.)
    # Alpha helix
    if (-100 <= phi <= -30) and (-60 <= psi <= 0):
        return 'favored'
    # Beta sheet
    if (-180 <= phi <= -90) and ((90 <= psi <= 180) or (-180 <= psi <= -90)):
        return 'favored'
    # Left-handed helix
    if (30 <= phi <= 90) and (0 <= psi <= 90):
        return 'favored'
    # Allowed (extended)
    if (-90 <= phi <= -30) and (90 <= psi <= 180):
        return 'allowed'
    # Otherwise disallowed
    return 'disallowed'


def plot_ramachandran_quality(gt_angles: Tuple, pred_angles: Tuple,
                              outpath: Path, dpi: int = 300):
    """Plot Ramachandran quality distribution (favored/allowed/disallowed)."""
    base_fs = plt.rcParams['font.size']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Analyze N-side
    phi_N_gt, psi_N_gt = gt_angles[0], gt_angles[1]
    phi_N_pred, psi_N_pred = pred_angles[0], pred_angles[1]
    
    mask_gt = np.isfinite(phi_N_gt) & np.isfinite(psi_N_gt)
    mask_pred = np.isfinite(phi_N_pred) & np.isfinite(psi_N_pred)
    
    # Classify GT
    gt_classes = {'favored': 0, 'allowed': 0, 'disallowed': 0}
    for phi, psi in zip(phi_N_gt[mask_gt], psi_N_gt[mask_gt]):
        region = classify_ramachandran_region(phi, psi)
        gt_classes[region] += 1
    
    # Classify Pred
    pred_classes = {'favored': 0, 'allowed': 0, 'disallowed': 0}
    for phi, psi in zip(phi_N_pred[mask_pred], psi_N_pred[mask_pred]):
        region = classify_ramachandran_region(phi, psi)
        pred_classes[region] += 1
    
    # Bar chart GT
    ax = axes[0]
    categories = ['Favored', 'Allowed', 'Disallowed']
    counts = [gt_classes['favored'], gt_classes['allowed'], gt_classes['disallowed']]
    total = sum(counts)
    percentages = [100 * c / total for c in counts]
    
    bars = ax.bar(categories, percentages, color=[COLORS['green'], COLORS['orange'], COLORS['red']],
                  alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('Percentage (%)', fontsize=base_fs)
    ax.set_title('Ground Truth', fontsize=base_fs+1)
    ax.tick_params(labelsize=base_fs-1)
    ax.set_ylim(0, 100)
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=base_fs-1)
    
    # Bar chart Pred
    ax = axes[1]
    counts_pred = [pred_classes['favored'], pred_classes['allowed'], pred_classes['disallowed']]
    total_pred = sum(counts_pred)
    percentages_pred = [100 * c / total_pred for c in counts_pred]
    
    bars = ax.bar(categories, percentages_pred, color=[COLORS['green'], COLORS['orange'], COLORS['red']],
                  alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('Percentage (%)', fontsize=base_fs)
    ax.set_title('Predicted', fontsize=base_fs+1)
    ax.tick_params(labelsize=base_fs-1)
    ax.set_ylim(0, 100)
    
    for bar, pct in zip(bars, percentages_pred):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=base_fs-1)
    
    fig.suptitle(f'Ramachandran Quality Distribution (N-side, N={total:,} residues)',
                fontsize=base_fs+2, y=0.98)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# ADVANCED METRIC 6: CLASH ANALYSIS
# =============================================================================

def detect_clashes(coords: np.ndarray, clash_cutoff: float = 2.0) -> Tuple[int, float]:
    """Detect steric clashes (atoms too close).
    
    Args:
        coords: Nx3 coordinate array
        clash_cutoff: Distance below which atoms clash (default 2.0 Å)
        
    Returns:
        n_clashes: Number of clashing atom pairs
        clash_fraction: Fraction of atom pairs that clash
    """
    if len(coords) < 2:
        return 0, 0.0
    
    dist_matrix = cdist(coords, coords)
    
    # Get lower triangle (excluding diagonal)
    n = len(coords)
    tril_indices = np.tril_indices(n, k=-1)
    distances = dist_matrix[tril_indices]
    
    n_clashes = np.sum(distances < clash_cutoff)
    n_pairs = len(distances)
    clash_fraction = n_clashes / n_pairs if n_pairs > 0 else 0.0
    
    return n_clashes, clash_fraction


def extract_per_frame_clashes(oscillators: List[Dict]) -> Tuple[List[int], List[int]]:
    """Extract clash counts for GT and predicted structures."""
    frames = {}
    
    for osc in oscillators:
        folder = osc.get('folder')
        frame = osc.get('frame')
        
        if folder is None or frame is None:
            continue
        
        key = (folder, frame)
        
        atoms_gt = extract_all_atoms_from_oscillator(osc, 'atoms')
        atoms_pred = extract_all_atoms_from_oscillator(osc, 'predicted_atoms')
        
        if len(atoms_gt) == 0 or len(atoms_pred) == 0:
            continue
        
        if key not in frames:
            frames[key] = {'gt': [], 'pred': []}
        
        frames[key]['gt'].append(atoms_gt)
        frames[key]['pred'].append(atoms_pred)
    
    clashes_gt = []
    clashes_pred = []
    
    for coords in frames.values():
        try:
            all_gt = np.vstack(coords['gt'])
            all_pred = np.vstack(coords['pred'])
            
            n_clash_gt, _ = detect_clashes(all_gt, clash_cutoff=2.0)
            n_clash_pred, _ = detect_clashes(all_pred, clash_cutoff=2.0)
            
            clashes_gt.append(n_clash_gt)
            clashes_pred.append(n_clash_pred)
        except:
            continue
    
    return clashes_gt, clashes_pred


def plot_clash_analysis(oscillators: List[Dict], outpath: Path, dpi: int = 300):
    """Plot clash comparison between GT and predicted structures."""
    base_fs = plt.rcParams['font.size']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    clashes_gt, clashes_pred = extract_per_frame_clashes(oscillators)
    
    if len(clashes_gt) == 0:
        print("[WARNING] No clash data available")
        return
    
    # Side-by-side bar chart
    x = np.arange(2)
    means = [np.mean(clashes_gt), np.mean(clashes_pred)]
    stds = [np.std(clashes_gt), np.std(clashes_pred)]
    
    bars = ax.bar(x, means, yerr=stds, color=[COLORS['blue'], COLORS['orange']],
                  alpha=0.7, edgecolor='black', linewidth=1, capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Ground Truth', 'Predicted'], fontsize=base_fs)
    ax.set_ylabel('Number of Clashes (<2Å)', fontsize=base_fs)
    ax.set_title(f'Steric Clash Analysis (N = {len(clashes_gt):,} frames)',
                fontsize=base_fs+2, pad=12)
    ax.tick_params(labelsize=base_fs-1)
    ax.grid(True, alpha=0.25, axis='y')
    
    # Add values on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
               f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=base_fs-1)
    
    stats_text = (
        f"GT:   {np.mean(clashes_gt):.1f} ± {np.std(clashes_gt):.1f}\n"
        f"Pred: {np.mean(clashes_pred):.1f} ± {np.std(clashes_pred):.1f}\n"
        f"Frames w/ 0 clashes:\n"
        f"  GT:   {100*np.sum(np.array(clashes_gt)==0)/len(clashes_gt):.1f}%\n"
        f"  Pred: {100*np.sum(np.array(clashes_pred)==0)/len(clashes_pred):.1f}%"
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=base_fs-1, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='gray', linewidth=0.8))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"✓ Saved: {outpath}")


# =============================================================================
# UPDATED MAIN FUNCTION - ADD THESE LINES WHERE INDICATED
# =============================================================================

# In the main() function, after the TM-score plot, add these lines:

"""
    # Add after TM-score plot in main():
    
    if args.advanced_metrics:
        print("\n" + "="*80)
        print("GENERATING ADVANCED METRICS (7 total)")
        print("="*80 + "\n")
        
        print(f"{plot_num}. TM-score distribution...")
        plot_tm_score_distribution(data.oscillators,
                                   output_dir / 'tm_score.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. GDT-TS distribution...")
        plot_gdt_ts_distribution(data.oscillators,
                                output_dir / 'gdt_ts.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. lDDT distribution...")
        plot_lddt_distribution(data.oscillators,
                              output_dir / 'lddt.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Contact map comparison...")
        plot_contact_map_comparison(data.oscillators,
                                   output_dir / 'contact_map.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Ramachandran quality assessment...")
        if has_rama:
            plot_ramachandran_quality(gt_angles, pred_angles,
                                     output_dir / 'rama_quality.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Clash analysis...")
        plot_clash_analysis(data.oscillators,
                           output_dir / 'clash_analysis.png', dpi=args.dpi)
        plot_num += 1
"""


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete backmapping analysis with all metrics and plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python plot_backmapping_complete.py --root results --splits val test --output-dir figures
  
  # High-res for publication
  python plot_backmapping_complete.py --root results --output-dir pub_figures --dpi 600 --font-size 10
  
  # Include advanced metrics
  python plot_backmapping_complete.py --root results --output-dir figures --advanced-metrics
        """
    )
    
    parser.add_argument('--root', type=str, required=True,
                       help='Root directory containing split folders')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       help='Split folders to analyze (default: val test)')
    parser.add_argument('--pattern', type=str, default='*.pkl',
                       help='Glob pattern for pickle files (default: *.pkl)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI (default: 300)')
    parser.add_argument('--font-size', type=int, default=10,
                       help='Base font size - ALL plots use this (default: 10)')
    parser.add_argument('--exclude-zero-angles', action='store_true', default=True,
                       help='Exclude terminal residue angles (default: True)')
    parser.add_argument('--advanced-metrics', action='store_true',
                       help='Compute advanced metrics (TM-score, etc.)')
    
    args = parser.parse_args()
    
    # Set style with consistent font sizing
    set_publication_style(base_font_size=args.font_size)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPLETE BACKMAPPING ANALYSIS SUITE")
    print("="*80)
    print(f"Font size: {args.font_size} (used throughout all plots)")
    print(f"DPI: {args.dpi}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {args.root}")
    print(f"Splits: {args.splits}")
    
    data = load_pickle_files(Path(args.root), args.splits, args.pattern)
    
    print(f"\n✓ Loaded {data.n_oscillators:,} oscillators from {data.n_files} files")
    
    # Extract angles
    print("\nExtracting Ramachandran angles...")
    gt_angles = extract_rama_angles(data.oscillators, 'rama_nnfs', 
                                    exclude_zero=args.exclude_zero_angles)
    pred_angles = extract_rama_angles(data.oscillators, 'predicted_rama_nnfs',
                                     exclude_zero=args.exclude_zero_angles)
    
    has_rama = all(len(a) > 0 for a in gt_angles) and all(len(a) > 0 for a in pred_angles)
    
    print(f"\nData availability:")
    print(f"  Ramachandran angles: {'✓' if has_rama else '✗'}")
    
    # Generate core plots
    print("\n" + "="*80)
    print("GENERATING CORE PLOTS (8)")
    print("="*80 + "\n")
    
    plot_num = 1
    
    if has_rama:
        print(f"{plot_num}. Ramachandran scatter (4-panel density)...")
        plot_ramachandran_scatter(gt_angles, pred_angles,
                                 output_dir / 'rama_scatter.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Ramachandran distributions (overlaid histograms)...")
        plot_ramachandran_distributions(gt_angles, pred_angles,
                                       output_dir / 'rama_distributions.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Ramachandran maps (φ vs ψ, 4-panel)...")
        plot_ramachandran_maps(gt_angles, pred_angles,
                              output_dir / 'rama_maps.png', dpi=args.dpi)
        plot_num += 1
        
        plot_ramachandran_maps_contour(gt_angles, pred_angles,
                              output_dir / 'rama_maps_contour.png', dpi=args.dpi)
        plot_num += 1
    
    print(f"{plot_num}. Bond length distributions...")
    plot_bond_length_distributions(data.oscillators,
                                   output_dir / 'bond_lengths.png', dpi=args.dpi)
    plot_num += 1
    
    print(f"{plot_num}. Dipole orientation analysis...")
    plot_dipole_analysis(data.oscillators,
                        output_dir / 'dipole_analysis.png', dpi=args.dpi)
    plot_num += 1
    
    plot_dipole_analysis_linear(data.oscillators,
                        output_dir / 'dipole_analysis_linear.png', dpi=args.dpi)
    plot_num += 1 
    
    print(f"{plot_num}. Dipole component correlations...")
    plot_dipole_components(data.oscillators,
                          output_dir / 'dipole_components.png', dpi=args.dpi)
    plot_num += 1
    
    print(f"{plot_num}. Per-frame all-atom RMSD...")
    plot_rmsd_per_frame(data.oscillators,
                       output_dir / 'rmsd_per_frame.png', dpi=args.dpi)
    plot_num += 1
    
    # Advanced metrics        
    if args.advanced_metrics:
        print("\n" + "="*80)
        print("GENERATING ADVANCED METRICS (7 total)")
        print("="*80 + "\n")
        
        print(f"{plot_num}. TM-score distribution...")
        plot_tm_score_distribution(data.oscillators,
                                   output_dir / 'tm_score.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. GDT-TS distribution...")
        plot_gdt_ts_distribution(data.oscillators,
                                output_dir / 'gdt_ts.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. lDDT distribution...")
        plot_lddt_distribution(data.oscillators,
                              output_dir / 'lddt.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Contact map comparison...")
        plot_contact_map_comparison(data.oscillators,
                                   output_dir / 'contact_map.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Ramachandran quality assessment...")
        if has_rama:
            plot_ramachandran_quality(gt_angles, pred_angles,
                                     output_dir / 'rama_quality.png', dpi=args.dpi)
        plot_num += 1
        
        print(f"{plot_num}. Clash analysis...")
        plot_clash_analysis(data.oscillators,
                           output_dir / 'clash_analysis.png', dpi=args.dpi)
        plot_num += 1

    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\n✓ All plots saved to: {output_dir}")
    print(f"✓ Font size {args.font_size} used consistently throughout\n")


if __name__ == '__main__':
    main()
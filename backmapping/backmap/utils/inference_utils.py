"""
Shared Inference Utilities

Common functions used by both robust and fast inference scripts:
- Ramachandran angle calculation (MDAnalysis-consistent)
- Plotting functions
- Result aggregation
- File I/O utilities

This avoids code duplication between infer_robust.py and infer_fast.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from MDAnalysis.lib.distances import calc_dihedrals
    HAS_MDANALYSIS = True
except ImportError:
    calc_dihedrals = None
    HAS_MDANALYSIS = False


# ============================================================================
# Ramachandran Angle Calculation (MDAnalysis-consistent)
# ============================================================================

def compute_phi_psi_from_oscillators(
    backbone_oscillators: List[Dict[str, Any]]
) -> Tuple[Dict[int, float], Dict[int, float], Optional[int], Optional[int]]:
    """
    Compute φ/ψ angles for all residues from backbone oscillators.
    
    Uses MDAnalysis-consistent dihedral definitions:
        φ(i) = dihedral(C_{i-1}, N_i, CA_i, C_i)
        ψ(i) = dihedral(N_i, CA_i, C_i, N_{i+1})
    
    Args:
        backbone_oscillators: List of backbone oscillator dicts
    
    Returns:
        phi_by_resid: Dict mapping resid → φ angle (degrees)
        psi_by_resid: Dict mapping resid → ψ angle (degrees)
        first_resid: First residue ID
        last_resid: Last residue ID
    """
    if not HAS_MDANALYSIS:
        return {}, {}, None, None
    
    # Build bond map: resid → oscillator
    bond_by_resid = {}
    all_resids = set()
    
    for osc in backbone_oscillators:
        bb_curr_key = osc.get("bb_curr_key")
        bb_next_key = osc.get("bb_next_key")
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
        
        atoms_i = osc_i.get("atoms", {})
        atoms_im1 = osc_im1.get("atoms", {})
        
        try:
            C_im1 = np.asarray(atoms_im1["C_prev"], dtype=np.float64)
            N_i = np.asarray(atoms_i["N_prev"], dtype=np.float64)
            CA_i = np.asarray(atoms_i["CA_prev"], dtype=np.float64)
            C_i = np.asarray(atoms_i["C_prev"], dtype=np.float64)
        except KeyError:
            continue
        
        if any(np.allclose(atom, 0.0) for atom in [C_im1, N_i, CA_i, C_i]):
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
        
        atoms_i = osc_i.get("atoms", {})
        try:
            N_i = np.asarray(atoms_i["N_prev"], dtype=np.float64)
            CA_i = np.asarray(atoms_i["CA_prev"], dtype=np.float64)
            C_i = np.asarray(atoms_i["C_prev"], dtype=np.float64)
            N_ip1 = np.asarray(atoms_i["N_curr"], dtype=np.float64)
        except KeyError:
            continue
        
        if any(np.allclose(atom, 0.0) for atom in [N_i, CA_i, C_i, N_ip1]):
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
    
    # Boundary convention: φ=ψ=0 for first and last residue
    if first_resid is not None:
        phi_by_resid[first_resid] = 0.0
        psi_by_resid[first_resid] = 0.0
    if last_resid is not None and last_resid != first_resid:
        phi_by_resid[last_resid] = 0.0
        psi_by_resid[last_resid] = 0.0
    
    return phi_by_resid, psi_by_resid, first_resid, last_resid


def compute_nnfs_angles_for_oscillator(
    osc: Dict[str, Any],
    phi_by_resid: Dict[int, float],
    psi_by_resid: Dict[int, float],
) -> Dict[str, Optional[float]]:
    """
    Compute NNFS angles for a single oscillator.
    
    NNFS convention:
        phi_N = φ(i)
        psi_N = ψ(i-1)
        phi_C = φ(i+1)
        psi_C = ψ(i)
    
    where i is the oscillator's residue_key[0].
    """
    res_key = osc.get("residue_key")
    if not res_key:
        return {"phi_N": None, "psi_N": None, "phi_C": None, "psi_C": None}
    
    resid = int(res_key[0])
    
    return {
        "phi_N": phi_by_resid.get(resid),
        "psi_N": psi_by_resid.get(resid - 1),
        "phi_C": phi_by_resid.get(resid + 1),
        "psi_C": psi_by_resid.get(resid),
    }


def add_predicted_rama_angles_to_oscillators(
    oscillators: List[Dict[str, Any]],
    predicted_atoms: Dict[str, Any],
) -> None:
    """
    Add predicted Ramachandran angles to oscillators in-place.
    
    This reconstructs oscillators with predicted atom coordinates,
    then calculates φ/ψ and NNFS angles.
    
    Args:
        oscillators: List of oscillator dicts (modified in-place)
        predicted_atoms: Dict from aggregate_predicted_from_oscillator_predictions
    """
    if not HAS_MDANALYSIS:
        for osc in oscillators:
            osc["predicted_rama_nnfs"] = {
                "phi_N": None,
                "psi_N": None,
                "phi_C": None,
                "psi_C": None,
            }
        return
    
    # Reconstruct backbone oscillators with predicted coordinates
    predicted_backbone_oscs = []
    
    for osc in oscillators:
        if osc.get("oscillator_type") != "backbone":
            osc["predicted_rama_nnfs"] = {
                "phi_N": None,
                "psi_N": None,
                "phi_C": None,
                "psi_C": None,
            }
            continue
        
        bb_curr_key = osc.get("bb_curr_key")
        bb_next_key = osc.get("bb_next_key")
        
        if not bb_curr_key or not bb_next_key:
            osc["predicted_rama_nnfs"] = {
                "phi_N": None,
                "psi_N": None,
                "phi_C": None,
                "psi_C": None,
            }
            continue
        
        # Build predicted atom dict with CORRECT naming convention
        # predicted_atoms structure: {(resid, resname): {'CA': coord, 'C': coord, ...}}
        # We need to map to: C_prev, O_prev, CA_prev, N_prev (from bb_curr_key)
        #                    N_curr, H_curr, CA_curr (from bb_next_key)
        
        pred_atoms = {}
        
        # Get atoms from bb_curr (residue i) → map to _prev
        if bb_curr_key in predicted_atoms:
            curr_atoms = predicted_atoms[bb_curr_key]
            if 'C' in curr_atoms:
                pred_atoms['C_prev'] = curr_atoms['C']
            if 'O' in curr_atoms:
                pred_atoms['O_prev'] = curr_atoms['O']
            if 'CA' in curr_atoms:
                pred_atoms['CA_prev'] = curr_atoms['CA']
            if 'N' in curr_atoms:
                pred_atoms['N_prev'] = curr_atoms['N']
        
        # Get atoms from bb_next (residue i+1) → map to _curr
        if bb_next_key in predicted_atoms:
            next_atoms = predicted_atoms[bb_next_key]
            if 'N' in next_atoms:
                pred_atoms['N_curr'] = next_atoms['N']
            if 'H' in next_atoms:
                pred_atoms['H_curr'] = next_atoms['H']
            if 'CA' in next_atoms:
                pred_atoms['CA_curr'] = next_atoms['CA']
        
        # Create temporary oscillator with predicted atoms
        pred_osc = {
            "bb_curr_key": bb_curr_key,
            "bb_next_key": bb_next_key,
            "atoms": pred_atoms,
        }
        predicted_backbone_oscs.append((osc, pred_osc))
    
    # Compute φ/ψ from predicted coordinates
    pred_phi, pred_psi, _, _ = compute_phi_psi_from_oscillators(
        [po for _, po in predicted_backbone_oscs]
    )
    
    # Add NNFS angles to original oscillators
    for orig_osc, pred_osc in predicted_backbone_oscs:
        nnfs = compute_nnfs_angles_for_oscillator(orig_osc, pred_phi, pred_psi)
        orig_osc["predicted_rama_nnfs"] = nnfs


def add_ground_truth_rama_angles_to_oscillators(
    oscillators: List[Dict[str, Any]]
) -> None:
    """
    Add ground truth Ramachandran angles to oscillators in-place.
    
    This calculates φ/ψ and NNFS angles from the ground truth atoms
    already present in the oscillators.
    
    Args:
        oscillators: List of oscillator dicts (modified in-place)
    """
    if not HAS_MDANALYSIS:
        for osc in oscillators:
            osc["rama_nnfs"] = {
                "phi_N": None,
                "psi_N": None,
                "phi_C": None,
                "psi_C": None,
            }
        return
    
    # Filter backbone oscillators
    backbone_oscs = [osc for osc in oscillators if osc.get("oscillator_type") == "backbone"]
    
    if not backbone_oscs:
        return
    
    # Compute φ/ψ from ground truth coordinates
    gt_phi, gt_psi, _, _ = compute_phi_psi_from_oscillators(backbone_oscs)
    
    # Add NNFS angles to all oscillators
    for osc in oscillators:
        if osc.get("oscillator_type") == "backbone":
            nnfs = compute_nnfs_angles_for_oscillator(osc, gt_phi, gt_psi)
            osc["rama_nnfs"] = nnfs
        else:
            # Sidechain oscillators have no NNFS angles
            osc["rama_nnfs"] = {
                "phi_N": None,
                "psi_N": None,
                "phi_C": None,
                "psi_C": None,
            }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_ramachandran_comparison(
    ground_truth_angles: List[Dict[str, Optional[float]]],
    predicted_angles: List[Dict[str, Optional[float]]],
    output_file: str,
    title_prefix: str = "",
):
    """
    Plot Ramachandran angle comparisons.
    
    Creates two plots:
    1. Scatter plots (pred vs true) for each angle
    2. Distribution overlays for each angle
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("WARNING: matplotlib or scipy not found. Skipping plots.")
        return
    
    angle_names = ["phi_N", "psi_N", "phi_C", "psi_C"]
    angle_labels = {
        "phi_N": "φ_N (N-side φ)",
        "psi_N": "ψ_N (N-side ψ)",
        "phi_C": "φ_C (C-side φ)",
        "psi_C": "ψ_C (C-side ψ)",
    }
    
    # Extract angle arrays
    angle_data = {}
    for angle_name in angle_names:
        gt_vals = []
        pred_vals = []
        for gt, pred in zip(ground_truth_angles, predicted_angles):
            gt_angle = gt.get(angle_name)
            pred_angle = pred.get(angle_name)
            if gt_angle is not None and pred_angle is not None:
                gt_vals.append(gt_angle)
                pred_vals.append(pred_angle)
        
        angle_data[angle_name] = {
            "gt": np.array(gt_vals),
            "pred": np.array(pred_vals),
        }
    
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, angle_name in enumerate(angle_names):
        ax = axes[idx]
        gt = angle_data[angle_name]["gt"]
        pred = angle_data[angle_name]["pred"]
        
        if len(gt) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(angle_labels[angle_name])
            continue
        
        ax.scatter(gt, pred, alpha=0.3, s=10, c="steelblue", edgecolors="none")
        
        # Identity line
        lims = [min(gt.min(), pred.min()) - 10, max(gt.max(), pred.max()) + 10]
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1.5, label="y=x")
        
        # Statistics
        error = pred - gt
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        r2 = np.corrcoef(gt, pred)[0, 1]**2 if len(gt) > 1 else 0
        
        stats_text = f"R²={r2:.3f}\nRMSE={rmse:.2f}°\nMAE={mae:.2f}°\nn={len(gt)}"
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )
        
        ax.set_xlabel(f"Ground Truth {angle_labels[angle_name]} (°)")
        ax.set_ylabel(f"Predicted {angle_labels[angle_name]} (°)")
        ax.set_title(angle_labels[angle_name])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Distribution overlays
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, angle_name in enumerate(angle_names):
        ax = axes[idx]
        gt = angle_data[angle_name]["gt"]
        pred = angle_data[angle_name]["pred"]
        
        if len(gt) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(angle_labels[angle_name])
            continue
        
        bins = np.linspace(-180, 180, 73)
        ax.hist(gt, bins=bins, alpha=0.5, label="Ground Truth", color="blue", density=True)
        ax.hist(pred, bins=bins, alpha=0.5, label="Predicted", color="red", density=True)
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(gt, pred)
        ax.text(
            0.05, 0.95,
            f"KS stat: {ks_stat:.4f}\nKS p: {ks_p:.4f}\nn={len(gt)}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )
        
        ax.set_xlabel(f"{angle_labels[angle_name]} (°)")
        ax.set_ylabel("Density")
        ax.set_title(angle_labels[angle_name])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved Ramachandran plots: {output_file}")


def plot_ramachandran_prediction_only(
    predicted_angles: List[Dict[str, Optional[float]]],
    output_file: str,
    title_prefix: str = "",
):
    """Plot Ramachandran distributions for predicted angles (no ground truth)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not found. Skipping plots.")
        return
    
    angle_names = ["phi_N", "psi_N", "phi_C", "psi_C"]
    angle_labels = {
        "phi_N": "φ_N (N-side φ)",
        "psi_N": "ψ_N (N-side ψ)",
        "phi_C": "φ_C (C-side φ)",
        "psi_C": "ψ_C (C-side ψ)",
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, angle_name in enumerate(angle_names):
        ax = axes[idx]
        
        pred_vals = []
        for pred in predicted_angles:
            angle = pred.get(angle_name)
            if angle is not None:
                pred_vals.append(angle)
        
        pred_vals = np.array(pred_vals)
        
        if len(pred_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(angle_labels[angle_name])
            continue
        
        bins = np.linspace(-180, 180, 73)
        ax.hist(pred_vals, bins=bins, alpha=0.7, color="steelblue", density=True)
        
        ax.text(
            0.05, 0.95,
            f"n={len(pred_vals)}\nmean={pred_vals.mean():.1f}°\nstd={pred_vals.std():.1f}°",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )
        
        ax.set_xlabel(f"{angle_labels[angle_name]} (°)")
        ax.set_ylabel("Density")
        ax.set_title(angle_labels[angle_name])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved Ramachandran plots: {output_file}")

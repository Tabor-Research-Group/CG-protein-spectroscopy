"""
Publication-Quality Plotting Module for CG→Atomistic Backmapping

This module provides comprehensive visualization functions for evaluating diffusion model
performance with publication-ready formatting.

Key Improvements:
1. Overlay histograms for direct comparison (not separate plots)
2. Scatter plots with y=x identity line and R² statistics
3. Uses ALL validation/test data (not just few batches)
4. 300+ DPI resolution with proper fonts and styling
5. Statistical reporting (KS test, RMSE, correlation)
6. Multi-panel figures with shared formatting

Author: Generated for scientific publication
Date: December 2024
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
from scipy import stats

# ============================================================================
# Publication-Quality Plot Styling
# ============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        
        # Axes and labels
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Figure
        'figure.dpi': 150,  # Display DPI
        'figure.constrained_layout.use': True,
        
        # Saving
        'savefig.dpi': 300,  # Publication DPI
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_array(x) -> np.ndarray:
    """Convert tensor/list to numpy array."""
    if x is None:
        return np.array([])
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    return arr[np.isfinite(arr)]  # Remove NaN/Inf


def compute_statistics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for model evaluation.
    
    Returns:
        Dictionary with keys: rmse, mae, r2, pearson_r, pearson_p, 
                              ks_stat, ks_p, mean_error, std_error
    """
    if len(true_vals) == 0 or len(pred_vals) == 0:
        return {}
    
    # Error metrics
    error = pred_vals - true_vals
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    mean_error = np.mean(error)
    std_error = np.std(error)
    
    # Correlation
    try:
        pearson_r, pearson_p = stats.pearsonr(true_vals, pred_vals)
        r2 = pearson_r**2
    except:
        pearson_r, pearson_p, r2 = np.nan, np.nan, np.nan
    
    # Distribution comparison
    try:
        ks_stat, ks_p = stats.ks_2samp(true_vals, pred_vals)
    except:
        ks_stat, ks_p = np.nan, np.nan
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'n_samples': int(len(true_vals)),
    }


def format_stat_text(stats_dict: Dict[str, float], metrics: List[str] = None) -> str:
    """
    Format statistics dictionary as text annotation for plots.
    
    Args:
        stats_dict: Dictionary from compute_statistics()
        metrics: List of metric keys to include (default: ['r2', 'rmse', 'n_samples'])
    """
    if metrics is None:
        metrics = ['r2', 'rmse', 'n_samples']
    
    lines = []
    for key in metrics:
        if key not in stats_dict:
            continue
        val = stats_dict[key]
        
        if key == 'r2':
            lines.append(f'$R^2$ = {val:.4f}')
        elif key == 'rmse':
            lines.append(f'RMSE = {val:.4f}')
        elif key == 'mae':
            lines.append(f'MAE = {val:.4f}')
        elif key == 'pearson_r':
            lines.append(f'$r$ = {val:.4f}')
        elif key == 'ks_stat':
            lines.append(f'KS = {val:.4f}')
        elif key == 'ks_p':
            lines.append(f'$p$ = {val:.2e}')
        elif key == 'n_samples':
            lines.append(f'$n$ = {int(val)}')
        elif key == 'mean_error':
            lines.append(f'$\\mu_{{err}}$ = {val:.4f}')
        elif key == 'std_error':
            lines.append(f'$\\sigma_{{err}}$ = {val:.4f}')
    
    return '\n'.join(lines)


# ============================================================================
# Core Plotting Functions
# ============================================================================

def plot_scatter_identity(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    stats_metrics: List[str] = ['r2', 'rmse', 'n_samples'],
    color: str = 'steelblue',
    alpha: float = 0.3,
    s: float = 2,
) -> Dict[str, float]:
    """
    Create scatter plot with y=x identity line and statistics.
    
    Args:
        true_vals: Ground truth values
        pred_vals: Predicted values
        xlabel, ylabel, title: Plot labels
        output_path: Save path
        stats_metrics: Which statistics to display
        color, alpha, s: Scatter plot styling
        
    Returns:
        Dictionary of computed statistics
    """
    setup_publication_style()
    
    true_vals = ensure_array(true_vals)
    pred_vals = ensure_array(pred_vals)
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print(f"Warning: Empty data for {output_path}, skipping...")
        return {}
    
    # Compute statistics
    stats_dict = compute_statistics(true_vals, pred_vals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Scatter plot
    ax.scatter(true_vals, pred_vals, s=s, alpha=alpha, color=color, rasterized=True)
    
    # Identity line (y=x)
    lims = [
        np.min([true_vals.min(), pred_vals.min()]),
        np.max([true_vals.max(), pred_vals.max()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.6, lw=1.5, label='$y=x$', zorder=10)
    
    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stat_text = format_stat_text(stats_dict, stats_metrics)
    ax.text(0.95, 0.05, stat_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', 
                     edgecolor='gray', alpha=0.9))
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_dict


def plot_overlay_histograms(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Path,
    bins: int = 50,
    density: bool = True,
    show_stats: bool = True,
) -> Dict[str, float]:
    """
    Create overlaid histograms for direct distribution comparison.
    
    Args:
        true_vals: Ground truth values
        pred_vals: Predicted values
        xlabel, title: Plot labels
        output_path: Save path
        bins: Number of histogram bins
        density: Normalize to probability density
        show_stats: Include KS test statistics
        
    Returns:
        Dictionary of computed statistics
    """
    setup_publication_style()
    
    true_vals = ensure_array(true_vals)
    pred_vals = ensure_array(pred_vals)
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print(f"Warning: Empty data for {output_path}, skipping...")
        return {}
    
    # Compute statistics
    stats_dict = compute_statistics(true_vals, pred_vals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histograms with transparency
    ax.hist(true_vals, bins=bins, alpha=0.6, label='Ground Truth', 
            color='#2E86C1', density=density, edgecolor='black', linewidth=0.5)
    ax.hist(pred_vals, bins=bins, alpha=0.6, label='Predicted', 
            color='#E74C3C', density=density, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density' if density else 'Count')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add KS statistic
    if show_stats and 'ks_stat' in stats_dict:
        stat_text = format_stat_text(stats_dict, ['ks_stat', 'ks_p', 'n_samples'])
        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor='gray', alpha=0.9))
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_dict


def plot_error_histogram(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Path,
    bins: int = 50,
    show_gaussian: bool = True,
) -> Dict[str, float]:
    """
    Plot histogram of prediction errors (pred - true).
    
    Args:
        true_vals: Ground truth values
        pred_vals: Predicted values
        xlabel: X-axis label for error
        title: Plot title
        output_path: Save path
        bins: Number of histogram bins
        show_gaussian: Overlay Gaussian fit
        
    Returns:
        Dictionary of error statistics
    """
    setup_publication_style()
    
    true_vals = ensure_array(true_vals)
    pred_vals = ensure_array(pred_vals)
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print(f"Warning: Empty data for {output_path}, skipping...")
        return {}
    
    # Compute errors
    errors = pred_vals - true_vals
    
    # Statistics
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    median_err = np.median(errors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histogram
    counts, edges, patches = ax.hist(errors, bins=bins, alpha=0.7, 
                                     color='#8E44AD', edgecolor='black', 
                                     linewidth=0.5, density=True)
    
    # Gaussian fit overlay
    if show_gaussian:
        x = np.linspace(errors.min(), errors.max(), 200)
        gaussian = stats.norm.pdf(x, mean_err, std_err)
        ax.plot(x, gaussian, 'r--', lw=2, label=f'$\\mathcal{{N}}({mean_err:.3f}, {std_err:.3f})$')
    
    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero error')
    
    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Statistics text
    stat_text = f'Mean = {mean_err:.4f}\nStd = {std_err:.4f}\nMedian = {median_err:.4f}'
    ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', 
                     edgecolor='gray', alpha=0.9))
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_error': float(mean_err),
        'std_error': float(std_err),
        'median_error': float(median_err),
        'n_samples': int(len(errors)),
    }


def plot_ramachandran(
    phi_true: np.ndarray,
    psi_true: np.ndarray,
    phi_pred: np.ndarray,
    psi_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """
    Create Ramachandran plot comparing true vs. predicted φ/ψ angles.
    
    Args:
        phi_true, psi_true: Ground truth Ramachandran angles (degrees)
        phi_pred, psi_pred: Predicted Ramachandran angles (degrees)
        title: Plot title
        output_path: Save path
    """
    setup_publication_style()
    
    phi_true = ensure_array(phi_true)
    psi_true = ensure_array(psi_true)
    phi_pred = ensure_array(phi_pred)
    psi_pred = ensure_array(psi_pred)
    
    if len(phi_true) == 0:
        print(f"Warning: Empty data for {output_path}, skipping...")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot true (blue) and predicted (red) with transparency
    ax.scatter(phi_true, psi_true, s=3, alpha=0.4, color='#2E86C1', 
               label='Ground Truth', rasterized=True)
    ax.scatter(phi_pred, psi_pred, s=3, alpha=0.4, color='#E74C3C', 
               label='Predicted', rasterized=True)
    
    # Ramachandran boundaries (approximate)
    # Right-handed α-helix region
    alpha_helix = patches.Rectangle((-80, -60), 40, 60, linewidth=1, 
                                   edgecolor='green', facecolor='none', 
                                   linestyle='--', label='α-helix')
    ax.add_patch(alpha_helix)
    
    # β-sheet region
    beta_sheet = patches.Rectangle((-160, 100), 80, 60, linewidth=1, 
                                   edgecolor='orange', facecolor='none', 
                                   linestyle='--', label='β-sheet')
    ax.add_patch(beta_sheet)
    
    # Formatting
    ax.set_xlabel('φ (degrees)')
    ax.set_ylabel('ψ (degrees)')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal grid lines
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    epoch_data: List[Dict],
    loss_keys: List[str],
    output_path: Path,
    title: str = "Training Loss Curves",
    splits: List[str] = ['train', 'val', 'test'],
) -> None:
    """
    Plot loss curves over epochs for multiple loss components and splits.
    
    Args:
        epoch_data: List of dictionaries, one per epoch, with keys:
                    {'epoch': int, 'split': str, 'losses': {...}}
        loss_keys: Which loss components to plot
        output_path: Save path
        title: Main title
        splits: Which data splits to include
    """
    setup_publication_style()
    
    # Organize data by split and loss
    data_by_split = {split: {key: [] for key in loss_keys} for split in splits}
    epochs_by_split = {split: [] for split in splits}
    
    for record in epoch_data:
        split = record.get('split')
        if split not in splits:
            continue
        epoch = record.get('epoch')
        losses = record.get('losses', {})
        
        epochs_by_split[split].append(epoch)
        for key in loss_keys:
            if key in losses:
                data_by_split[split][key].append(losses[key])
    
    # Create subplots (one per loss component)
    n_losses = len(loss_keys)
    fig, axes = plt.subplots(n_losses, 1, figsize=(8, 3*n_losses), sharex=True)
    
    if n_losses == 1:
        axes = [axes]
    
    colors = {'train': '#2E86C1', 'val': '#E67E22', 'test': '#E74C3C'}
    
    for idx, loss_key in enumerate(loss_keys):
        ax = axes[idx]
        
        for split in splits:
            epochs = epochs_by_split[split]
            values = data_by_split[split][loss_key]
            
            if len(epochs) > 0 and len(values) > 0:
                ax.plot(epochs, values, label=split.capitalize(), 
                       color=colors.get(split, 'gray'), linewidth=2, alpha=0.8)
        
        ax.set_ylabel(loss_key.replace('_', ' ').title())
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log' if 'total' in loss_key else 'linear')
    
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Multi-Panel Figure Generation
# ============================================================================

def create_figure_geometric_accuracy(
    bond_true: np.ndarray,
    bond_pred: np.ndarray,
    angle_true: np.ndarray,
    angle_pred: np.ndarray,
    output_path: Path,
    epoch: int,
    split: str,
) -> Dict[str, Dict[str, float]]:
    """
    Create 2x2 multi-panel figure for geometric accuracy.
    
    Panels:
    (A) Bond length scatter
    (B) Bond length error histogram
    (C) Bond angle scatter
    (D) Bond angle error histogram
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    stats = {}
    
    # (A) Bond length scatter
    ax1 = fig.add_subplot(gs[0, 0])
    bond_true_arr = ensure_array(bond_true)
    bond_pred_arr = ensure_array(bond_pred)
    
    if len(bond_true_arr) > 0:
        stats['bond_scatter'] = compute_statistics(bond_true_arr, bond_pred_arr)
        ax1.scatter(bond_true_arr, bond_pred_arr, s=2, alpha=0.3, color='steelblue', rasterized=True)
        lims = [min(bond_true_arr.min(), bond_pred_arr.min()),
                max(bond_true_arr.max(), bond_pred_arr.max())]
        ax1.plot(lims, lims, 'k--', lw=1.5, alpha=0.6)
        ax1.set_xlabel('True Bond Length (Å)')
        ax1.set_ylabel('Predicted Bond Length (Å)')
        ax1.set_title('(A) Bond Length Accuracy', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        stat_text = format_stat_text(stats['bond_scatter'], ['r2', 'rmse'])
        ax1.text(0.95, 0.05, stat_text, transform=ax1.transAxes,
                va='bottom', ha='right', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.9))
    
    # (B) Bond length error
    ax2 = fig.add_subplot(gs[0, 1])
    if len(bond_true_arr) > 0:
        errors = bond_pred_arr - bond_true_arr
        ax2.hist(errors, bins=50, alpha=0.7, color='#8E44AD', edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_xlabel('Prediction Error (Å)')
        ax2.set_ylabel('Count')
        ax2.set_title('(B) Bond Length Error Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        ax2.text(0.98, 0.98, f'μ = {mean_err:.4f} Å\nσ = {std_err:.4f} Å',
                transform=ax2.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # (C) Bond angle scatter
    ax3 = fig.add_subplot(gs[1, 0])
    angle_true_arr = ensure_array(angle_true)
    angle_pred_arr = ensure_array(angle_pred)
    
    if len(angle_true_arr) > 0:
        # Convert to degrees if in radians
        if angle_true_arr.max() < 10:  # Likely radians
            angle_true_arr = np.degrees(angle_true_arr)
            angle_pred_arr = np.degrees(angle_pred_arr)
        
        stats['angle_scatter'] = compute_statistics(angle_true_arr, angle_pred_arr)
        ax3.scatter(angle_true_arr, angle_pred_arr, s=2, alpha=0.3, color='darkorange', rasterized=True)
        lims = [min(angle_true_arr.min(), angle_pred_arr.min()),
                max(angle_true_arr.max(), angle_pred_arr.max())]
        ax3.plot(lims, lims, 'k--', lw=1.5, alpha=0.6)
        ax3.set_xlabel('True Angle (degrees)')
        ax3.set_ylabel('Predicted Angle (degrees)')
        ax3.set_title('(C) Bond Angle Accuracy', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        stat_text = format_stat_text(stats['angle_scatter'], ['r2', 'rmse'])
        ax3.text(0.95, 0.05, stat_text, transform=ax3.transAxes,
                va='bottom', ha='right', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.9))
    
    # (D) Bond angle error
    ax4 = fig.add_subplot(gs[1, 1])
    if len(angle_true_arr) > 0:
        errors = angle_pred_arr - angle_true_arr
        ax4.hist(errors, bins=50, alpha=0.7, color='#16A085', edgecolor='black', linewidth=0.5)
        ax4.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.set_xlabel('Prediction Error (degrees)')
        ax4.set_ylabel('Count')
        ax4.set_title('(D) Angle Error Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        ax4.text(0.98, 0.98, f'μ = {mean_err:.4f}°\nσ = {std_err:.4f}°',
                transform=ax4.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Overall title
    fig.suptitle(f'Geometric Accuracy | {split.upper()} | Epoch {epoch}',
                fontsize=14, fontweight='bold')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats


# ============================================================================
# Main Epoch Plotting Function (Replacement for existing)
# ============================================================================

def plot_epoch_metrics(
    out_dir: Path,
    epoch: int,
    split: str,
    losses: Dict[str, float],
    metrics: Dict[str, np.ndarray],
    cfg=None,          
    plot_cfg=None, 
    bins: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Create complete set of publication-quality diagnostic plots for one epoch.
    
    This function REPLACES the existing plot_epoch_metrics() with improved versions.
    
    Args:
        out_dir: Output directory
        epoch: Current epoch number
        split: Data split ('train', 'val', or 'test')
        losses: Dictionary of scalar loss values
        metrics: Dictionary of metric arrays (all data, not just batches!)
        bins: Number of histogram bins
        
    Returns:
        Dictionary of computed statistics for all metrics
    """
    setup_publication_style()
    
    base_dir = Path(out_dir) / "plots" / f"epoch_{epoch:04d}" / split
    base_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    # ========================================================================
    # 1. Geometric Accuracy Multi-Panel Figure
    # ========================================================================
    bond_true = ensure_array(metrics.get('bond_true', []))
    bond_pred = ensure_array(metrics.get('bond_pred', []))
    angle_true = ensure_array(metrics.get('angle_true', []))
    angle_pred = ensure_array(metrics.get('angle_pred', []))
    
    if len(bond_true) > 0 or len(angle_true) > 0:
        geom_stats = create_figure_geometric_accuracy(
            bond_true, bond_pred, angle_true, angle_pred,
            base_dir / 'geometric_accuracy_multipanel.png',
            epoch, split
        )
        all_stats.update(geom_stats)
    
    # ========================================================================
    # 2. Dipole Correlation
    # ========================================================================
    dipole_cos = ensure_array(metrics.get('dipole_cos', []))
    if len(dipole_cos) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dipole_cos, bins=bins, alpha=0.7, color='#E67E22', 
               edgecolor='black', linewidth=0.5)
        ax.axvline(dipole_cos.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean = {dipole_cos.mean():.4f}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title(f'Dipole Alignment | {split.upper()} | Epoch {epoch}', 
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(base_dir / 'dipole_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        all_stats['dipole'] = {
            'mean': float(dipole_cos.mean()),
            'std': float(dipole_cos.std()),
            'median': float(np.median(dipole_cos)),
        }
    
    # ========================================================================
    # 3. Radial Distance Overlay
    # ========================================================================
    r_true = ensure_array(metrics.get('radial_true', []))
    r_pred = ensure_array(metrics.get('radial_pred', []))
    if len(r_true) > 0 and len(r_pred) > 0:
        rad_stats = plot_overlay_histograms(
            r_true, r_pred,
            xlabel='Radial Distance from BB Bead (Å)',
            title=f'Local Radial Distance | {split.upper()} | Epoch {epoch}',
            output_path=base_dir / 'radial_distance_overlay.png',
            bins=bins
        )
        all_stats['radial'] = rad_stats
    
    # ========================================================================
    # 4. Dihedral Angles Overlay
    # ========================================================================
    dih_true = ensure_array(metrics.get('dihedral_true', []))
    dih_pred = ensure_array(metrics.get('dihedral_pred', []))
    if len(dih_true) > 0 and len(dih_pred) > 0:
        # Convert to degrees if needed
        if dih_true.max() < 10:
            dih_true = np.degrees(dih_true)
            dih_pred = np.degrees(dih_pred)
        
        dih_stats = plot_overlay_histograms(
            dih_true, dih_pred,
            xlabel='Dihedral Angle (degrees)',
            title=f'Dihedral Angles | {split.upper()} | Epoch {epoch}',
            output_path=base_dir / 'dihedral_overlay.png',
            bins=bins
        )
        all_stats['dihedral'] = dih_stats
    
    # ========================================================================
    # 5. Non-bonded Distances
    # ========================================================================
    nb_true = ensure_array(metrics.get('nonbond_min_true', []))
    nb_pred = ensure_array(metrics.get('nonbond_min_pred', []))
    if len(nb_true) > 0 and len(nb_pred) > 0:
        nb_stats = plot_overlay_histograms(
            nb_true, nb_pred,
            xlabel='Minimum Non-bonded Distance (Å)',
            title=f'Steric Quality | {split.upper()} | Epoch {epoch}',
            output_path=base_dir / 'nonbonded_distance_overlay.png',
            bins=bins
        )
        all_stats['nonbonded'] = nb_stats
    
    # ========================================================================
    # 6. Repulsion Energy Scatter
    # ========================================================================
    e_true = ensure_array(metrics.get('repulsion_energy_true', []))
    e_pred = ensure_array(metrics.get('repulsion_energy_pred', []))
    if len(e_true) > 0 and len(e_pred) > 0:
        e_stats = plot_scatter_identity(
            e_true, e_pred,
            xlabel='True Repulsion Energy',
            ylabel='Predicted Repulsion Energy',
            title=f'Repulsion Energy | {split.upper()} | Epoch {epoch}',
            output_path=base_dir / 'repulsion_energy_scatter.png',
            color='#8E44AD',
        )
        all_stats['repulsion'] = e_stats
    
    # ========================================================================
    # 7. Save Statistics Summary
    # ========================================================================
    summary = {
        'epoch': int(epoch),
        'split': str(split),
        'losses': {k: float(v) for k, v in losses.items()},
        'statistics': all_stats,
    }
    
    with open(base_dir / 'statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save as text for quick viewing
    with open(base_dir / 'statistics.txt', 'w') as f:
        f.write(f"Epoch {epoch} | Split: {split}\n")
        f.write("=" * 60 + "\n\n")
        f.write("LOSSES:\n")
        for k, v in losses.items():
            f.write(f"  {k}: {v:.6e}\n")
        f.write("\n")
        
        for metric_name, metric_stats in all_stats.items():
            f.write(f"{metric_name.upper()}:\n")
            for stat_key, stat_val in metric_stats.items():
                f.write(f"  {stat_key}: {stat_val:.6f}\n")
            f.write("\n")
    
    return all_stats


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage showing how to integrate with training loop.
    """
    
    # Simulated data
    np.random.seed(42)
    bond_true = np.random.normal(1.5, 0.1, 1000)
    bond_pred = bond_true + np.random.normal(0, 0.05, 1000)
    
    # Create plots
    stats = plot_scatter_identity(
        bond_true, bond_pred,
        xlabel='True Bond Length (Å)',
        ylabel='Predicted Bond Length (Å)',
        title='Bond Length Accuracy (Example)',
        output_path=Path('example_scatter.png')
    )
    
    print("Statistics computed:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")
    
    print("\nPublication-quality plot saved to: example_scatter.png")

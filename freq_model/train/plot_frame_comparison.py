"""
Create detailed frame-by-frame spectrum comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path


def plot_individual_frames_detailed(
    sample_results: list,
    omega_grid: np.ndarray,
    save_dir: Path,
    max_frames: int = 20
):
    """
    Create individual plots for each frame showing:
    1. Spectrum comparison
    2. Site energy comparison
    3. Metrics

    Args:
        sample_results: List of sample dictionaries
        omega_grid: Frequency grid
        save_dir: Directory to save plots
        max_frames: Maximum number of frames to plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_frames = min(len(sample_results), max_frames)

    for idx in range(n_frames):
        sample = sample_results[idx]
        spectrum_pred = sample['spectrum_pred']
        spectrum_true = sample['spectrum_true']
        H_pred = sample['H_diag_pred']
        H_true = sample['H_diag_true']
        frame_idx = sample['frame_idx']

        # Remove padding
        mask = H_true > 0
        H_pred_clean = H_pred[mask]
        H_true_clean = H_true[mask]

        if len(H_pred_clean) == 0:
            continue

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ===== Left: Spectrum Comparison =====
        ax = axes[0]

        # Plot spectra
        ax.plot(omega_grid, spectrum_true, label='Ground Truth', color='blue', linewidth=2.5, alpha=0.8)
        ax.plot(omega_grid, spectrum_pred, label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.8)

        # Mark peaks
        peak_true_idx = np.argmax(spectrum_true)
        peak_pred_idx = np.argmax(spectrum_pred)
        peak_true_freq = omega_grid[peak_true_idx]
        peak_pred_freq = omega_grid[peak_pred_idx]

        ax.axvline(peak_true_freq, color='blue', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(peak_pred_freq, color='red', linestyle=':', alpha=0.5, linewidth=1)

        # Compute metrics
        corr = np.corrcoef(spectrum_pred, spectrum_true)[0, 1]
        mse = np.mean((spectrum_pred - spectrum_true)**2)
        peak_error = abs(peak_pred_freq - peak_true_freq)

        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectrum Comparison (Frame {frame_idx})', fontsize=13, fontweight='bold')
        ax.set_xlim(omega_grid[0], omega_grid[-1])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

        # Add metrics text box
        metrics_text = f'Correlation: {corr:.4f}\nMSE: {mse:.6f}\nPeak Error: {peak_error:.2f} cm⁻¹'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== Right: Site Energy Comparison =====
        ax = axes[1]

        # Scatter plot
        ax.scatter(H_true_clean, H_pred_clean, alpha=0.6, s=40, c='purple', edgecolors='black', linewidth=0.5)

        # Identity line
        min_val = min(H_true_clean.min(), H_pred_clean.min())
        max_val = max(H_true_clean.max(), H_pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y=x', alpha=0.7)

        # MAE and RMSE
        mae = np.mean(np.abs(H_pred_clean - H_true_clean))
        rmse = np.sqrt(np.mean((H_pred_clean - H_true_clean)**2))
        r2_corr = np.corrcoef(H_pred_clean, H_true_clean)[0, 1]

        ax.set_xlabel('True Site Energy (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Site Energy (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_title(f'Site Energy Correlation (N={len(H_pred_clean)})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        ax.axis('equal')

        # Add metrics text box
        metrics_text = f'MAE: {mae:.2f} cm⁻¹\nRMSE: {rmse:.2f} cm⁻¹\nCorrelation: {r2_corr:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        save_path = save_dir / f'frame_{frame_idx:04d}_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {n_frames} individual frame comparison plots to {save_dir}")


def plot_spectra_grid_comparison(
    sample_results: list,
    omega_grid: np.ndarray,
    save_path: str,
    max_frames: int = 16
):
    """
    Plot grid of spectrum comparisons (more frames visible at once).

    Args:
        sample_results: List of sample dictionaries
        omega_grid: Frequency grid
        save_path: Output path
        max_frames: Number of frames to show
    """
    n_frames = min(len(sample_results), max_frames)
    n_cols = 4
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_frames):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        sample = sample_results[idx]
        spectrum_pred = sample['spectrum_pred']
        spectrum_true = sample['spectrum_true']
        frame_idx = sample['frame_idx']

        # Plot
        ax.plot(omega_grid, spectrum_true, label='True', color='blue', linewidth=1.5, alpha=0.7)
        ax.plot(omega_grid, spectrum_pred, label='Pred', color='red', linewidth=1.5, linestyle='--', alpha=0.7)

        # Correlation
        corr = np.corrcoef(spectrum_pred, spectrum_true)[0, 1]
        mse = np.mean((spectrum_pred - spectrum_true)**2)

        ax.set_title(f'Frame {frame_idx} (r={corr:.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.set_xlim(omega_grid[0], omega_grid[-1])
        ax.set_ylim(0, 1.1)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    # Remove empty subplots
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid comparison to {save_path}")


if __name__ == '__main__':
    # Can be called from main.py
    pass

"""
Extended plotting utilities for comprehensive analysis.

Includes:
- Loss curve decomposition
- Epoch-by-epoch correlation evolution
- Peak position tracking
- Spectral width analysis
- Residual analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from scipy.signal import find_peaks


def plot_detailed_loss_curves(history: Dict, save_path: Path):
    """
    Plot detailed loss curves with multiple subplots showing different aspects.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss curves (log scale)
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, color='C0')
    ax.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2, color='C1')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss Evolution (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 2. Correlation curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_spectrum_corr'], label='Train Spectrum Corr', linewidth=2, color='C2')
    ax.plot(epochs, history['test_spectrum_corr'], label='Test Spectrum Corr', linewidth=2, color='C3')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Spectrum Correlation Evolution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='red', linestyle='--', label='Target: 0.95', linewidth=1.5)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 3. Peak error evolution
    ax = axes[1, 0]
    ax.plot(epochs, history['train_peak_error_cm'], label='Train Peak Error', linewidth=2, color='C4')
    ax.plot(epochs, history['test_peak_error_cm'], label='Test Peak Error', linewidth=2, color='C5')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Peak Error (cm⁻¹)', fontsize=12)
    ax.set_title('Peak Position Error Evolution', fontsize=14, fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', label='Target: 5 cm⁻¹', linewidth=1.5)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 4. Site energy MAE
    ax = axes[1, 1]
    ax.plot(epochs, history['train_site_energy_mae'], label='Train Site Energy MAE', linewidth=2, color='C6')
    ax.plot(epochs, history['test_site_energy_mae'], label='Test Site Energy MAE', linewidth=2, color='C7')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Site Energy MAE (cm⁻¹)', fontsize=12)
    ax.set_title('Site Energy MAE Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved detailed loss curves: {save_path}")


def plot_convergence_analysis(history: Dict, save_path: Path):
    """
    Analyze convergence behavior: learning rate, gradient norms, etc.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss improvement per epoch
    ax = axes[0, 0]
    train_loss_diff = np.diff(history['train_loss'])
    test_loss_diff = np.diff(history['test_loss'])
    ax.plot(epochs[1:], train_loss_diff, label='Train Loss Change', linewidth=2, alpha=0.7, color='C0')
    ax.plot(epochs[1:], test_loss_diff, label='Test Loss Change', linewidth=2, alpha=0.7, color='C1')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Δ Loss', fontsize=12)
    ax.set_title('Loss Change Per Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 2. Correlation improvement per epoch
    ax = axes[0, 1]
    train_corr_diff = np.diff(history['train_spectrum_corr'])
    test_corr_diff = np.diff(history['test_spectrum_corr'])
    ax.plot(epochs[1:], train_corr_diff, label='Train Corr Change', linewidth=2, alpha=0.7, color='C2')
    ax.plot(epochs[1:], test_corr_diff, label='Test Corr Change', linewidth=2, alpha=0.7, color='C3')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Δ Correlation', fontsize=12)
    ax.set_title('Correlation Change Per Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 3. Convergence rate (moving average of improvements)
    ax = axes[1, 0]
    window = 10
    if len(test_corr_diff) > window:
        moving_avg = np.convolve(test_corr_diff, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(test_corr_diff)+1), moving_avg, linewidth=2, color='C3')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(f'Moving Avg Δ Corr (window={window})', fontsize=12)
        ax.set_title('Convergence Rate (Smoothed)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

    # 4. Overfitting metric (test - train)
    ax = axes[1, 1]
    train_corr = np.array(history['train_spectrum_corr'])
    test_corr = np.array(history['test_spectrum_corr'])
    overfit_metric = train_corr - test_corr
    ax.plot(epochs, overfit_metric, linewidth=2, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='No overfitting')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Warning threshold', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Corr - Test Corr', fontsize=12)
    ax.set_title('Overfitting Metric', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved convergence analysis: {save_path}")


def plot_spectrum_residuals(sample_results: List[Dict], omega_grid: np.ndarray, save_path: Path):
    """
    Plot residuals (predicted - true) to identify systematic errors.
    """
    # Collect all spectra
    spectra_pred = np.array([s['spectrum_pred'] for s in sample_results])
    spectra_true = np.array([s['spectrum_true'] for s in sample_results])

    # Compute residuals
    residuals = spectra_pred - spectra_true
    avg_residual = np.mean(residuals, axis=0)
    std_residual = np.std(residuals, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Residual plot
    ax = axes[0]
    ax.plot(omega_grid, avg_residual, linewidth=2, color='red', label='Mean Residual')
    ax.fill_between(omega_grid, avg_residual - std_residual, avg_residual + std_residual,
                     alpha=0.3, color='red', label='±1 Std')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - True)', fontsize=12)
    ax.set_title('Spectrum Residuals (Averaged Over Frames)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 2. Absolute residual
    ax = axes[1]
    abs_residual = np.abs(avg_residual)
    ax.plot(omega_grid, abs_residual, linewidth=2, color='purple')
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('|Residual|', fontsize=12)
    ax.set_title('Absolute Residual', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Find regions with large errors
    threshold = 0.05
    problem_regions = omega_grid[abs_residual > threshold]
    if len(problem_regions) > 0:
        ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        ax.legend(fontsize=11)
        print(f"\nWarning: Large residuals (>{threshold}) at frequencies: {problem_regions[:5]}... cm⁻¹")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved residual analysis: {save_path}")


def plot_peak_position_analysis(sample_results: List[Dict], omega_grid: np.ndarray, save_path: Path):
    """
    Analyze peak positions: distribution, errors, correlations.
    """
    # Extract peak positions
    peak_pos_true = []
    peak_pos_pred = []

    for sample in sample_results:
        spec_true = sample['spectrum_true']
        spec_pred = sample['spectrum_pred']

        # Find peak (max intensity)
        peak_true = omega_grid[np.argmax(spec_true)]
        peak_pred = omega_grid[np.argmax(spec_pred)]

        peak_pos_true.append(peak_true)
        peak_pos_pred.append(peak_pred)

    peak_pos_true = np.array(peak_pos_true)
    peak_pos_pred = np.array(peak_pos_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(peak_pos_true, peak_pos_pred, alpha=0.6, s=50, color='C0')
    min_val = min(peak_pos_true.min(), peak_pos_pred.min())
    max_val = max(peak_pos_true.max(), peak_pos_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('True Peak Position (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Predicted Peak Position (cm⁻¹)', fontsize=12)
    ax.set_title('Peak Position Correlation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Compute metrics
    mae = np.mean(np.abs(peak_pos_pred - peak_pos_true))
    corr = np.corrcoef(peak_pos_pred, peak_pos_true)[0, 1]
    ax.text(0.05, 0.95, f'MAE: {mae:.2f} cm⁻¹\nCorr: {corr:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Error distribution
    ax = axes[0, 1]
    errors = peak_pos_pred - peak_pos_true
    ax.hist(errors, bins=30, color='C1', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(x=np.mean(errors), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.set_xlabel('Peak Position Error (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Peak Position Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 3. True peak distribution
    ax = axes[1, 0]
    ax.hist(peak_pos_true, bins=30, color='C2', alpha=0.7, edgecolor='black', label='True')
    ax.axvline(x=np.mean(peak_pos_true), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(peak_pos_true):.1f}')
    ax.set_xlabel('Peak Position (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('True Peak Position Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 4. Predicted peak distribution
    ax = axes[1, 1]
    ax.hist(peak_pos_pred, bins=30, color='C3', alpha=0.7, edgecolor='black', label='Predicted')
    ax.axvline(x=np.mean(peak_pos_pred), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(peak_pos_pred):.1f}')
    ax.set_xlabel('Peak Position (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Predicted Peak Position Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved peak position analysis: {save_path}")
    print(f"  Peak position MAE: {mae:.2f} cm⁻¹")
    print(f"  Peak position correlation: {corr:.4f}")


def plot_site_energy_residuals(sample_results: List[Dict], save_path: Path):
    """
    Analyze site energy prediction residuals.
    """
    # Collect all site energies
    H_pred_all = []
    H_true_all = []

    for sample in sample_results:
        H_pred = sample['H_diag_pred']
        H_true = sample['H_diag_true']

        # Remove padding
        mask = H_true > 0
        H_pred_all.extend(H_pred[mask])
        H_true_all.extend(H_true[mask])

    H_pred_all = np.array(H_pred_all)
    H_true_all = np.array(H_true_all)

    residuals = H_pred_all - H_true_all

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residual vs true value
    ax = axes[0, 0]
    ax.scatter(H_true_all, residuals, alpha=0.3, s=10, color='C0')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('True Site Energy (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - True)', fontsize=12)
    ax.set_title('Site Energy Residuals', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Residual distribution
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, color='C1', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(x=np.mean(residuals), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(residuals):.2f}')
    ax.set_xlabel('Residual (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 3. Q-Q plot (check if residuals are normally distributed)
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Absolute error vs true value
    ax = axes[1, 1]
    abs_errors = np.abs(residuals)
    ax.scatter(H_true_all, abs_errors, alpha=0.3, s=10, color='C2')
    ax.set_xlabel('True Site Energy (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Absolute Error (cm⁻¹)', fontsize=12)
    ax.set_title('Absolute Error vs True Value', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add horizontal line at mean absolute error
    mae = np.mean(abs_errors)
    ax.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved site energy residuals: {save_path}")
    print(f"  Site energy MAE: {mae:.2f} cm⁻¹")
    print(f"  Residual mean: {np.mean(residuals):.2f} cm⁻¹")
    print(f"  Residual std: {np.std(residuals):.2f} cm⁻¹")


def generate_all_extended_plots(
    history: Dict,
    sample_results: List[Dict],
    omega_grid: np.ndarray,
    output_dir: Path
):
    """
    Generate all extended analysis plots.
    """
    print("\n" + "="*80)
    print("EXTENDED ANALYSIS PLOTS")
    print("="*80)

    output_dir = Path(output_dir)
    extended_dir = output_dir / 'extended_analysis'
    extended_dir.mkdir(exist_ok=True, parents=True)

    print("\n1. Detailed loss curves...")
    plot_detailed_loss_curves(history, extended_dir / 'detailed_loss_curves.png')

    print("\n2. Convergence analysis...")
    plot_convergence_analysis(history, extended_dir / 'convergence_analysis.png')

    print("\n3. Spectrum residuals...")
    plot_spectrum_residuals(sample_results, omega_grid, extended_dir / 'spectrum_residuals.png')

    print("\n4. Peak position analysis...")
    plot_peak_position_analysis(sample_results, omega_grid, extended_dir / 'peak_position_analysis.png')

    print("\n5. Site energy residuals...")
    plot_site_energy_residuals(sample_results, extended_dir / 'site_energy_residuals.png')

    print("\n" + "="*80)
    print(f"Extended analysis plots saved to: {extended_dir}")
    print("="*80)

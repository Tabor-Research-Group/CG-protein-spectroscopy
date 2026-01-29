"""
Evaluation and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List
from pathlib import Path


def plot_training_curves(
    history: Dict[str, List],
    save_path: Path
):
    """
    Plot training curves.

    Args:
        history: Dictionary with lists of metrics over epochs
        save_path: Output path
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0].plot(epochs, history['test_loss'], label='Test', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Spectrum MSE
    axes[1].plot(epochs, history['train_spectrum_mse'], label='Train', marker='o')
    axes[1].plot(epochs, history['test_spectrum_mse'], label='Test', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Spectrum MSE')
    axes[1].set_title('Spectrum MSE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Spectrum Correlation
    axes[2].plot(epochs, history['train_spectrum_corr'], label='Train', marker='o')
    axes[2].plot(epochs, history['test_spectrum_corr'], label='Test', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Spectrum Correlation')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # Peak Error
    axes[3].plot(epochs, history['train_peak_error_cm'], label='Train', marker='o')
    axes[3].plot(epochs, history['test_peak_error_cm'], label='Test', marker='s')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Peak Error (cm⁻¹)')
    axes[3].set_title('Peak Position Error')
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    # Site Energy MAE
    axes[4].plot(epochs, history['train_site_energy_mae'], label='Train', marker='o')
    axes[4].plot(epochs, history['test_site_energy_mae'], label='Test', marker='s')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('MAE (cm⁻¹)')
    axes[4].set_title('Site Energy MAE')
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_spectra_comparison(
    sample_results: List[Dict],
    omega_grid: np.ndarray,
    save_path: Path,
    max_samples: int = 12
):
    """
    Plot predicted vs true spectra for multiple samples.

    Args:
        sample_results: List of dictionaries with spectrum_pred, spectrum_true
        omega_grid: Frequency grid
        save_path: Output path
        max_samples: Maximum number of samples to plot
    """
    n_samples = min(len(sample_results), max_samples)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        sample = sample_results[idx]
        spectrum_pred = sample['spectrum_pred']
        spectrum_true = sample['spectrum_true']
        frame_idx = sample['frame_idx']

        # Plot
        ax.plot(omega_grid, spectrum_true, label='Ground Truth', color='blue', linewidth=2)
        ax.plot(omega_grid, spectrum_pred, label='Predicted', color='red', linestyle='--', linewidth=2)

        # Compute correlation
        corr = np.corrcoef(spectrum_pred, spectrum_true)[0, 1]

        ax.set_title(f"Frame {frame_idx} (Corr: {corr:.3f})", fontsize=10)
        ax.set_xlabel("Frequency (cm⁻¹)", fontsize=9)
        ax.set_ylabel("Intensity", fontsize=9)
        ax.set_xlim(omega_grid[0], omega_grid[-1])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Remove empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectra comparison saved to {save_path}")


def plot_site_energy_comparison(
    sample_results: List[Dict],
    save_path: Path,
    max_samples: int = 6
):
    """
    Plot predicted vs true site energies.

    Args:
        sample_results: List of sample results
        save_path: Output path
        max_samples: Max samples to plot
    """
    n_samples = min(len(sample_results), max_samples)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        sample = sample_results[idx]
        H_pred = sample['H_diag_pred']
        H_true = sample['H_diag_true']
        frame_idx = sample['frame_idx']

        # Remove padding (zeros)
        mask = H_true > 0
        H_pred = H_pred[mask]
        H_true = H_true[mask]

        if len(H_pred) == 0:
            continue

        # Scatter plot
        ax.scatter(H_true, H_pred, alpha=0.6, s=20)

        # Identity line
        min_val = min(H_true.min(), H_pred.min())
        max_val = max(H_true.max(), H_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')

        # MAE
        mae = np.mean(np.abs(H_pred - H_true))

        ax.set_title(f"Frame {frame_idx} (MAE: {mae:.1f} cm⁻¹)", fontsize=10)
        ax.set_xlabel("True Site Energy (cm⁻¹)", fontsize=9)
        ax.set_ylabel("Predicted Site Energy (cm⁻¹)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Remove empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Site energy comparison saved to {save_path}")


def plot_site_energy_distribution(
    sample_results: List[Dict],
    save_path: Path
):
    """
    Plot distribution of site energies (predicted vs true).

    Args:
        sample_results: List of sample results
        save_path: Output path
    """
    # Collect all site energies
    H_pred_all = []
    H_true_all = []

    for sample in sample_results:
        H_pred = sample['H_diag_pred']
        H_true = sample['H_diag_true']

        # Remove padding
        mask = H_true > 0
        H_pred_all.extend(H_pred[mask].tolist())
        H_true_all.extend(H_true[mask].tolist())

    H_pred_all = np.array(H_pred_all)
    H_true_all = np.array(H_true_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histograms
    axes[0].hist(H_true_all, bins=50, alpha=0.6, label='Ground Truth', color='blue')
    axes[0].hist(H_pred_all, bins=50, alpha=0.6, label='Predicted', color='red')
    axes[0].set_xlabel('Site Energy (cm⁻¹)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Site Energy Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter plot
    axes[1].scatter(H_true_all, H_pred_all, alpha=0.3, s=5)
    min_val = min(H_true_all.min(), H_pred_all.min())
    max_val = max(H_true_all.max(), H_pred_all.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')

    mae = np.mean(np.abs(H_pred_all - H_true_all))
    rmse = np.sqrt(np.mean((H_pred_all - H_true_all)**2))
    corr = np.corrcoef(H_pred_all, H_true_all)[0, 1]

    axes[1].set_xlabel('True Site Energy (cm⁻¹)')
    axes[1].set_ylabel('Predicted Site Energy (cm⁻¹)')
    axes[1].set_title(f'Site Energy Correlation\n(MAE: {mae:.1f}, RMSE: {rmse:.1f}, Corr: {corr:.3f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Site energy distribution saved to {save_path}")


def plot_average_spectrum(
    sample_results: List[Dict],
    omega_grid: np.ndarray,
    save_path: Path
):
    """
    Plot average predicted vs true spectrum.

    Args:
        sample_results: List of sample results
        omega_grid: Frequency grid
        save_path: Output path
    """
    # Collect all spectra
    spectra_pred = []
    spectra_true = []

    for sample in sample_results:
        spectra_pred.append(sample['spectrum_pred'])
        spectra_true.append(sample['spectrum_true'])

    spectra_pred = np.array(spectra_pred)
    spectra_true = np.array(spectra_true)

    # Compute averages and stds
    avg_pred = np.mean(spectra_pred, axis=0)
    std_pred = np.std(spectra_pred, axis=0)
    avg_true = np.mean(spectra_true, axis=0)
    std_true = np.std(spectra_true, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot averages
    ax.plot(omega_grid, avg_true, label='Ground Truth (avg)', color='blue', linewidth=2)
    ax.fill_between(omega_grid, avg_true - std_true, avg_true + std_true, alpha=0.2, color='blue')

    ax.plot(omega_grid, avg_pred, label='Predicted (avg)', color='red', linewidth=2, linestyle='--')
    ax.fill_between(omega_grid, avg_pred - std_pred, avg_pred + std_pred, alpha=0.2, color='red')

    # Compute correlation
    corr = np.corrcoef(avg_pred, avg_true)[0, 1]
    mse = np.mean((avg_pred - avg_true)**2)

    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'Average Spectrum (n={len(sample_results)})\n(Corr: {corr:.4f}, MSE: {mse:.6f})', fontsize=14)
    ax.set_xlim(omega_grid[0], omega_grid[-1])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Average spectrum saved to {save_path}")


def save_metrics_summary(
    history: Dict[str, List],
    final_test_metrics: Dict,
    save_path: Path
):
    """
    Save metrics summary to text file.

    Args:
        history: Training history
        final_test_metrics: Final test metrics
        save_path: Output path
    """
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total epochs: {len(history['train_loss'])}\n\n")

        f.write("Final Training Metrics:\n")
        f.write(f"  Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Spectrum MSE: {history['train_spectrum_mse'][-1]:.6f}\n")
        f.write(f"  Spectrum Correlation: {history['train_spectrum_corr'][-1]:.4f}\n")
        f.write(f"  Peak Error: {history['train_peak_error_cm'][-1]:.2f} cm⁻¹\n")
        f.write(f"  Site Energy MAE: {history['train_site_energy_mae'][-1]:.2f} cm⁻¹\n\n")

        f.write("Final Test Metrics:\n")
        f.write(f"  Loss: {final_test_metrics['loss']:.6f}\n")
        f.write(f"  Spectrum MSE: {final_test_metrics['spectrum_mse']:.6f}\n")
        f.write(f"  Spectrum Correlation: {final_test_metrics['spectrum_corr']:.4f}\n")
        f.write(f"  Peak Error: {final_test_metrics['peak_error_cm']:.2f} cm⁻¹\n")
        f.write(f"  Site Energy MAE: {final_test_metrics['site_energy_mae']:.2f} cm⁻¹\n\n")

        f.write("Best Test Metrics (across epochs):\n")
        best_corr_epoch = np.argmax(history['test_spectrum_corr']) + 1
        f.write(f"  Best Correlation: {max(history['test_spectrum_corr']):.4f} (epoch {best_corr_epoch})\n")

        best_mse_epoch = np.argmin(history['test_spectrum_mse']) + 1
        f.write(f"  Best MSE: {min(history['test_spectrum_mse']):.6f} (epoch {best_mse_epoch})\n")

        best_peak_epoch = np.argmin(history['test_peak_error_cm']) + 1
        f.write(f"  Best Peak Error: {min(history['test_peak_error_cm']):.2f} cm⁻¹ (epoch {best_peak_epoch})\n")

    print(f"Metrics summary saved to {save_path}")


def evaluate_protein_file(
    model,
    frames_dict: Dict[int, List],
    omega_grid: np.ndarray,
    device: str,
    output_dir: Path,
    protein_id: str,
    config: Dict,
    n_sample_frames: int = 5,
    protein_data: Dict = None
) -> Dict:
    """
    Evaluate model on a single protein and generate plots.

    Args:
        model: Trained model
        frames_dict: Frames dictionary for this protein
        omega_grid: Frequency grid
        device: Device to run on
        output_dir: Output directory for this protein
        protein_id: Protein identifier
        config: Model configuration
        n_sample_frames: Number of sample frames to plot

    Returns:
        results: Dictionary with metrics
    """
    from dataset import create_dataloaders
    from train_optimized import evaluate
    import torch

    model.eval()

    # Get frame indices
    frame_indices = sorted(frames_dict.keys())

    # Create dataloader for all frames
    _, protein_loader = create_dataloaders(
        train_data=protein_data,
        test_data=protein_data,
        train_frame_indices=frame_indices,
        test_frame_indices=frame_indices,
        batch_size=8,
        num_workers=0,
        cutoff=config['cutoff'],
        max_neighbors=config['max_neighbors']
    )

    # Evaluate
    omega_grid_tensor = torch.from_numpy(omega_grid).float().to(device)

    # Create a simple criterion for evaluation using config values
    from train_optimized import SpectrumLoss
    criterion = SpectrumLoss(
        lambda_peak=config.get('lambda_peak', 0.5),
        lambda_correlation=config.get('lambda_correlation', 0.3),
        omega_grid=omega_grid_tensor,
        peak_scale=100.0
    )

    with torch.no_grad():
        avg_metrics, sample_results = evaluate(
            model=model,
            test_loader=protein_loader,
            criterion=criterion,
            device=device,
            omega_grid=omega_grid_tensor
        )

    # Clear GPU cache after evaluation to free memory
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate average spectrum plot
    if sample_results:
        plot_average_spectrum(
            sample_results=sample_results,
            omega_grid=omega_grid,
            save_path=output_dir / f'{protein_id}_average_spectrum.png'
        )

    # Generate sample frame plots
    from plot_frame_comparison import plot_individual_frames_detailed

    if sample_results and len(sample_results) > 0:
        frame_details_dir = output_dir / 'frame_details'
        frame_details_dir.mkdir(exist_ok=True)

        # Use the sample_results from evaluation (already has all the data)
        plot_individual_frames_detailed(
            sample_results=sample_results[:n_sample_frames],  # Limit to n_sample_frames
            omega_grid=omega_grid,
            save_dir=frame_details_dir,
            max_frames=n_sample_frames
        )

    return avg_metrics

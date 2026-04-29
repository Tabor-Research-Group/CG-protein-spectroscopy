"""
Per-protein tracking during training.

Tracks metrics for each validation protein across epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import torch
from collections import defaultdict


def evaluate_per_protein(
    model,
    val_files_data: Dict[str, Dict],
    omega_grid: np.ndarray,
    device: str,
    config: Dict,
    criterion
) -> Dict[str, Dict]:
    """
    Evaluate model on each validation protein separately.

    Args:
        model: Trained model
        val_files_data: Dict of {protein_id: data_dict}
        omega_grid: Frequency grid
        device: Device
        config: Model config
        criterion: Loss function

    Returns:
        per_protein_metrics: {protein_id: {metric: value}}
    """
    from data_utils import organize_by_frames, filter_frames_by_quality
    from dataset import create_dataloaders
    from train_optimized import evaluate

    model.eval()
    per_protein_metrics = {}

    omega_grid_tensor = torch.from_numpy(omega_grid).float().to(device)

    for protein_id, protein_data in val_files_data.items():
        # Organize and filter frames
        frames = organize_by_frames(protein_data)
        frames, _, _ = filter_frames_by_quality(frames, verbose=False)

        if len(frames) == 0:
            continue

        # Get frame indices
        frame_indices = sorted(frames.keys())

        # Create dataloader - use same data and indices for both train and test
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
        with torch.no_grad():
            avg_metrics, _ = evaluate(
                model=model,
                test_loader=protein_loader,
                criterion=criterion,
                device=device,
                omega_grid=omega_grid_tensor
            )
            results = avg_metrics  # Use avg_metrics from the tuple return

        per_protein_metrics[protein_id] = {
            'spectrum_corr': results['spectrum_corr'],
            'peak_error_cm': results['peak_error_cm'],
            'spectrum_mse': results['spectrum_mse'],
            'site_energy_mae': results['site_energy_mae'],
        }

    return per_protein_metrics


def plot_per_protein_evolution(
    per_protein_history: Dict[str, Dict[str, List]],
    output_path: Path
):
    """
    Plot how each protein's metrics evolved during training.

    Args:
        per_protein_history: {protein_id: {metric: [values_per_epoch]}}
        output_path: Save path
    """
    proteins = sorted(per_protein_history.keys())
    n_proteins = len(proteins)

    if n_proteins == 0:
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Protein Evolution During Training', fontsize=16, fontweight='bold')

    metrics = [
        ('spectrum_corr', 'Spectrum Correlation', axes[0, 0]),
        ('peak_error_cm', 'Peak Error (cm⁻¹)', axes[0, 1]),
        ('spectrum_mse', 'Spectrum MSE', axes[1, 0]),
        ('site_energy_mae', 'Site Energy MAE (cm⁻¹)', axes[1, 1])
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n_proteins))

    for metric_key, metric_name, ax in metrics:
        for i, protein_id in enumerate(proteins):
            if metric_key in per_protein_history[protein_id]:
                epochs = range(1, len(per_protein_history[protein_id][metric_key]) + 1)
                values = per_protein_history[protein_id][metric_key]
                ax.plot(epochs, values, label=protein_id, marker='o',
                       color=colors[i], linewidth=2, markersize=4)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

        # Special handling for correlation (higher is better)
        if metric_key == 'spectrum_corr':
            ax.axhline(y=0.9, color='red', linestyle='--',
                      label='Target: 0.9', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-protein evolution plot saved: {output_path}")


def plot_final_per_protein_comparison(
    per_protein_history: Dict[str, Dict[str, List]],
    output_path: Path
):
    """
    Plot final metrics comparison across proteins.

    Args:
        per_protein_history: {protein_id: {metric: [values_per_epoch]}}
        output_path: Save path
    """
    proteins = sorted(per_protein_history.keys())

    if len(proteins) == 0:
        return

    # Get final values
    final_corr = [per_protein_history[p]['spectrum_corr'][-1] for p in proteins]
    final_peak = [per_protein_history[p]['peak_error_cm'][-1] for p in proteins]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Final Per-Protein Performance', fontsize=14, fontweight='bold')

    # Correlation
    axes[0].barh(proteins, final_corr, color='steelblue')
    axes[0].axvline(x=0.9, color='red', linestyle='--', label='Target: 0.9', linewidth=2)
    axes[0].set_xlabel('Spectrum Correlation', fontsize=12)
    axes[0].set_title('Final Correlation per Protein', fontsize=13)
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)

    # Peak error
    axes[1].barh(proteins, final_peak, color='coral')
    axes[1].axvline(x=5.0, color='red', linestyle='--', label='Target: 5 cm⁻¹', linewidth=2)
    axes[1].set_xlabel('Peak Error (cm⁻¹)', fontsize=12)
    axes[1].set_title('Final Peak Error per Protein', fontsize=13)
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Final per-protein comparison saved: {output_path}")

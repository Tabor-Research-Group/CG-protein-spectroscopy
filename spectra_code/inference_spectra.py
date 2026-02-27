#!/usr/bin/env python3
"""
Inference Script with Hamiltonian and Dipole Export

This script extends the corrected inference pipeline to export:
1. Predicted Hamiltonians (H_diag + J_ij) - flattened matrices
2. Ground Truth Hamiltonians (H_diag + J_ij) - flattened matrices
3. Predicted Dipoles (x, y, z components separately)
4. Ground Truth Dipoles (x, y, z components separately)

Format for dipoles: frame_number [all x-components] [all y-components] [all z-components]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Import from main codebase - SAME as training
from train.model import create_model
from train.dataset import SpectrumDataset, collate_fn_pad
from train.data_utils import load_pkl_data, organize_by_frames, filter_frames_by_quality
from torch.utils.data import DataLoader

# Import physics functions - SAME as training
from train.physics import (
    calculate_torii_dipole_batch_torch,
    batch_generate_spectra_torch
)

# Import the SAME coupling function as training
from train.train_optimized import calculate_tasumi_coupling_batch_torch

# Publication-quality plotting
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18


def inference_on_dataloader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0,
) -> List[Dict]:
    """
    Run inference on a dataloader - SAME AS EVALUATION IN TRAINING.

    Uses EXACT same pipeline as training:
    1. Batched processing with padding
    2. Torch-based spectrum generation
    3. Masking for padded oscillators

    Returns:
        List of result dictionaries, one per frame
    """
    model.eval()
    results = []

    # Frequency grid for later use
    omega_grid = np.arange(omega_min, omega_max + omega_step, omega_step)

    print(f"\nRunning inference on {len(data_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Inference')):
            # Move to device - SAME as training
            own_features = batch['own_features'].to(device)
            neighbor_features = batch['neighbor_features'].to(device)
            neighbor_mask = batch['neighbor_mask'].to(device)
            H_diag_true = batch['H_diag_true'].to(device)
            C_positions_pred = batch['C_positions_pred'].to(device)
            O_positions_pred = batch['O_positions_pred'].to(device)
            N_positions_pred = batch['N_positions_pred'].to(device)
            spectrum_true = batch['spectrum_true'].to(device)
            oscillator_mask = batch['oscillator_mask'].to(device)
            frame_indices = batch['frame_indices']

            # Forward pass: predict H_diag - SAME as training
            H_diag_pred = model(own_features, neighbor_features, neighbor_mask)

            # Calculate dipoles - SAME as training
            dipoles_pred = calculate_torii_dipole_batch_torch(
                C_positions_pred, O_positions_pred, N_positions_pred
            )

            # Calculate dipoles for ground truth - SAME as training
            dipoles_true = batch['dipoles_true'].to(device)

            # Calculate couplings for predicted - SAME as training (WITH MASK!)
            J_matrix_pred = calculate_tasumi_coupling_batch_torch(
                dipoles_pred, C_positions_pred, oscillator_mask
            )

            # Get ground truth coupling matrix (already calculated in dataset)
            J_matrix_true = batch['J_matrix_true'].to(device)

            # Generate IR spectrum - SAME as training (WITH MASK!)
            spectrum_pred = batch_generate_spectra_torch(
                H_diag_pred, J_matrix_pred, dipoles_pred,
                mask_batch=oscillator_mask,  # CRITICAL: masking for padded oscillators
                omega_min=omega_min,
                omega_max=omega_max,
                omega_step=omega_step,
                gamma=gamma
            )

            # Store results for each frame in the batch
            B = H_diag_pred.shape[0]
            for i in range(B):
                # Extract valid oscillators (remove padding)
                mask_i = oscillator_mask[i].cpu().numpy().astype(bool)
                n_valid = mask_i.sum()

                result = {
                    'frame_idx': frame_indices[i],
                    'H_diag_pred': H_diag_pred[i, :n_valid].cpu().numpy(),
                    'H_diag_true': H_diag_true[i, :n_valid].cpu().numpy(),
                    'J_matrix_pred': J_matrix_pred[i, :n_valid, :n_valid].cpu().numpy(),
                    'J_matrix_true': J_matrix_true[i, :n_valid, :n_valid].cpu().numpy(),
                    'spectrum_pred': spectrum_pred[i].cpu().numpy(),
                    'spectrum_true': spectrum_true[i].cpu().numpy(),
                    'dipoles_pred': dipoles_pred[i, :n_valid].cpu().numpy(),
                    'dipoles_true': dipoles_true[i, :n_valid].cpu().numpy(),
                    'omega_grid': omega_grid,
                }
                results.append(result)

    print(f"✓ Inference complete: {len(results)} frames processed")
    return results


def save_results(results: List[Dict], output_dir: Path):
    """Save numerical results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Determine number of oscillators from first frame
    if len(results) > 0:
        N_oscillators = len(results[0]['H_diag_pred'])

        # Check if constant across all frames
        all_N = [len(r['H_diag_pred']) for r in results]
        is_constant = all(n == N_oscillators for n in all_N)

        # Save oscillator information
        with open(output_dir / 'oscillator_info.txt', 'w') as f:
            f.write(f"# Oscillator Information\n")
            f.write(f"# Generated from {len(results)} frames\n")
            f.write(f"#\n")
            if is_constant:
                f.write(f"# Number of oscillators: CONSTANT across all frames\n")
                f.write(f"N_oscillators: {N_oscillators}\n")
            else:
                f.write(f"# Number of oscillators: VARIABLE across frames\n")
                f.write(f"# Min: {min(all_N)}, Max: {max(all_N)}, Mean: {sum(all_N)/len(all_N):.1f}\n")
                f.write(f"N_oscillators_min: {min(all_N)}\n")
                f.write(f"N_oscillators_max: {max(all_N)}\n")
                f.write(f"N_oscillators_mean: {sum(all_N)/len(all_N):.1f}\n")
                f.write(f"\n# Per-frame oscillator counts:\n")
                for r in results:
                    f.write(f"{r['frame_idx']}: {len(r['H_diag_pred'])}\n")

        print(f"  ✓ Saved oscillator_info.txt (N={N_oscillators}{'*' if not is_constant else ''})")

    # Save Predicted Full Hamiltonian matrices (H_diag + J_matrix)
    # Format: NISE-compatible upper triangular (row by row: diagonal + upper triangle)
    with open(output_dir / 'hamiltonians_predicted.dat', 'w') as f:
        for result in results:
            frame_idx = result['frame_idx']
            H_pred = result['H_diag_pred']

            # Build full Hamiltonian matrix: H = diag(H_diag) + J_matrix
            N = len(H_pred)
            H_full = np.diag(H_pred)

            # Add coupling matrix if available
            if 'J_matrix_pred' in result:
                H_full += result['J_matrix_pred']

            # Flatten in NISE format: row by row, each row contains diagonal + upper triangle
            # Row 0: H[0,0], H[0,1], H[0,2], ..., H[0,N-1]
            # Row 1: H[1,1], H[1,2], ..., H[1,N-1]
            # Row 2: H[2,2], H[2,3], ..., H[2,N-1]
            # etc.
            hamiltonian_values = []
            for i in range(N):
                hamiltonian_values.append(H_full[i, i])  # Diagonal element
                hamiltonian_values.extend(H_full[i, i+1:])  # Upper triangle elements

            line = f"{frame_idx}"
            for val in hamiltonian_values:
                line += f" {val:.6f}"
            f.write(line + "\n")
    print(f"  ✓ Saved hamiltonians_predicted.dat (NISE-compatible upper triangular format)")

    # Save Ground Truth Full Hamiltonian matrices (H_diag + J_matrix)
    # Format: NISE-compatible upper triangular (row by row: diagonal + upper triangle)
    with open(output_dir / 'hamiltonians_groundtruth.dat', 'w') as f:
        for result in results:
            frame_idx = result['frame_idx']
            H_true = result['H_diag_true']

            # Build full Hamiltonian matrix: H = diag(H_diag) + J_matrix
            N = len(H_true)
            H_full = np.diag(H_true)

            # Add coupling matrix for ground truth
            if 'J_matrix_true' in result:
                H_full += result['J_matrix_true']

            # Flatten in NISE format: row by row, each row contains diagonal + upper triangle
            # Row 0: H[0,0], H[0,1], H[0,2], ..., H[0,N-1]
            # Row 1: H[1,1], H[1,2], ..., H[1,N-1]
            # Row 2: H[2,2], H[2,3], ..., H[2,N-1]
            # etc.
            hamiltonian_values = []
            for i in range(N):
                hamiltonian_values.append(H_full[i, i])  # Diagonal element
                hamiltonian_values.extend(H_full[i, i+1:])  # Upper triangle elements

            line = f"{frame_idx}"
            for val in hamiltonian_values:
                line += f" {val:.6f}"
            f.write(line + "\n")
    print(f"  ✓ Saved hamiltonians_groundtruth.dat (NISE-compatible upper triangular format)")

    # Save Predicted Dipoles (x, y, z components)
    with open(output_dir / 'dipoles_predicted.dat', 'w') as f:
        for result in results:
            frame_idx = result['frame_idx']
            dipoles = result['dipoles_pred']  # Shape: (N_oscillators, 3)
            N = len(dipoles)

            # Write frame number, then all x-components, then all y-components, then all z-components
            line = f"{frame_idx}"

            # All x-components
            for i in range(N):
                line += f" {dipoles[i, 0]:.6f}"

            # All y-components
            for i in range(N):
                line += f" {dipoles[i, 1]:.6f}"

            # All z-components
            for i in range(N):
                line += f" {dipoles[i, 2]:.6f}"

            f.write(line + "\n")
    print(f"  ✓ Saved dipoles_predicted.dat (frame_number x-components y-components z-components)")

    # Save Ground Truth Dipoles (x, y, z components)
    with open(output_dir / 'dipoles_groundtruth.dat', 'w') as f:
        for result in results:
            frame_idx = result['frame_idx']
            dipoles = result['dipoles_true']  # Shape: (N_oscillators, 3)
            N = len(dipoles)

            # Write frame number, then all x-components, then all y-components, then all z-components
            line = f"{frame_idx}"

            # All x-components
            for i in range(N):
                line += f" {dipoles[i, 0]:.6f}"

            # All y-components
            for i in range(N):
                line += f" {dipoles[i, 1]:.6f}"

            # All z-components
            for i in range(N):
                line += f" {dipoles[i, 2]:.6f}"

            f.write(line + "\n")
    print(f"  ✓ Saved dipoles_groundtruth.dat (frame_number x-components y-components z-components)")

    # Save spectra
    with open(output_dir / 'spectra.dat', 'w') as f:
        f.write("# Frame_Number Frequency_(cm-1) Intensity_Predicted Intensity_True\n")
        for result in results:
            frame_idx = result['frame_idx']
            omega = result['omega_grid']
            spec_pred = result['spectrum_pred']
            spec_true = result['spectrum_true']

            for freq, int_pred, int_true in zip(omega, spec_pred, spec_true):
                f.write(f"{frame_idx} {freq:.2f} {int_pred:.6f} {int_true:.6f}\n")
    print(f"  ✓ Saved spectra.dat")


def plot_spectra_comparison(results: List[Dict], output_dir: Path, max_plots: int = 9):
    """Plot individual frame spectra comparisons."""
    n_plots = min(len(results), max_plots)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx in range(n_plots):
        ax = axes[idx]
        result = results[idx]

        omega = result['omega_grid']
        spec_pred = result['spectrum_pred']
        spec_true = result['spectrum_true']
        frame_idx = result['frame_idx']

        # Plot - updated labels
        ax.plot(omega, spec_true, 'k-', linewidth=2, label='Atomistic', alpha=0.7)
        ax.plot(omega, spec_pred, 'r--', linewidth=2, label='CG')

        # Calculate correlation
        corr = np.corrcoef(spec_pred, spec_true)[0, 1]

        ax.set_xlabel('Frequency (cm⁻¹)', fontweight='bold')
        ax.set_ylabel('Normalized Intensity', fontweight='bold')
        ax.set_title(f'Frame {frame_idx} (r = {corr:.3f})', fontweight='bold')
        ax.set_xlim(1500, 1750)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper left', frameon=False)  # Changed to upper left

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'spectra_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved spectra_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Corrected Inference - Matches Training Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to inference config JSON')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    print("\n" + "="*80)
    print("CORRECTED INFERENCE - MATCHES TRAINING PIPELINE")
    print("="*80)
    print(f"\nConfiguration: {args.config}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print("\nLoading trained model...")
    model_config = config['model_config']

    # Ensure min/max energy are set
    if 'min_energy' not in model_config:
        model_config['min_energy'] = 1450.0
        print("  WARNING: min_energy not in config, using default 1450.0")
    if 'max_energy' not in model_config:
        model_config['max_energy'] = 1850.0
        print("  WARNING: max_energy not in config, using default 1850.0")

    model = create_model(model_config)
    checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from: {config['model_path']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Energy scaling: [{model.output_head.min_energy:.1f}, {model.output_head.max_energy:.1f}] cm⁻¹")

    # Load data and create dataset - SAME AS TRAINING
    print("\nLoading data...")
    pkl_data = load_pkl_data(config['data_path'])
    frames_dict = organize_by_frames(pkl_data)
    frames_dict, _, _ = filter_frames_by_quality(frames_dict, verbose=True)

    all_frame_indices = sorted(frames_dict.keys())
    max_frames = config.get('max_frames', len(all_frame_indices))
    frame_indices = all_frame_indices[:max_frames]

    print(f"✓ Selected {len(frame_indices)} frames for inference")

    # Create dataset - SAME AS TRAINING
    dataset = SpectrumDataset(
        pkl_data=pkl_data,
        frame_indices=frame_indices,
        cutoff=config['cutoff'],
        max_neighbors=config['max_neighbors'],
        omega_min=config['omega_min'],
        omega_max=config['omega_max'],
        omega_step=config['omega_step'],
        gamma=config['gamma']
    )

    # Create dataloader with SAME collate_fn as training
    data_loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),  # Can adjust batch size for inference
        shuffle=False,
        num_workers=config.get('num_workers', 0),  # Multiprocessing for data loading
        collate_fn=collate_fn_pad  # CRITICAL: Same padding as training!
    )

    print(f"✓ Created dataloader with {len(data_loader)} batches")

    # Run inference - SAME PIPELINE AS TRAINING EVALUATION
    results = inference_on_dataloader(
        model=model,
        data_loader=data_loader,
        device=device,
        omega_min=config['omega_min'],
        omega_max=config['omega_max'],
        omega_step=config['omega_step'],
        gamma=config['gamma']
    )

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    save_results(results, output_dir)
    plot_spectra_comparison(results, output_dir, max_plots=config.get('max_plots', 9))
    # plot_average_spectrum() removed - not applicable for randomly sampled frames

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Compute overall metrics
    all_corrs = [np.corrcoef(r['spectrum_pred'], r['spectrum_true'])[0, 1] for r in results]
    all_mse = [np.mean((r['spectrum_pred'] - r['spectrum_true'])**2) for r in results]
    all_h_mae = [np.mean(np.abs(r['H_diag_pred'] - r['H_diag_true'])) for r in results]

    print(f"\nSpectrum Correlation:")
    print(f"  Mean: {np.mean(all_corrs):.4f}")
    print(f"  Std:  {np.std(all_corrs):.4f}")
    print(f"  Min:  {np.min(all_corrs):.4f}")
    print(f"  Max:  {np.max(all_corrs):.4f}")

    print(f"\nSpectrum MSE:")
    print(f"  Mean: {np.mean(all_mse):.6f}")
    print(f"  Std:  {np.std(all_mse):.6f}")

    print(f"\nSite Energy MAE:")
    print(f"  Mean: {np.mean(all_h_mae):.2f} cm⁻¹")
    print(f"  Std:  {np.std(all_h_mae):.2f} cm⁻¹")

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey insight: This inference script now uses EXACTLY the same")
    print("computational pipeline as training, including:")
    print("  ✓ Batched processing with padding")
    print("  ✓ Torch-based spectrum generation")
    print("  ✓ Oscillator masking for padded atoms")
    print("  ✓ Same coupling calculation")
    print("\nResults should now match training validation perfectly!")


if __name__ == '__main__':
    main()

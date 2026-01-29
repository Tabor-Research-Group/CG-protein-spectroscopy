"""
Training loop with metrics and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import time
from physics import (
    calculate_torii_dipole_batch_torch,
    calculate_tasumi_coupling_torch,
    batch_generate_spectra_torch
)


class SpectrumLoss(nn.Module):
    """
    Loss function for spectrum matching.

    Combines:
    1. Spectrum MSE loss (main objective)
    2. Optional site energy regularization
    """

    def __init__(
        self,
        lambda_site_energy: float = 0.0,
        expected_site_energy_range: Tuple[float, float] = (1550.0, 1700.0)
    ):
        super().__init__()
        self.lambda_site_energy = lambda_site_energy
        self.expected_min, self.expected_max = expected_site_energy_range

    def forward(
        self,
        spectrum_pred: torch.Tensor,
        spectrum_true: torch.Tensor,
        H_diag_pred: torch.Tensor,
        oscillator_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss.

        Args:
            spectrum_pred: [B, M] predicted spectra
            spectrum_true: [B, M] ground truth spectra
            H_diag_pred: [B, N] predicted site energies
            oscillator_mask: [B, N] mask for valid oscillators

        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss components
        """
        # Spectrum MSE loss
        loss_spectrum = torch.mean((spectrum_pred - spectrum_true) ** 2)

        # Site energy regularization (optional)
        loss_site_energy = torch.tensor(0.0, device=H_diag_pred.device)

        if self.lambda_site_energy > 0:
            # Penalize site energies outside expected range
            too_low = torch.relu(self.expected_min - H_diag_pred)
            too_high = torch.relu(H_diag_pred - self.expected_max)
            penalty = (too_low + too_high) * oscillator_mask
            loss_site_energy = torch.sum(penalty) / torch.sum(oscillator_mask)

        # Total loss
        loss_total = loss_spectrum + self.lambda_site_energy * loss_site_energy

        loss_dict = {
            'total': loss_total.item(),
            'spectrum': loss_spectrum.item(),
            'site_energy_reg': loss_site_energy.item(),
        }

        return loss_total, loss_dict


def compute_metrics(
    spectrum_pred: torch.Tensor,
    spectrum_true: torch.Tensor,
    H_diag_pred: torch.Tensor,
    H_diag_true: torch.Tensor,
    oscillator_mask: torch.Tensor,
    omega_grid: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        spectrum_pred: [B, M]
        spectrum_true: [B, M]
        H_diag_pred: [B, N]
        H_diag_true: [B, N]
        oscillator_mask: [B, N]
        omega_grid: [M]

    Returns:
        Dictionary of metrics
    """
    B = spectrum_pred.shape[0]

    # Spectrum MSE
    spectrum_mse = torch.mean((spectrum_pred - spectrum_true) ** 2).item()

    # Spectrum correlation (per sample, then average)
    correlations = []
    for i in range(B):
        pred = spectrum_pred[i].cpu().numpy()
        true = spectrum_true[i].cpu().numpy()
        corr = np.corrcoef(pred, true)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    spectrum_corr = np.mean(correlations) if correlations else 0.0

    # Peak position error
    peak_errors = []
    omega_np = omega_grid.cpu().numpy()
    for i in range(B):
        pred = spectrum_pred[i].cpu().numpy()
        true = spectrum_true[i].cpu().numpy()

        peak_pred = omega_np[np.argmax(pred)]
        peak_true = omega_np[np.argmax(true)]

        peak_errors.append(abs(peak_pred - peak_true))

    peak_error = np.mean(peak_errors)

    # Site energy MAE (only valid oscillators)
    valid_mask = oscillator_mask > 0.5
    if valid_mask.sum() > 0:
        H_diff = torch.abs(H_diag_pred - H_diag_true) * oscillator_mask
        site_energy_mae = torch.sum(H_diff) / torch.sum(oscillator_mask)
        site_energy_mae = site_energy_mae.item()
    else:
        site_energy_mae = 0.0

    return {
        'spectrum_mse': spectrum_mse,
        'spectrum_corr': spectrum_corr,
        'peak_error_cm': peak_error,
        'site_energy_mae': site_energy_mae,
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: SpectrumLoss,
    device: torch.device,
    epoch: int,
    omega_grid: torch.Tensor,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: SiteEnergyPredictor
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        epoch: Current epoch number
        omega_grid: Frequency grid [M]
        log_interval: Print interval

    Returns:
        Dictionary with average metrics
    """
    model.train()

    losses = []
    metrics_accum = {
        'spectrum_mse': [],
        'spectrum_corr': [],
        'peak_error_cm': [],
        'site_energy_mae': [],
    }

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        own_features = batch['own_features'].to(device)
        neighbor_features = batch['neighbor_features'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        H_diag_true = batch['H_diag_true'].to(device)
        C_positions_pred = batch['C_positions_pred'].to(device)
        O_positions_pred = batch['O_positions_pred'].to(device)
        N_positions_pred = batch['N_positions_pred'].to(device)
        spectrum_true = batch['spectrum_true'].to(device)
        oscillator_mask = batch['oscillator_mask'].to(device)

        # Forward pass: predict site energies
        H_diag_pred = model(own_features, neighbor_features, neighbor_mask)

        # Calculate predicted dipoles (from predicted atoms)
        dipoles_pred = calculate_torii_dipole_batch_torch(
            C_positions_pred, O_positions_pred, N_positions_pred
        )

        # Calculate predicted coupling matrices
        B, N = H_diag_pred.shape
        J_matrix_pred = torch.zeros(B, N, N, device=device)

        for i in range(B):
            # Only compute for valid oscillators
            n_valid = int(oscillator_mask[i].sum().item())
            if n_valid > 0:
                J_matrix_pred[i, :n_valid, :n_valid] = calculate_tasumi_coupling_torch(
                    dipoles_pred[i, :n_valid],
                    C_positions_pred[i, :n_valid]
                )

        # Generate predicted spectra
        # CRITICAL FIX: Pass oscillator_mask to exclude padded oscillators
        spectrum_pred = batch_generate_spectra_torch(
            H_diag_pred, J_matrix_pred, dipoles_pred,
            mask_batch=oscillator_mask,  # FIXED: Now passing the mask to exclude padding
            omega_min=1500.0, omega_max=1750.0, omega_step=1.0, gamma=10.0
        )

        # Compute loss
        loss, loss_dict = criterion(spectrum_pred, spectrum_true, H_diag_pred, oscillator_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(
                spectrum_pred, spectrum_true, H_diag_pred, H_diag_true,
                oscillator_mask, omega_grid
            )

        losses.append(loss_dict['total'])
        for key in metrics_accum:
            metrics_accum[key].append(metrics[key])

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss_dict['total']:.6f} "
                  f"SpecMSE: {metrics['spectrum_mse']:.6f} "
                  f"Corr: {metrics['spectrum_corr']:.4f} "
                  f"PeakErr: {metrics['peak_error_cm']:.2f} cm⁻¹ "
                  f"({elapsed:.1f}s)")
            start_time = time.time()

    # Average metrics
    avg_metrics = {
        'loss': np.mean(losses),
        'spectrum_mse': np.mean(metrics_accum['spectrum_mse']),
        'spectrum_corr': np.mean(metrics_accum['spectrum_corr']),
        'peak_error_cm': np.mean(metrics_accum['peak_error_cm']),
        'site_energy_mae': np.mean(metrics_accum['site_energy_mae']),
    }

    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: SpectrumLoss,
    device: torch.device,
    omega_grid: torch.Tensor
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate model on test set.

    Args:
        model: Model
        test_loader: Test data loader
        criterion: Loss function
        device: Device
        omega_grid: Frequency grid

    Returns:
        avg_metrics: Dictionary with average metrics
        sample_results: List of sample results for visualization
    """
    model.eval()

    losses = []
    metrics_accum = {
        'spectrum_mse': [],
        'spectrum_corr': [],
        'peak_error_cm': [],
        'site_energy_mae': [],
    }

    sample_results = []

    for batch_idx, batch in enumerate(test_loader):
        # Move to device
        own_features = batch['own_features'].to(device)
        neighbor_features = batch['neighbor_features'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        H_diag_true = batch['H_diag_true'].to(device)
        C_positions_pred = batch['C_positions_pred'].to(device)
        O_positions_pred = batch['O_positions_pred'].to(device)
        N_positions_pred = batch['N_positions_pred'].to(device)
        spectrum_true = batch['spectrum_true'].to(device)
        oscillator_mask = batch['oscillator_mask'].to(device)

        # Forward pass
        H_diag_pred = model(own_features, neighbor_features, neighbor_mask)

        # Calculate dipoles
        dipoles_pred = calculate_torii_dipole_batch_torch(
            C_positions_pred, O_positions_pred, N_positions_pred
        )

        # Calculate couplings
        B, N = H_diag_pred.shape
        J_matrix_pred = torch.zeros(B, N, N, device=device)

        for i in range(B):
            n_valid = int(oscillator_mask[i].sum().item())
            if n_valid > 0:
                J_matrix_pred[i, :n_valid, :n_valid] = calculate_tasumi_coupling_torch(
                    dipoles_pred[i, :n_valid],
                    C_positions_pred[i, :n_valid]
                )

        # Generate spectra
        # CRITICAL FIX: Pass oscillator_mask to exclude padded oscillators
        spectrum_pred = batch_generate_spectra_torch(
            H_diag_pred, J_matrix_pred, dipoles_pred,
            mask_batch=oscillator_mask,  # FIXED: Now passing the mask to exclude padding
            omega_min=1500.0, omega_max=1750.0, omega_step=1.0, gamma=10.0
        )

        # Compute loss
        loss, loss_dict = criterion(spectrum_pred, spectrum_true, H_diag_pred, oscillator_mask)

        # Metrics
        metrics = compute_metrics(
            spectrum_pred, spectrum_true, H_diag_pred, H_diag_true,
            oscillator_mask, omega_grid
        )

        losses.append(loss_dict['total'])
        for key in metrics_accum:
            metrics_accum[key].append(metrics[key])

        # Save sample results (first 10 batches)
        if batch_idx < 10:
            for i in range(min(B, 5)):  # Save up to 5 samples per batch
                sample_results.append({
                    'spectrum_pred': spectrum_pred[i].cpu().numpy(),
                    'spectrum_true': spectrum_true[i].cpu().numpy(),
                    'H_diag_pred': H_diag_pred[i].cpu().numpy(),
                    'H_diag_true': H_diag_true[i].cpu().numpy(),
                    'frame_idx': batch['frame_indices'][i],
                })

    # Average metrics
    avg_metrics = {
        'loss': np.mean(losses),
        'spectrum_mse': np.mean(metrics_accum['spectrum_mse']),
        'spectrum_corr': np.mean(metrics_accum['spectrum_corr']),
        'peak_error_cm': np.mean(metrics_accum['peak_error_cm']),
        'site_energy_mae': np.mean(metrics_accum['site_energy_mae']),
    }

    return avg_metrics, sample_results


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_metrics: Dict,
    test_metrics: Dict,
    save_path: Path
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: Path
) -> Tuple[int, Dict, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_metrics = checkpoint['train_metrics']
    test_metrics = checkpoint['test_metrics']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, train_metrics, test_metrics

"""
Optimized training loop with SIMPLE, SCIENTIFICALLY SOUND loss function.

Key improvements:
1. REMOVED: Multi-scale smoothing (caused spectral shifts)
2. REMOVED: Multi-peak detection (unreliable, too harsh)
3. REMOVED: Gradient matching (too sensitive)
4. KEPT: Simple Correlation + MSE (naturally balanced, robust)

Loss = λ_corr × (1 - Pearson_correlation) + λ_mse × MSE

Rationale:
- Correlation ensures shape similarity (robust to small shifts)
- MSE ensures proper alignment (prevents shifts)
- Both naturally similar magnitude for normalized spectra (no complex balancing needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from physics import (
    calculate_torii_dipole_batch_torch,
    batch_generate_spectra_torch
)


def calculate_tasumi_coupling_batch_torch(dipoles: torch.Tensor, C_positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Batched Tasumi coupling calculation (OPTIMIZED - no Python loops).

    Args:
        dipoles: [B, N, 3]
        C_positions: [B, N, 3]
        mask: [B, N] - 1 for valid, 0 for padded

    Returns:
        J_matrix: [B, N, N]
    """
    B, N, _ = dipoles.shape
    device = dipoles.device

    # Pairwise distance vectors: r_ij = r_j - r_i
    r_ij = C_positions.unsqueeze(2) - C_positions.unsqueeze(1)

    # Distance magnitudes [B, N, N]
    r_mag = torch.norm(r_ij, dim=3)

    # For close pairs (<1 Å) and diagonal, use safe distance to prevent division issues
    # Note: Padded atoms are at z=1000 Å, so their distances are ~1000 Å (safe)
    r_mag_safe = torch.clamp(r_mag, min=1.0)  # Minimum distance 1 Å

    # Unit vectors [B, N, N, 3]
    r_unit = r_ij / r_mag_safe.unsqueeze(3)

    # Compute dot products for Tasumi formula
    mu_dot = torch.sum(dipoles.unsqueeze(2) * dipoles.unsqueeze(1), dim=3)  # [B, N, N]
    mu_i_dot_r = torch.sum(dipoles.unsqueeze(2) * r_unit, dim=3)  # [B, N, N]
    mu_j_dot_r = torch.sum(dipoles.unsqueeze(1) * r_unit, dim=3)  # [B, N, N]

    # Tasumi coupling formula: J = 5034 * [μ_i·μ_j / r³ - 3(μ_i·r)(μ_j·r) / r⁵]
    r3 = r_mag_safe**3
    r5 = r_mag_safe**5
    J = 5034.0 * (mu_dot / r3 - 3.0 * mu_i_dot_r * mu_j_dot_r / r5)

    # Zero out diagonal (self-coupling)
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    J = J * (1 - eye)

    # Zero out very close pairs (< 1 Angstrom) - physical cutoff
    close_mask = (r_mag >= 1.0).float()
    J = J * close_mask

    # Apply oscillator mask to zero out contributions from padded oscillators
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, N, N]
    J = J * mask_2d

    return J


class SpectrumLoss(nn.Module):
    """
    Simple, scientifically sound loss function for IR spectrum matching.

    Loss = λ_corr × (1 - Pearson_correlation) + λ_mse × MSE

    Design principles:
    1. Correlation captures shape similarity (robust to small shifts)
    2. MSE captures alignment and intensity (prevents shifts)
    3. Both naturally similar magnitude for normalized spectra
    4. No complicated peak matching (unreliable with overlapping peaks)
    5. No multi-scale smoothing (can cause unintended shifts)

    Args:
        lambda_peak: Weight for correlation loss (default: 1.0, ignores input value)
        lambda_correlation: Weight for MSE loss (default: 1.0, ignores input value)
        omega_grid: Frequency grid (kept for compatibility, not used in loss)
        peak_scale: Peak scale (kept for compatibility, not used in loss)
    """

    def __init__(
        self,
        lambda_peak: float = 0.5,
        lambda_correlation: float = 0.3,
        omega_grid: torch.Tensor = None,
        peak_scale: float = 100.0
    ):
        super().__init__()
        # Fixed weights for optimal balance
        self.lambda_corr = 1.0       # Weight for correlation loss
        self.lambda_mse = 1.0        # Weight for MSE loss
        self.lambda_grad = 500.0       # Weight for gradient loss (prevents flat/mean predictions)

        # Keep these for compatibility
        self.register_buffer('omega_grid', omega_grid)
        self.peak_scale = peak_scale

        print(f"\n  SpectrumLoss initialized:")
        print(f"    Loss = {self.lambda_corr:.1f} × (1 - corr) + {self.lambda_mse:.1f} × MSE + {self.lambda_grad:.1f} × Gradient_MSE")
        print(f"    Correlation: shape similarity (robust to shifts)")
        print(f"    MSE: alignment & intensity (prevents shifts)")
        print(f"    Gradient: spectral structure (peaks/valleys, prevents flat/mean predictions)")

    def forward(
        self,
        spectrum_pred: torch.Tensor,
        spectrum_true: torch.Tensor,
        H_diag_pred: torch.Tensor = None,
        oscillator_mask: torch.Tensor = None,
        H_diag_true: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute spectrum matching loss with H_diag supervision.

        Args:
            spectrum_pred: [B, N_freq] predicted spectra (normalized 0-1)
            spectrum_true: [B, N_freq] true spectra (normalized 0-1)
            H_diag_pred: [B, N] predicted site energies (cm^-1)
            oscillator_mask: [B, N] mask for valid oscillators
            H_diag_true: [B, N] true site energies (cm^-1)

        Returns:
            loss: Total loss scalar
            loss_dict: Dictionary with loss components
        """
        B, N_freq = spectrum_pred.shape

        # 1. MSE loss (point-wise matching)
        # Penalizes shifts, intensity differences, baseline errors
        loss_mse = F.mse_loss(spectrum_pred, spectrum_true)

        # 2. Correlation loss (shape matching)
        # Compute Pearson correlation for each sample in batch
        correlations = []
        for i in range(B):
            pred = spectrum_pred[i]
            true = spectrum_true[i]

            # Center the spectra (subtract mean)
            pred_centered = pred - pred.mean()
            true_centered = true - true.mean()

            # Pearson correlation coefficient
            numerator = torch.sum(pred_centered * true_centered)
            denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(true_centered**2))

            # Prevent division by zero
            corr = numerator / (denominator + 1e-8)

            # Clamp to valid range [-1, 1] to handle numerical errors
            corr = torch.clamp(corr, -1.0, 1.0)

            correlations.append(corr)

        # Average correlation across batch
        correlation = torch.mean(torch.stack(correlations))

        # Correlation loss: 1 - correlation
        # Want to maximize correlation, so minimize (1 - correlation)
        loss_corr = 1.0 - correlation

        # 3. Gradient loss (spectral structure)
        # This ensures peaks and valleys are in correct positions, prevents flat/mean predictions
        # Compute first derivative of spectra (finite difference)
        grad_pred = spectrum_pred[:, 1:] - spectrum_pred[:, :-1]  # [B, N_freq-1]
        grad_true = spectrum_true[:, 1:] - spectrum_true[:, :-1]  # [B, N_freq-1]

        # MSE on gradients
        loss_grad = F.mse_loss(grad_pred, grad_true)

        # 4. Total loss (weighted combination)
        loss_total = self.lambda_corr * loss_corr + self.lambda_mse * loss_mse + self.lambda_grad * loss_grad
      
       # loss_total = self.lambda_mse * loss_mse 

        # Prepare loss dictionary
        loss_dict = {
            'total': loss_total.item(),
            'mse': loss_mse.item(),
            'correlation': correlation.item(),
            'corr_loss': loss_corr.item(),
            'grad_loss': loss_grad.item(),
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
        spectrum_pred: [B, N_freq] predicted spectra
        spectrum_true: [B, N_freq] true spectra
        H_diag_pred: [B, N] predicted site energies (cm^-1)
        H_diag_true: [B, N] true site energies (cm^-1)
        oscillator_mask: [B, N] mask for valid oscillators
        omega_grid: [N_freq] frequency grid

    Returns:
        Dictionary of metrics
    """
    B = spectrum_pred.shape[0]

    # Spectrum MSE
    spectrum_mse = torch.mean((spectrum_pred - spectrum_true) ** 2).item()

    # Spectrum correlation (Pearson)
    correlations = []
    for i in range(B):
        pred = spectrum_pred[i].cpu().numpy()
        true = spectrum_true[i].cpu().numpy()
        corr = np.corrcoef(pred, true)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    spectrum_corr = np.mean(correlations) if correlations else 0.0

    # Peak position error (using argmax - simple, robust)
    peak_errors = []
    omega_np = omega_grid.cpu().numpy()
    for i in range(B):
        pred = spectrum_pred[i].cpu().numpy()
        true = spectrum_true[i].cpu().numpy()
        peak_pred = omega_np[np.argmax(pred)]
        peak_true = omega_np[np.argmax(true)]
        peak_errors.append(abs(peak_pred - peak_true))
    peak_error = np.mean(peak_errors)

    # Site energy MAE (only for valid oscillators)
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
    scheduler=None,
) -> Dict[str, float]:
    """
    Train for one epoch with progress bar and diagnostics.

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (SpectrumLoss)
        device: torch device
        epoch: Current epoch number
        omega_grid: Frequency grid tensor
        scheduler: Optional learning rate scheduler

    Returns:
        Dictionary of averaged training metrics
    """
    model.train()

    # Accumulators for metrics
    losses = []
    loss_mse_accum = []
    loss_corr_accum = []
    loss_grad_accum = []
    metrics_accum = {
        'spectrum_mse': [],
        'spectrum_corr': [],
        'peak_error_cm': [],
        'site_energy_mae': [],
    }
    grad_norms = []
    skipped_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)

    for batch in pbar:
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

        # Input feature sanity check
        own_max = own_features.abs().max().item()
        neighbor_max = neighbor_features.abs().max().item()

        if own_max > 1000.0 or neighbor_max > 1000.0:
            print(f"\n⚠️  EXTREME INPUT FEATURES at batch {len(losses)+1}!")
            print(f"    own_features max: {own_max:.2f}")
            print(f"    neighbor_features max: {neighbor_max:.2f}")
            print(f"    Suggests feature normalization issue or padding leak")

        optimizer.zero_grad()

        # Forward pass: predict H_diag
        H_diag_pred = model(own_features, neighbor_features, neighbor_mask)

        # NaN/Inf detection in model output
        if torch.isnan(H_diag_pred).any() or torch.isinf(H_diag_pred).any():
            print(f"\n❌ NaN/Inf in H_diag_pred at batch {len(losses)+1}!")
            print(f"    H_diag_pred range: [{H_diag_pred.min().item():.2f}, {H_diag_pred.max().item():.2f}]")
            print(f"    Indicates numerical instability - SKIPPING batch")
            skipped_batches += 1
            continue

        # Calculate dipoles from predicted atom positions
        dipoles_pred = calculate_torii_dipole_batch_torch(
            C_positions_pred, O_positions_pred, N_positions_pred
        )

        # Calculate couplings (vectorized, no Python loops)
        J_matrix_pred = calculate_tasumi_coupling_batch_torch(
            dipoles_pred, C_positions_pred, oscillator_mask
        )

        # Generate IR spectrum from H_diag, J_matrix, dipoles
        spectrum_pred = batch_generate_spectra_torch(
            H_diag_pred, J_matrix_pred, dipoles_pred,
            mask_batch=oscillator_mask,
            omega_min=1500.0, omega_max=1750.0, omega_step=1.0, gamma=10.0
        )

        # Compute loss: Correlation + MSE + H_diag
        loss, loss_dict = criterion(spectrum_pred, spectrum_true, H_diag_pred, oscillator_mask, H_diag_true)

        # Backward pass
        loss.backward()

        # Gradient monitoring
        total_grad_norm = 0.0
        max_grad = 0.0
        has_nan_grad = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                max_grad = max(max_grad, grad_norm)

                if torch.isnan(param.grad).any():
                    has_nan_grad = True

        total_grad_norm = total_grad_norm ** 0.5

        # Skip batch if NaN or extreme gradients
        GRADIENT_THRESHOLD = 100.0
        if has_nan_grad or total_grad_norm > GRADIENT_THRESHOLD:
            print(f"\n⚠️  Skipping batch {len(losses)+1}: grad_norm={total_grad_norm:.2f}")
            optimizer.zero_grad()
            skipped_batches += 1
            continue

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Update learning rate scheduler (if provided)
        if scheduler is not None:
            scheduler.step()

        # Compute metrics for monitoring
        with torch.no_grad():
            metrics = compute_metrics(
                spectrum_pred, spectrum_true, H_diag_pred, H_diag_true,
                oscillator_mask, omega_grid
            )

        # Accumulate loss components and metrics
        losses.append(loss_dict['total'])
        loss_mse_accum.append(loss_dict['mse'])
        loss_corr_accum.append(loss_dict['corr_loss'])
        loss_grad_accum.append(loss_dict['grad_loss'])
        for key in metrics_accum:
            metrics_accum[key].append(metrics[key])
        grad_norms.append(total_grad_norm)

        # Update progress bar with key metrics
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'corr': f"{metrics['spectrum_corr']:.3f}",
            'mse': f"{loss_dict['mse']:.4f}"
        })

    # Compute epoch averages
    avg_metrics = {
        'loss': np.mean(losses) if losses else 0.0,
        'loss_mse_component': np.mean(loss_mse_accum) if loss_mse_accum else 0.0,
        'loss_corr_component': np.mean(loss_corr_accum) if loss_corr_accum else 0.0,
        'loss_grad_component': np.mean(loss_grad_accum) if loss_grad_accum else 0.0,
        'spectrum_mse': np.mean(metrics_accum['spectrum_mse']) if metrics_accum['spectrum_mse'] else 0.0,
        'spectrum_corr': np.mean(metrics_accum['spectrum_corr']) if metrics_accum['spectrum_corr'] else 0.0,
        'peak_error_cm': np.mean(metrics_accum['peak_error_cm']) if metrics_accum['peak_error_cm'] else 0.0,
        'site_energy_mae': np.mean(metrics_accum['site_energy_mae']) if metrics_accum['site_energy_mae'] else 0.0,
    }

    # Print detailed epoch statistics
    print(f"\n  Epoch {epoch} Training Statistics:")
    print(f"    Loss Components:")
    print(f"      Total Loss:      {avg_metrics['loss']:.6f}")
    print(f"      MSE component:   {avg_metrics['loss_mse_component']:.6f} (alignment)")
    print(f"      Corr component:  {avg_metrics['loss_corr_component']:.6f} (shape)")
    print(f"      Grad component:  {avg_metrics['loss_grad_component']:.6f} (structure)")
    print(f"    Spectrum Metrics:")
    print(f"      Correlation:    {avg_metrics['spectrum_corr']:.4f} (0=bad, 1=perfect)")
    print(f"      MSE:            {avg_metrics['spectrum_mse']:.6f}")
    print(f"      Peak Error:     {avg_metrics['peak_error_cm']:.2f} cm⁻¹")
    print(f"    H_diag Prediction:")
    print(f"      MAE:            {avg_metrics['site_energy_mae']:.2f} cm⁻¹")

    # Gradient statistics
    if len(grad_norms) > 0:
        print(f"    Gradient Norms:")
        print(f"      Mean: {np.mean(grad_norms):.4f}, Max: {np.max(grad_norms):.4f}, Std: {np.std(grad_norms):.4f}")

    if skipped_batches > 0:
        print(f"    ⚠️  Skipped {skipped_batches} batches due to NaN/extreme gradients")

    # Loss balance check
    if avg_metrics['loss_mse_component'] > 0 and avg_metrics['loss_corr_component'] > 0:
        ratio = avg_metrics['loss_mse_component'] / avg_metrics['loss_corr_component']
        if ratio < 0.1 or ratio > 10.0:
            print(f"    ⚠️  Loss imbalance detected: MSE/Corr ratio = {ratio:.2f}")
            print(f"       (Optimal range: 0.1-10.0)")

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
    Evaluate model on test set with detailed diagnostics.

    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function (SpectrumLoss)
        device: torch device
        omega_grid: Frequency grid tensor

    Returns:
        avg_metrics: Dictionary of averaged metrics
        sample_results: List of sample predictions for visualization
    """
    model.eval()

    # Accumulators
    losses = []
    loss_mse_accum = []
    loss_corr_accum = []
    loss_grad_accum = []
    metrics_accum = {
        'spectrum_mse': [],
        'spectrum_corr': [],
        'peak_error_cm': [],
        'site_energy_mae': [],
    }

    sample_results = []

    pbar = tqdm(test_loader, desc='Evaluating', leave=False)

    for batch_idx, batch in enumerate(pbar):
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

        # Calculate couplings (vectorized)
        J_matrix_pred = calculate_tasumi_coupling_batch_torch(
            dipoles_pred, C_positions_pred, oscillator_mask
        )

        # Generate spectra
        spectrum_pred = batch_generate_spectra_torch(
            H_diag_pred, J_matrix_pred, dipoles_pred,
            mask_batch=oscillator_mask,
            omega_min=1500.0, omega_max=1750.0, omega_step=1.0, gamma=10.0
        )

        # Compute loss
        loss, loss_dict = criterion(spectrum_pred, spectrum_true, H_diag_pred, oscillator_mask, H_diag_true)

        # Compute metrics
        metrics = compute_metrics(
            spectrum_pred, spectrum_true, H_diag_pred, H_diag_true,
            oscillator_mask, omega_grid
        )

        # Accumulate
        losses.append(loss_dict['total'])
        loss_mse_accum.append(loss_dict['mse'])
        loss_corr_accum.append(loss_dict['corr_loss'])
        loss_grad_accum.append(loss_dict['grad_loss'])
        for key in metrics_accum:
            metrics_accum[key].append(metrics[key])

        # Update progress bar
        pbar.set_postfix({
            'corr': f"{metrics['spectrum_corr']:.3f}",
            'mse': f"{loss_dict['mse']:.4f}"
        })

        # Collect samples from ALL batches for representative visualization
        B = spectrum_pred.shape[0]
        for i in range(B):
            frame_idx = batch['frame_idx'][i].item() if 'frame_idx' in batch else batch_idx * B + i

            sample_results.append({
                'frame_idx': frame_idx,
                'spectrum_pred': spectrum_pred[i].cpu().numpy(),
                'spectrum_true': spectrum_true[i].cpu().numpy(),
                'H_diag_pred': H_diag_pred[i].cpu().numpy(),
                'H_diag_true': H_diag_true[i].cpu().numpy(),
            })

    # Compute averages
    avg_metrics = {
        'loss': np.mean(losses) if losses else 0.0,
        'loss_mse_component': np.mean(loss_mse_accum) if loss_mse_accum else 0.0,
        'loss_corr_component': np.mean(loss_corr_accum) if loss_corr_accum else 0.0,
        'loss_grad_component': np.mean(loss_grad_accum) if loss_grad_accum else 0.0,
        'spectrum_mse': np.mean(metrics_accum['spectrum_mse']) if metrics_accum['spectrum_mse'] else 0.0,
        'spectrum_corr': np.mean(metrics_accum['spectrum_corr']) if metrics_accum['spectrum_corr'] else 0.0,
        'peak_error_cm': np.mean(metrics_accum['peak_error_cm']) if metrics_accum['peak_error_cm'] else 0.0,
        'site_energy_mae': np.mean(metrics_accum['site_energy_mae']) if metrics_accum['site_energy_mae'] else 0.0,
    }

    # Random sample for plotting (limit to 50 frames for efficiency)
    if len(sample_results) > 50:
        import random
        random.seed(42)  # Reproducible sampling
        sample_results = random.sample(sample_results, 50)

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
    print(f"  Checkpoint saved to {save_path}")

"""
Physics calculations for amide I spectroscopy.

Implements:
1. Torii dipole calculation with correct formula
2. Tasumi transition dipole coupling (TDC)
3. Spectrum generation from Hamiltonian
"""

import numpy as np
import torch
from typing import Tuple, Optional


def calculate_torii_dipole_numpy(C: np.ndarray, O: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Calculate Torii dipole with correct formula (NumPy version).

    Formula:
        μ = 0.276 * (s - ((CO·s) + sqrt(|s|^2 - (CO·s)^2) / tan(10°)) * CO)
    where:
        s = 0.665*CO + 0.258*CN
        CO, CN are normalized vectors

    Args:
        C: Carbon position [3]
        O: Oxygen position [3]
        N: Nitrogen position [3]

    Returns:
        mu: Dipole vector in Debye [3]
    """
    # Vectors
    CO = O - C
    CN = N - C

    # Normalize
    CO_norm = np.linalg.norm(CO)
    CN_norm = np.linalg.norm(CN)

    if CO_norm < 1e-6 or CN_norm < 1e-6:
        return np.zeros(3, dtype=np.float32)

    CO_unit = CO / CO_norm
    CN_unit = CN / CN_norm

    # s vector
    s = 0.665 * CO_unit + 0.258 * CN_unit

    # CO · s
    CO_dot_s = np.dot(CO_unit, s)

    # |s|^2
    s_mag_sq = np.dot(s, s)

    # sqrt(|s|^2 - (CO · s)^2)
    discriminant = s_mag_sq - CO_dot_s**2
    if discriminant < 0:
        discriminant = 0  # numerical safety
    sqrt_term = np.sqrt(discriminant)

    # tan(10°)
    tan_10 = np.tan(np.radians(10.0))

    # Full formula with AIM prefactor (0.276 instead of 2.73)
    mu = 0.276 * (s - (CO_dot_s + sqrt_term / tan_10) * CO_unit)

    return mu.astype(np.float32)


def calculate_torii_dipole_batch_numpy(C: np.ndarray, O: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Vectorized Torii dipole calculation for multiple oscillators (NumPy).

    Args:
        C: Carbon positions [N, 3]
        O: Oxygen positions [N, 3]
        N: Nitrogen positions [N, 3]

    Returns:
        mu: Dipole vectors [N, 3]
    """
    # Vectors
    CO = O - C  # [N, 3]
    CN = N - C  # [N, 3]

    # Normalize
    CO_norm = np.linalg.norm(CO, axis=1, keepdims=True)  # [N, 1]
    CN_norm = np.linalg.norm(CN, axis=1, keepdims=True)  # [N, 1]

    # Avoid division by zero
    CO_norm = np.maximum(CO_norm, 1e-6)
    CN_norm = np.maximum(CN_norm, 1e-6)

    CO_unit = CO / CO_norm  # [N, 3]
    CN_unit = CN / CN_norm  # [N, 3]

    # s vector
    s = 0.665 * CO_unit + 0.258 * CN_unit  # [N, 3]

    # CO · s
    CO_dot_s = np.sum(CO_unit * s, axis=1, keepdims=True)  # [N, 1]

    # |s|^2
    s_mag_sq = np.sum(s * s, axis=1, keepdims=True)  # [N, 1]

    # sqrt(|s|^2 - (CO · s)^2)
    discriminant = s_mag_sq - CO_dot_s**2
    discriminant = np.maximum(discriminant, 0)  # numerical safety
    sqrt_term = np.sqrt(discriminant)  # [N, 1]

    # tan(10°)
    tan_10 = np.tan(np.radians(10.0))

    # Full formula with AIM prefactor
    mu = 0.276 * (s - (CO_dot_s + sqrt_term / tan_10) * CO_unit)  # [N, 3]

    return mu.astype(np.float32)


def calculate_torii_dipole_batch_torch(C: torch.Tensor, O: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Torii dipole calculation for multiple oscillators (PyTorch).

    Args:
        C: Carbon positions [N, 3]
        O: Oxygen positions [N, 3]
        N: Nitrogen positions [N, 3]

    Returns:
        mu: Dipole vectors [N, 3]
    """
    # Vectors
    CO = O - C  # [N, 3]
    CN = N - C  # [N, 3]

    # Normalize
    CO_norm = torch.norm(CO, dim=1, keepdim=True)  # [N, 1]
    CN_norm = torch.norm(CN, dim=1, keepdim=True)  # [N, 1]

    # Avoid division by zero
    CO_norm = torch.clamp(CO_norm, min=1e-6)
    CN_norm = torch.clamp(CN_norm, min=1e-6)

    CO_unit = CO / CO_norm  # [N, 3]
    CN_unit = CN / CN_norm  # [N, 3]

    # s vector
    s = 0.665 * CO_unit + 0.258 * CN_unit  # [N, 3]

    # CO · s
    CO_dot_s = torch.sum(CO_unit * s, dim=1, keepdim=True)  # [N, 1]

    # |s|^2
    s_mag_sq = torch.sum(s * s, dim=1, keepdim=True)  # [N, 1]

    # sqrt(|s|^2 - (CO · s)^2)
    discriminant = s_mag_sq - CO_dot_s**2
    discriminant = torch.clamp(discriminant, min=0)  # numerical safety
    sqrt_term = torch.sqrt(discriminant)  # [N, 1]

    # tan(10°)
    tan_10 = np.tan(np.radians(10.0))

    # Full formula with AIM prefactor
    mu = 0.276 * (s - (CO_dot_s + sqrt_term / tan_10) * CO_unit)  # [N, 3]

    return mu


def calculate_tasumi_coupling_numpy(dipoles: np.ndarray, C_positions: np.ndarray) -> np.ndarray:
    """
    Calculate Tasumi transition dipole coupling (TDC) matrix.

    Formula:
        J_ij = 5034 * (
            (m_i · m_j) / |r_ij|^3 -
            3 * (m_i · r_ij) * (m_j · r_ij) / |r_ij|^5
        )

    Args:
        dipoles: Dipole vectors [N, 3] in Debye
        C_positions: Carbon positions [N, 3] in Angstroms

    Returns:
        J: Coupling matrix [N, N] in cm^-1
    """
    N = len(dipoles)

    # Pairwise distance vectors: r_ij = r_j - r_i
    r_ij = C_positions[:, np.newaxis, :] - C_positions[np.newaxis, :, :]  # [N, N, 3]

    # Distance magnitudes
    r_mag = np.linalg.norm(r_ij, axis=2)  # [N, N]

    # Avoid division by zero on diagonal
    r_mag_safe = np.where(r_mag > 1e-6, r_mag, 1.0)

    # r_ij / |r_ij| (unit vectors)
    r_unit = r_ij / r_mag_safe[:, :, np.newaxis]  # [N, N, 3]

    # m_i · m_j
    mu_dot = np.sum(dipoles[:, np.newaxis, :] * dipoles[np.newaxis, :, :], axis=2)  # [N, N]

    # m_i · r_ij
    mu_i_dot_r = np.sum(dipoles[:, np.newaxis, :] * r_unit, axis=2)  # [N, N]

    # m_j · r_ij
    mu_j_dot_r = np.sum(dipoles[np.newaxis, :, :] * r_unit, axis=2)  # [N, N]

    # Coupling formula
    r3 = r_mag_safe**3
    r5 = r_mag_safe**5

    J = 5034.0 * (mu_dot / r3 - 3.0 * mu_i_dot_r * mu_j_dot_r / r5)  # [N, N]

    # Set diagonal to zero (self-coupling is not included)
    np.fill_diagonal(J, 0.0)

    # Set very close oscillators (< 1 Angstrom) to zero
    J = np.where(r_mag < 1.0, 0.0, J)

    return J.astype(np.float32)


def calculate_tasumi_coupling_torch(dipoles: torch.Tensor, C_positions: torch.Tensor) -> torch.Tensor:
    """
    Calculate Tasumi TDC matrix (PyTorch version).

    Args:
        dipoles: Dipole vectors [N, 3]
        C_positions: Carbon positions [N, 3]

    Returns:
        J: Coupling matrix [N, N]
    """
    N = dipoles.shape[0]

    # Pairwise distance vectors
    r_ij = C_positions.unsqueeze(1) - C_positions.unsqueeze(0)  # [N, N, 3]

    # Distance magnitudes
    r_mag = torch.norm(r_ij, dim=2)  # [N, N]

    # Avoid division by zero
    r_mag_safe = torch.where(r_mag > 1e-6, r_mag, torch.ones_like(r_mag))

    # Unit vectors
    r_unit = r_ij / r_mag_safe.unsqueeze(2)  # [N, N, 3]

    # m_i · m_j
    mu_dot = torch.sum(dipoles.unsqueeze(1) * dipoles.unsqueeze(0), dim=2)  # [N, N]

    # m_i · r_ij and m_j · r_ij
    mu_i_dot_r = torch.sum(dipoles.unsqueeze(1) * r_unit, dim=2)  # [N, N]
    mu_j_dot_r = torch.sum(dipoles.unsqueeze(0) * r_unit, dim=2)  # [N, N]

    # Coupling formula
    r3 = r_mag_safe**3
    r5 = r_mag_safe**5

    J = 5034.0 * (mu_dot / r3 - 3.0 * mu_i_dot_r * mu_j_dot_r / r5)

    # Zero out diagonal and close pairs
    mask = (r_mag >= 1.0)
    J = J * mask.float()
    J = J - torch.diag(torch.diag(J))  # ensure diagonal is zero

    return J


def generate_spectrum_numpy(
    H_diag: np.ndarray,
    J_matrix: np.ndarray,
    dipoles: np.ndarray,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate amide I spectrum from Hamiltonian.

    Steps:
        1. Build full Hamiltonian: H = diag(H_diag) + J_matrix
        2. Diagonalize to get eigenvalues and eigenvectors
        3. Calculate transition dipole moments
        4. Apply Lorentzian broadening

    Args:
        H_diag: Site energies [N] in cm^-1
        J_matrix: Coupling matrix [N, N] in cm^-1
        dipoles: Dipole vectors [N, 3] in Debye
        omega_min: Minimum frequency (cm^-1)
        omega_max: Maximum frequency (cm^-1)
        omega_step: Frequency resolution (cm^-1)
        gamma: Lorentzian width (cm^-1)

    Returns:
        omega_grid: Frequency grid [M]
        spectrum: Intensity [M]
    """
    N = len(H_diag)

    # Build Hamiltonian
    H = np.diag(H_diag) + J_matrix  # [N, N]

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)  # eigenvalues [N], eigenvectors [N, N]

    # Calculate transition dipole moments
    # μ_α = Σ_i c_{iα} * μ_i
    # where c_{iα} is eigenvector coefficient
    transition_dipoles = eigenvectors.T @ dipoles  # [N, 3]

    # Transition strengths: |μ_α|^2
    strengths = np.sum(transition_dipoles**2, axis=1)  # [N]

    # Frequency grid
    omega_grid = np.arange(omega_min, omega_max + omega_step, omega_step)  # [M]

    # Lorentzian broadening
    # I(ω) = Σ_α A_α * γ / ((ω - ω_α)^2 + γ^2)
    spectrum = np.zeros_like(omega_grid)

    for i in range(N):
        lorentzian = gamma / ((omega_grid - eigenvalues[i])**2 + gamma**2)
        spectrum += strengths[i] * lorentzian

    # Normalize
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)

    return omega_grid, spectrum


def generate_spectrum_torch(
    H_diag: torch.Tensor,
    J_matrix: torch.Tensor,
    dipoles: torch.Tensor,
    mask: torch.Tensor = None,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate spectrum (PyTorch version, differentiable).

    Args:
        H_diag: Site energies [N]
        J_matrix: Coupling matrix [N, N]
        dipoles: Dipole vectors [N, 3]
        mask: Oscillator mask [N] - 1 for valid, 0 for padded (optional)
        omega_min, omega_max, omega_step: Frequency grid parameters
        gamma: Lorentzian width

    Returns:
        omega_grid: Frequency grid [M]
        spectrum: Intensity [M]
    """
    N_full = H_diag.shape[0]
    device = H_diag.device

    # If mask is provided, only use valid oscillators
    if mask is not None:
        valid_indices = torch.where(mask > 0)[0]
        if len(valid_indices) == 0:
            # No valid oscillators - return zero spectrum
            omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
            return omega_grid, torch.zeros_like(omega_grid)

        # Extract only valid oscillators
        H_diag = H_diag[valid_indices]
        J_matrix = J_matrix[valid_indices, :][:, valid_indices]
        dipoles = dipoles[valid_indices, :]

    N = H_diag.shape[0]

    # CHECK FOR NaN/Inf BEFORE building Hamiltonian (catches model output issues)
    if torch.isnan(H_diag).any() or torch.isinf(H_diag).any():
        print(f"ERROR: NaN/Inf detected in H_diag before building Hamiltonian!")
        print(f"  H_diag range: [{H_diag.min().item() if not torch.isnan(H_diag).all() else 'all NaN'}, {H_diag.max().item() if not torch.isnan(H_diag).all() else 'all NaN'}]")
        print(f"  J_matrix range: [{J_matrix.min().item():.2f}, {J_matrix.max().item():.2f}]")
        print(f"  This indicates model output is NaN (gradient explosion or numerical instability)")
        # Return zero spectrum to allow training to continue
        omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
        return omega_grid, torch.zeros_like(omega_grid)

    if torch.isnan(J_matrix).any() or torch.isinf(J_matrix).any():
        print(f"ERROR: NaN/Inf detected in J_matrix!")
        print(f"  H_diag range: [{H_diag.min().item():.2f}, {H_diag.max().item():.2f}]")
        print(f"  J_matrix range: [{J_matrix.min().item() if not torch.isnan(J_matrix).all() else 'all NaN'}, {J_matrix.max().item() if not torch.isnan(J_matrix).all() else 'all NaN'}]")
        # Return zero spectrum
        omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
        return omega_grid, torch.zeros_like(omega_grid)

    # Build Hamiltonian
    H = torch.diag(H_diag) + J_matrix  # [N, N]

    # Check Hamiltonian for NaN (shouldn't happen if inputs are clean, but double-check)
    if torch.isnan(H).any() or torch.isinf(H).any():
        print(f"ERROR: NaN/Inf in Hamiltonian matrix after construction!")
        print(f"  H range: [{H.min().item() if not torch.isnan(H).all() else 'all NaN'}, {H.max().item() if not torch.isnan(H).all() else 'all NaN'}]")
        omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
        return omega_grid, torch.zeros_like(omega_grid)

    # Diagonalize (use symeig for differentiability)
    # Add small regularization for numerical stability
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(H)  # [N], [N, N]
    except RuntimeError as e:
        # If eigh fails, try with small regularization
        print(f"WARNING: eigh failed, adding regularization. Error: {e}")
        print(f"  H_diag range: [{H_diag.min().item():.2f}, {H_diag.max().item():.2f}]")
        print(f"  J_matrix range: [{J_matrix.min().item():.2f}, {J_matrix.max().item():.2f}]")
        print(f"  H_matrix range: [{H.min().item():.2f}, {H.max().item():.2f}]")

        # Add small regularization to diagonal
        eps = 1e-6
        H_reg = H + eps * torch.eye(N, device=device)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)  # [N], [N, N]
        except RuntimeError as e2:
            print(f"ERROR: eigh still failed after regularization: {e2}")
            print(f"  Returning zero spectrum to allow training to continue")
            omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
            return omega_grid, torch.zeros_like(omega_grid)

    # Transition dipoles
    transition_dipoles = torch.matmul(eigenvectors.T, dipoles)  # [N, 3]

    # Strengths
    strengths = torch.sum(transition_dipoles**2, dim=1)  # [N]

    # Frequency grid
    omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)  # [M]

    # Lorentzian broadening (vectorized)
    # omega_grid: [M], eigenvalues: [N]
    # Create [M, N] grid
    omega_expand = omega_grid.unsqueeze(1)  # [M, 1]
    eigen_expand = eigenvalues.unsqueeze(0)  # [1, N]

    lorentzian = gamma / ((omega_expand - eigen_expand)**2 + gamma**2)  # [M, N]

    # Weighted sum
    spectrum = torch.matmul(lorentzian, strengths)  # [M]

    # Normalize
    max_val = torch.max(spectrum)
    if max_val > 0:
        spectrum = spectrum / max_val

    return omega_grid, spectrum


def batch_generate_spectra_torch(
    H_diag_batch: torch.Tensor,
    J_matrix_batch: torch.Tensor,
    dipoles_batch: torch.Tensor,
    mask_batch: torch.Tensor = None,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0
) -> torch.Tensor:
    """
    Generate spectra for a batch (for training).

    Args:
        H_diag_batch: [B, N_i] site energies for B frames (padded)
        J_matrix_batch: [B, N_i, N_i] coupling matrices
        dipoles_batch: [B, N_i, 3] dipole vectors
        mask_batch: [B, N_i] oscillator mask (1=valid, 0=padded) - CRITICAL for excluding padded oscillators

    Returns:
        spectra_batch: [B, M] where M is number of frequency points
    """
    B = H_diag_batch.shape[0]
    device = H_diag_batch.device

    # Frequency grid (same for all)
    omega_grid = torch.arange(omega_min, omega_max + omega_step, omega_step, device=device)
    M = len(omega_grid)

    spectra = []

    for i in range(B):
        # CRITICAL FIX: Pass the mask to exclude padded oscillators from eigenvalue decomposition
        # Without this, padded atoms at z=1000 Å cause numerical instability and NaN values
        current_mask = mask_batch[i] if mask_batch is not None else None
        _, spectrum = generate_spectrum_torch(
            H_diag_batch[i],
            J_matrix_batch[i],
            dipoles_batch[i],
            mask=current_mask,  # FIXED: Now passing the mask
            omega_min=omega_min,
            omega_max=omega_max,
            omega_step=omega_step,
            gamma=gamma
        )
        spectra.append(spectrum)

    return torch.stack(spectra, dim=0)  # [B, M]

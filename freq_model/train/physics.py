"""
Physics calculations for amide I spectroscopy.
"""

import warnings

import numpy as np
import torch
from typing import Tuple, Dict
from pathlib import Path

# tan(10°)
tan_10 = np.tan(np.radians(10.0))


NNC_map = {}
mapfile = Path(__file__).parent / 'nnc_map.dat'
with open(mapfile) as f:
    for map_num in range(5):
        mapname = f.readline().strip()
        mapdata = []
        for _ in range(13):
            mapdata.append(f.readline().strip())
        mapdata = np.loadtxt(mapdata)
        NNC_map[mapname] = mapdata

def calculate_torii_dipole(C: np.ndarray, O: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Calculate Torii dipole (NumPy version).

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

    # Full formula with AIM prefactor
    mu = s - (CO_dot_s + sqrt_term / tan_10) * CO_unit
    mu = mu / np.linalg.norm(mu)
    mu = mu * 0.276

    return mu.astype(np.float32)


def calculate_torii_dipole_batch(C: np.ndarray, O: np.ndarray, N: np.ndarray) -> np.ndarray:
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

    # Full formula with AIM prefactor
    mu = s - (CO_dot_s + sqrt_term / tan_10) * CO_unit  # [N, 3]
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    mu = mu * 0.276

    return mu.astype(np.float32)

def calc_dihedral(pos1: np.ndarray, pos2: np.ndarray, pos3: np.ndarray, pos4: np.ndarray) -> float:
    
    u1 = pos2 - pos1
    u2 = pos3 - pos2
    u3 = pos4 - pos3
    
    cross12 = np.cross(u1, u2)
    cross23 = np.cross(u2, u3)
    
    norm2 = np.linalg.norm(u2)
    
    y = np.dot((norm2 * u1), cross23)
    x = np.dot(cross12, cross23)
    
    return np.rad2deg(np.atan2(y, x))
    

def calculate_coupling_matrix(data: Dict, use_predicted: bool = False) -> np.ndarray:
    
    J = calculate_TDC(data['dipoles'], data['C_positions'], data['N_positions'], data['O_positions'])
    
    backbone_indices = np.where(np.logical_or(data['oscillator_types'] == 0, data['oscillator_types'] == 1))[0]
    proline_indices = np.where(data['oscillator_types'] == 1)[0]
    
    for i in backbone_indices[:-1]:
        j = backbone_indices[i+1]
        mapname = "Coupling"
        
        osc = data['frame_oscillators'][i]
        # If proline is present, NNC maps needs stereochemistry
        if i in proline_indices or j in proline_indices:
            atoms = osc['predicted_atoms'] if use_predicted else osc['atoms']
            omega = calc_dihedral(atoms['CA_prev'], atoms['C_prev'], atoms['N_curr'], atoms['CA_curr'])
            
            if i in proline_indices:
                if np.abs(omega) < 90.0:
                    mapname += '_cisPro_transGly'
                else:
                    mapname += '_transPro_transGly'
            else:
                if np.abs(omega) < 90.0:
                    mapname += '_cisGly_transPro'
                else:
                    mapname += '_transGly_transPro'
        
        rama_angles = osc['predicted_rama_nnfs'] if use_predicted else osc['rama_nnfs']
        nnc_val = calculate_NNC(rama_angles['phi_C'], rama_angles['psi_C'], mapname)
        J[i,j] = nnc_val
        J[j,i] = nnc_val
        
    return J



def calculate_TDC(dipoles: np.ndarray, C_positions: np.ndarray, N_positions: np.ndarray, O_positions: np.ndarray) -> np.ndarray:
    """
    Calculate transition dipole coupling (TDC) matrix.

    Formula:
        J_ij = 5034 * (
            (m_i · m_j) / |r_ij|^3 -
            3 * (m_i · r_ij) * (m_j · r_ij) / |r_ij|^5
        )

    Args:
        dipoles: Dipole vectors [N, 3] in Debye
        C_positions: Carbon positions [N, 3] in Å

    Returns:
        J: Coupling matrix [N, N] in cm^-1
    """
    N = len(dipoles)

    CO = O_positions - C_positions
    CN = N_positions - C_positions
    
    CO /= np.linalg.norm(CO, axis=1, keepdims=True)
    CN /= np.linalg.norm(CN, axis=1, keepdims=True)
    
    s = 0.665*CO + 0.258*CN
    r = s + C_positions
    
    # Pairwise distance vectors: r_ij = r_j - r_i
    r_ij = r[:, np.newaxis, :] - r[np.newaxis, :, :]  # [N, N, 3]

    # Distances
    r_mag = np.linalg.norm(r_ij, axis=2)  # [N, N]

    # m_i · m_j
    mu_dot = np.sum(dipoles[:, np.newaxis, :] * dipoles[np.newaxis, :, :], axis=2)  # [N, N]

    # m_i · r_ij
    mu_i_dot_r = np.sum(dipoles[:, np.newaxis, :] * r_ij, axis=2)  # [N, N]

    # m_j · r_ij
    mu_j_dot_r = np.sum(dipoles[np.newaxis, :, :] * r_ij, axis=2)  # [N, N]

    # Coupling formula
    r3 = r_mag**3
    r5 = r_mag**5
    
    # Ignore the division by zero warnings from i=j
    with warnings.catch_warnings(action='ignore'):
        J = 5034.0 * (mu_dot / r3 - 3.0 * mu_i_dot_r * mu_j_dot_r / r5)  # [N, N]

    # Set diagonal to zero
    np.fill_diagonal(J, 0.0)

    return J.astype(np.float32)
    


def calculate_NNC(phi: float, psi: float, mapname: str) -> float:
    dim = 13
    space = 30
    
    # Rounding based on recent bug fix in AIM
    phi = round(phi, 4)
    psi = round(psi, 4)
    
    phi_N = int((phi + 180) // space)
    psi_N = int((psi + 180) // space)
    if phi_N == dim - 1:
        phi_N = dim-2
    if psi_N == dim - 1:
        psi_N = dim-2
    
    map = NNC_map[mapname]
    if phi_N >= 0 and phi_N < dim-1 and psi_N >= 0 and psi_N < dim-1:
        # determine lower and higher bound
        x1l = phi_N * space - 180
        x2l = psi_N * space - 180

        y1 = map[psi_N, phi_N]
        y2 = map[psi_N+1, phi_N]
        y3 = map[psi_N+1, phi_N+1]
        y4 = map[psi_N, phi_N+1]

        u = (phi - x1l)/space
        t = (psi - x2l)/space

        # bilinear interpolation
        delta = (1-u)*(1-t)*y1 + (1-u)*t*y2 + u*t*y3 + u*(1-t)*y4
    else:
        delta = 0.0
    
    return delta
    

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

    # Diagonalize
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
        mask_batch: [B, N_i] oscillator mask (1=valid, 0=padded) for excluding padded oscillators

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
        current_mask = mask_batch[i] if mask_batch is not None else None
        _, spectrum = generate_spectrum_torch(
            H_diag_batch[i],
            J_matrix_batch[i],
            dipoles_batch[i],
            mask=current_mask,
            omega_min=omega_min,
            omega_max=omega_max,
            omega_step=omega_step,
            gamma=gamma
        )
        spectra.append(spectrum)

    return torch.stack(spectra, dim=0)  # [B, M]

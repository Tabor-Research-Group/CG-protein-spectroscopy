"""
PyTorch dataset for site energy prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
from .data_utils import organize_by_frames, extract_ground_truth_data, extract_predicted_data
from .features import extract_features_for_frame
from .physics import (
    calculate_torii_dipole_batch_numpy,
    calculate_tasumi_coupling_numpy,
    generate_spectrum_numpy
)


class SpectrumDataset(Dataset):
    """
    Dataset for training site energy prediction.

    Each sample is one frame containing N oscillators.
    """

    def __init__(
        self,
        pkl_data: Dict,
        frame_indices: List[int],
        cutoff: float = 20.0,
        max_neighbors: int = 80,
        omega_min: float = 1500.0,
        omega_max: float = 1750.0,
        omega_step: float = 1.0,
        gamma: float = 10.0,
    ):
        """
        Args:
            pkl_data: Dictionary loaded from PKL file
            frame_indices: List of frame indices to use
            cutoff: Neighbor cutoff distance
            max_neighbors: Max neighbors per oscillator
            omega_min, omega_max, omega_step: Spectrum frequency grid
            gamma: Lorentzian width
        """
        self.frames_dict = organize_by_frames(pkl_data)
        self.frame_indices = frame_indices
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.omega_step = omega_step
        self.gamma = gamma

        # Pre-compute spectrum frequency grid
        self.omega_grid = np.arange(omega_min, omega_max + omega_step, omega_step)

        print(f"Dataset initialized with {len(frame_indices)} frames")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one frame's data.

        Returns:
            Dictionary with:
                - own_features: [N, F_own]
                - neighbor_features: [N, K, F_neighbor]
                - neighbor_mask: [N, K]
                - H_diag_true: [N] ground truth site energies
                - C_positions_pred: [N, 3] predicted carbon positions
                - O_positions_pred: [N, 3] predicted oxygen positions
                - N_positions_pred: [N, 3] predicted nitrogen positions
                - spectrum_true: [M] ground truth spectrum
                - dipoles_true: [N, 3] ground truth dipoles
                - J_matrix_true: [N, N] ground truth coupling matrix
        """
        frame_idx = self.frame_indices[idx]
        frame_oscillators = self.frames_dict[frame_idx]

        # Extract ground truth data (for target spectrum)
        gt_data = extract_ground_truth_data(frame_oscillators)

        # Calculate ground truth dipoles (from atomistic atoms)
        dipoles_true = calculate_torii_dipole_batch_numpy(
            gt_data['C_positions'],
            gt_data['O_positions'],
            gt_data['N_positions']
        )

        # Calculate ground truth coupling matrix
        J_matrix_true = calculate_tasumi_coupling_numpy(dipoles_true, gt_data['C_positions'])

        # Generate ground truth spectrum
        _, spectrum_true = generate_spectrum_numpy(
            gt_data['H_diag'],
            J_matrix_true,
            dipoles_true,
            self.omega_min, self.omega_max, self.omega_step, self.gamma
        )

        # Extract predicted data (for model input)
        pred_data = extract_predicted_data(frame_oscillators)

        # Extract features
        features = extract_features_for_frame(pred_data, self.cutoff, self.max_neighbors)

        # Convert to tensors
        return {
            'own_features': torch.from_numpy(features['own_features']).float(),
            'neighbor_features': torch.from_numpy(features['neighbor_features']).float(),
            'neighbor_mask': torch.from_numpy(features['neighbor_mask']).float(),
            'H_diag_true': torch.from_numpy(gt_data['H_diag']).float(),
            'C_positions_pred': torch.from_numpy(pred_data['C_positions']).float(),
            'O_positions_pred': torch.from_numpy(pred_data['O_positions']).float(),
            'N_positions_pred': torch.from_numpy(pred_data['N_positions']).float(),
            'spectrum_true': torch.from_numpy(spectrum_true).float(),
            'dipoles_true': torch.from_numpy(dipoles_true).float(),
            'J_matrix_true': torch.from_numpy(J_matrix_true).float(),
            'frame_idx': frame_idx,
        }


def collate_fn_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle variable number of oscillators per frame.

    Pads to max oscillators in batch.
    """
    # Find max number of oscillators in batch
    max_N = max(sample['own_features'].shape[0] for sample in batch)
    max_K = batch[0]['neighbor_features'].shape[1]  # Should be same for all
    F_own = batch[0]['own_features'].shape[1]
    F_neighbor = batch[0]['neighbor_features'].shape[2]
    M = batch[0]['spectrum_true'].shape[0]

    B = len(batch)

    # Initialize padded tensors
    # IMPORTANT: Place padded atoms far away (z=1000 Å) so coupling J ∝ 1/r³ → 0
    # BUT: C, O, N must be at DIFFERENT positions to avoid degenerate dipoles
    own_features = torch.zeros(B, max_N, F_own)
    neighbor_features = torch.zeros(B, max_N, max_K, F_neighbor)
    neighbor_mask = torch.zeros(B, max_N, max_K)
    H_diag_true = torch.zeros(B, max_N)

    # Place C, O, N at different positions to form a valid (but far away) amide group
    # C at (0, 0, 1000), O at (1.2, 0, 1000), N at (0, 1.3, 1000)
    # These form realistic C=O and C-N bond lengths (~1.2-1.3 Å)
    C_positions_pred = torch.zeros(B, max_N, 3)
    C_positions_pred[:, :, 2] = 1000.0  # C at z=1000

    O_positions_pred = torch.zeros(B, max_N, 3)
    O_positions_pred[:, :, 0] = 1.2   # O offset in x
    O_positions_pred[:, :, 2] = 1000.0  # O at z=1000

    N_positions_pred = torch.zeros(B, max_N, 3)
    N_positions_pred[:, :, 1] = 1.3   # N offset in y
    N_positions_pred[:, :, 2] = 1000.0  # N at z=1000

    spectrum_true = torch.zeros(B, M)
    dipoles_true = torch.zeros(B, max_N, 3)
    J_matrix_true = torch.zeros(B, max_N, max_N)

    # Oscillator mask (for loss calculation)
    oscillator_mask = torch.zeros(B, max_N)

    frame_indices = []

    # Fill in data
    for i, sample in enumerate(batch):
        N = sample['own_features'].shape[0]

        own_features[i, :N] = sample['own_features']
        neighbor_features[i, :N] = sample['neighbor_features']
        neighbor_mask[i, :N] = sample['neighbor_mask']
        H_diag_true[i, :N] = sample['H_diag_true']
        C_positions_pred[i, :N] = sample['C_positions_pred']
        O_positions_pred[i, :N] = sample['O_positions_pred']
        N_positions_pred[i, :N] = sample['N_positions_pred']
        spectrum_true[i] = sample['spectrum_true']
        dipoles_true[i, :N] = sample['dipoles_true']
        J_matrix_true[i, :N, :N] = sample['J_matrix_true']

        oscillator_mask[i, :N] = 1.0

        frame_indices.append(sample['frame_idx'])

    return {
        'own_features': own_features,
        'neighbor_features': neighbor_features,
        'neighbor_mask': neighbor_mask,
        'H_diag_true': H_diag_true,
        'C_positions_pred': C_positions_pred,
        'O_positions_pred': O_positions_pred,
        'N_positions_pred': N_positions_pred,
        'spectrum_true': spectrum_true,
        'dipoles_true': dipoles_true,
        'J_matrix_true': J_matrix_true,
        'oscillator_mask': oscillator_mask,
        'frame_indices': frame_indices,
    }


def create_dataloaders(
    train_data: Dict,
    test_data: Dict,
    train_frame_indices: List[int],
    test_frame_indices: List[int],
    batch_size: int = 8,
    num_workers: int = 4,
    cutoff: float = 20.0,
    max_neighbors: int = 80,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        train_data: Train PKL data
        test_data: Test PKL data
        train_frame_indices: Frame indices for training
        test_frame_indices: Frame indices for testing
        batch_size: Batch size
        num_workers: Number of workers for data loading
        Other args: Dataset parameters

    Returns:
        train_loader, test_loader
    """
    train_dataset = SpectrumDataset(
        train_data, train_frame_indices, cutoff, max_neighbors,
        omega_min, omega_max, omega_step, gamma
    )

    test_dataset = SpectrumDataset(
        test_data, test_frame_indices, cutoff, max_neighbors,
        omega_min, omega_max, omega_step, gamma
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True,
    )

    return train_loader, test_loader

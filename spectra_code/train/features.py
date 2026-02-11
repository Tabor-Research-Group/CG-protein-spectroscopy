"""
Feature extraction for oscillators.

Implements:
1. Local frame construction (CON plane, CO as z-axis)
2. Neighbor finding with cutoff
3. Spherical coordinate transformation
4. Feature padding and masking
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
from scipy.spatial.transform import Rotation
from .data_utils import get_secondary_structure_from_rama


def build_local_frame(C: np.ndarray, O: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Build local coordinate frame for an oscillator.

    Frame definition:
        - Origin: C position
        - z-axis: CO vector (C → O direction)
        - x-axis: Perpendicular to CON plane (CO × CN direction)
        - y-axis: Complete right-handed system (z × x)

    Args:
        C: Carbon position [3]
        O: Oxygen position [3]
        N: Nitrogen position [3]

    Returns:
        rotation_matrix: [3, 3] transformation from global to local frame
    """
    # CO vector (z-axis)
    CO = O - C
    CO_norm = np.linalg.norm(CO)
    if CO_norm < 1e-6:
        # Degenerate case: return identity
        return np.eye(3, dtype=np.float32)
    z_axis = CO / CO_norm

    # CN vector
    CN = N - C
    CN_norm = np.linalg.norm(CN)
    if CN_norm < 1e-6:
        # Degenerate case: use arbitrary perpendicular
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_axis, x_axis)) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
    else:
        CN_unit = CN / CN_norm
        # x-axis: perpendicular to CON plane
        x_axis = np.cross(z_axis, CN_unit)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            # CO and CN are parallel: use arbitrary perpendicular
            x_axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0.0, 1.0, 0.0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
        else:
            x_axis = x_axis / x_norm

    # y-axis: complete right-handed system
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix (rows are local axes in global coordinates)
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)

    return rotation_matrix


def global_to_local(points_global: np.ndarray, C: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points from global to local frame.

    Args:
        points_global: [N, 3] points in global coordinates
        C: [3] origin of local frame
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        points_local: [N, 3] points in local coordinates
    """
    # Translate to origin
    points_centered = points_global - C[np.newaxis, :]

    # Rotate
    points_local = np.dot(points_centered, rotation_matrix.T)

    return points_local


def cartesian_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical.

    Args:
        points: [N, 3] in Cartesian (x, y, z)

    Returns:
        r: [N] radial distance
        theta: [N] polar angle (angle from z-axis), range [0, π]
        phi: [N] azimuthal angle (angle in xy-plane), range [-π, π]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))
    phi = np.arctan2(y, x)

    return r, theta, phi


def find_neighbors(
    C_positions: np.ndarray,
    query_idx: int,
    cutoff: float = 20.0
) -> np.ndarray:
    """
    Find neighbors within cutoff distance.

    Args:
        C_positions: [N, 3] carbon positions for all oscillators
        query_idx: Index of query oscillator
        cutoff: Distance cutoff in Angstroms

    Returns:
        neighbor_indices: [K] indices of neighbors (excluding self)
    """
    query_pos = C_positions[query_idx]
    distances = np.linalg.norm(C_positions - query_pos[np.newaxis, :], axis=1)

    # Find neighbors (excluding self)
    neighbor_mask = (distances < cutoff) & (np.arange(len(C_positions)) != query_idx)
    neighbor_indices = np.where(neighbor_mask)[0]

    return neighbor_indices


def extract_features_for_oscillator(
    osc_idx: int,
    frame_data: Dict,
    cutoff: float = 20.0,
    max_neighbors: int = 80
) -> Dict:
    """
    Extract all features for one oscillator.

    Args:
        osc_idx: Index of query oscillator
        frame_data: Dictionary with C_positions, charges, rama_angles, etc.
        cutoff: Neighbor cutoff distance
        max_neighbors: Maximum number of neighbors (for padding)

    Returns:
        Dictionary with:
            - own_features: [F_own] own oscillator features
            - neighbor_features: [max_neighbors, F_neighbor] neighbor features
            - neighbor_mask: [max_neighbors] mask for valid neighbors (1=valid, 0=padded)
            - num_real_neighbors: scalar
    """
    C_positions = frame_data['C_positions']
    O_positions = frame_data['O_positions']
    N_positions = frame_data['N_positions']
    oscillator_types = frame_data['oscillator_types']
    charges = frame_data['charges']
    rama_angles = frame_data['rama_angles']  # [N, 4]

    N_osc = len(C_positions)

    # Own features
    C_own = C_positions[osc_idx]
    O_own = O_positions[osc_idx]
    N_own = N_positions[osc_idx]

    osc_type_own = oscillator_types[osc_idx]
    charge_own = charges[osc_idx]
    rama_own = rama_angles[osc_idx]  # [4]

    # One-hot encode oscillator type (3 types)
    osc_type_onehot = np.zeros(3, dtype=np.float32)
    osc_type_onehot[osc_type_own] = 1.0

    # Rama angles as sin/cos (8 features)
    rama_sin_cos = np.zeros(8, dtype=np.float32)
    for i in range(4):
        angle_rad = np.deg2rad(rama_own[i])
        rama_sin_cos[2*i] = np.sin(angle_rad)
        rama_sin_cos[2*i + 1] = np.cos(angle_rad)

    # Secondary structure (one-hot, 4 types)
    ss_own = get_secondary_structure_from_rama(rama_own[0], rama_own[3])
    ss_onehot = np.zeros(4, dtype=np.float32)
    ss_onehot[ss_own] = 1.0

    # Combine own features
    own_features = np.concatenate([
        osc_type_onehot,  # 3
        rama_sin_cos,     # 8
        ss_onehot,        # 4
        [charge_own]      # 1
    ])  # Total: 16

    # Build local frame
    rotation_matrix = build_local_frame(C_own, O_own, N_own)

    # Find neighbors
    neighbor_indices = find_neighbors(C_positions, osc_idx, cutoff)
    num_real_neighbors = len(neighbor_indices)

    # Limit to max_neighbors
    if num_real_neighbors > max_neighbors:
        neighbor_indices = neighbor_indices[:max_neighbors]
        num_real_neighbors = max_neighbors

    # Initialize neighbor features (padded)
    neighbor_features = np.zeros((max_neighbors, 18), dtype=np.float32)
    neighbor_mask = np.zeros(max_neighbors, dtype=np.float32)

    # Fill real neighbor features
    if num_real_neighbors > 0:
        # Transform neighbor positions to local frame
        neighbor_C_global = C_positions[neighbor_indices]  # [K, 3]
        neighbor_C_local = global_to_local(neighbor_C_global, C_own, rotation_matrix)  # [K, 3]

        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(neighbor_C_local)

        # Distance feature: 1/r^3 (scaled)
        r_safe = np.maximum(r, 2.0)  # Minimum 2 Angstrom
        inv_r3 = 1.0 / (r_safe ** 3)
        inv_r3_scaled = np.clip((inv_r3 - 1.0/(20.0**3)) / (1.0/(2.0**3) - 1.0/(20.0**3)), 0.0, 1.0)

        # Angular features: sin/cos
        theta_sin = np.sin(theta)
        theta_cos = np.cos(theta)
        phi_sin = np.sin(phi)
        phi_cos = np.cos(phi)

        # Neighbor charges
        neighbor_charges = charges[neighbor_indices]

        # Neighbor rama angles (sin/cos)
        neighbor_rama = rama_angles[neighbor_indices]  # [K, 4]
        neighbor_rama_sin_cos = np.zeros((num_real_neighbors, 8), dtype=np.float32)
        for i in range(4):
            angle_rad = np.deg2rad(neighbor_rama[:, i])
            neighbor_rama_sin_cos[:, 2*i] = np.sin(angle_rad)
            neighbor_rama_sin_cos[:, 2*i + 1] = np.cos(angle_rad)

        # Neighbor secondary structure (one-hot)
        neighbor_ss = np.array([
            get_secondary_structure_from_rama(neighbor_rama[i, 0], neighbor_rama[i, 3])
            for i in range(num_real_neighbors)
        ])
        neighbor_ss_onehot = np.zeros((num_real_neighbors, 4), dtype=np.float32)
        neighbor_ss_onehot[np.arange(num_real_neighbors), neighbor_ss] = 1.0

        # Combine neighbor features
        neighbor_features[:num_real_neighbors] = np.column_stack([
            inv_r3_scaled,            # 1
            theta_sin, theta_cos,     # 2
            phi_sin, phi_cos,         # 2
            neighbor_charges,         # 1
            neighbor_rama_sin_cos,    # 8
            neighbor_ss_onehot,       # 4
        ])  # Total: 18

        neighbor_mask[:num_real_neighbors] = 1.0

    return {
        'own_features': own_features,              # [16]
        'neighbor_features': neighbor_features,    # [max_neighbors, 18]
        'neighbor_mask': neighbor_mask,            # [max_neighbors]
        'num_real_neighbors': num_real_neighbors,
        'rotation_matrix': rotation_matrix,        # [3, 3] for debugging
    }


def extract_features_for_frame(
    frame_data: Dict,
    cutoff: float = 20.0,
    max_neighbors: int = 80
) -> Dict:
    """
    Extract features for all oscillators in a frame.

    Args:
        frame_data: Dictionary with positions, charges, rama angles, etc.
        cutoff: Neighbor cutoff
        max_neighbors: Max neighbors per oscillator

    Returns:
        Dictionary with batched features:
            - own_features: [N, F_own]
            - neighbor_features: [N, max_neighbors, F_neighbor]
            - neighbor_mask: [N, max_neighbors]
    """
    N = len(frame_data['C_positions'])

    own_features_list = []
    neighbor_features_list = []
    neighbor_mask_list = []

    for i in range(N):
        feats = extract_features_for_oscillator(i, frame_data, cutoff, max_neighbors)
        own_features_list.append(feats['own_features'])
        neighbor_features_list.append(feats['neighbor_features'])
        neighbor_mask_list.append(feats['neighbor_mask'])

    return {
        'own_features': np.stack(own_features_list, axis=0),           # [N, 16]
        'neighbor_features': np.stack(neighbor_features_list, axis=0), # [N, max_neighbors, 18]
        'neighbor_mask': np.stack(neighbor_mask_list, axis=0),         # [N, max_neighbors]
    }


def compute_radial_distribution(
    C_positions: np.ndarray,
    query_idx: int,
    cutoff: float = 20.0,
    num_bins: int = 20
) -> np.ndarray:
    """
    Compute radial distribution g(r) around an oscillator.

    Args:
        C_positions: [N, 3] carbon positions
        query_idx: Query oscillator index
        cutoff: Maximum distance
        num_bins: Number of radial bins

    Returns:
        g_r: [num_bins] counts in each bin
    """
    query_pos = C_positions[query_idx]
    distances = np.linalg.norm(C_positions - query_pos[np.newaxis, :], axis=1)

    # Exclude self
    distances = distances[np.arange(len(C_positions)) != query_idx]

    # Histogram
    bins = np.linspace(0, cutoff, num_bins + 1)
    g_r, _ = np.histogram(distances, bins=bins)

    return g_r.astype(np.float32)

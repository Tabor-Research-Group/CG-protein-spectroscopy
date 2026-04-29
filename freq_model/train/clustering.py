"""
Spectrum clustering and sampling for diverse training data.

Uses K-means clustering on spectral features to sample diverse frames.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from .physics import calculate_torii_dipole_batch_numpy, calculate_tasumi_coupling_numpy, generate_spectrum_numpy
from .data_utils import extract_ground_truth_data


def extract_spectral_features(spectrum: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """
    Extract features from a spectrum for clustering.

    Features:
        - Peak position (center of mass)
        - Peak width (second moment)
        - Peak height (max value)
        - Spectrum moments (mean, std)
        - Spectrum at key frequencies

    Args:
        spectrum: [M] intensity values
        omega_grid: [M] frequency values

    Returns:
        features: [F] feature vector
    """
    # Normalize
    spectrum_norm = spectrum / (np.max(spectrum) + 1e-6)

    # Peak position (center of mass)
    total_intensity = np.sum(spectrum_norm)
    if total_intensity > 0:
        peak_position = np.sum(omega_grid * spectrum_norm) / total_intensity
    else:
        peak_position = np.mean(omega_grid)

    # Peak width (second moment)
    peak_width = np.sqrt(np.sum((omega_grid - peak_position)**2 * spectrum_norm) / (total_intensity + 1e-6))

    # Peak height
    peak_height = np.max(spectrum_norm)

    # Moments
    mean_intensity = np.mean(spectrum_norm)
    std_intensity = np.std(spectrum_norm)

    # Sample at key frequencies (every 10 cm^-1)
    key_frequencies = np.arange(1500, 1750, 10)
    key_intensities = []
    for freq in key_frequencies:
        idx = np.argmin(np.abs(omega_grid - freq))
        key_intensities.append(spectrum_norm[idx])

    # Combine features
    features = np.array([
        peak_position,
        peak_width,
        peak_height,
        mean_intensity,
        std_intensity,
    ] + key_intensities)

    return features


def cluster_spectra(
    spectra: np.ndarray,
    omega_grid: np.ndarray,
    n_clusters: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster spectra using K-means.

    Args:
        spectra: [N, M] array of spectra
        omega_grid: [M] frequency grid
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        labels: [N] cluster labels
        kmeans: Fitted KMeans object
    """
    N = len(spectra)

    print(f"Extracting spectral features from {N} spectra...")

    # Extract features
    features_list = []
    for i in range(N):
        features = extract_spectral_features(spectra[i], omega_grid)
        features_list.append(features)

    features_array = np.stack(features_list, axis=0)  # [N, F]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # Cluster
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # Print cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: min={np.min(counts)}, max={np.max(counts)}, mean={np.mean(counts):.1f}")

    return labels, kmeans


def sample_from_clusters(
    labels: np.ndarray,
    samples_per_cluster: int = 1,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample frames from each cluster.

    Args:
        labels: [N] cluster labels
        samples_per_cluster: Number of samples per cluster
        random_state: Random seed

    Returns:
        sampled_indices: [K] indices of sampled frames
    """
    np.random.seed(random_state)

    unique_labels = np.unique(labels)
    sampled_indices = []

    for label in unique_labels:
        # Get all frames in this cluster
        cluster_indices = np.where(labels == label)[0]

        # Sample
        if len(cluster_indices) <= samples_per_cluster:
            # Take all if not enough samples
            sampled_indices.extend(cluster_indices)
        else:
            # Random sample
            sampled = np.random.choice(cluster_indices, size=samples_per_cluster, replace=False)
            sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices)
    print(f"Sampled {len(sampled_indices)} frames from {len(unique_labels)} clusters")

    return sampled_indices


def generate_and_cluster_spectra(
    frames_dict: Dict[int, List[Dict]],
    n_clusters: int = 50,
    samples_per_cluster: int = 1,
    omega_min: float = 1500.0,
    omega_max: float = 1750.0,
    omega_step: float = 1.0,
    gamma: float = 10.0,
    random_state: int = 42
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Generate ground truth spectra for all frames, cluster, and sample.

    Args:
        frames_dict: Dictionary {frame_idx: [oscillators]}
        n_clusters: Number of clusters
        samples_per_cluster: Samples per cluster
        omega_min, omega_max, omega_step: Frequency grid
        gamma: Lorentzian width
        random_state: Random seed

    Returns:
        sampled_frame_indices: List of selected frame indices
        all_spectra: [N_frames, M] all spectra (for visualization)
        omega_grid: [M] frequency grid
    """


    frame_indices = sorted(frames_dict.keys())
    print(f"\nGenerating ground truth spectra for {len(frame_indices)} frames...")

    spectra_list = []
    omega_grid = None

    for frame_idx in frame_indices:
        # Get ground truth data
        gt_data = extract_ground_truth_data(frames_dict[frame_idx])

        # Calculate ground truth dipoles (from atomistic atoms)
        dipoles = calculate_torii_dipole_batch_numpy(
            gt_data['C_positions'],
            gt_data['O_positions'],
            gt_data['N_positions']
        )

        # Calculate coupling matrix
        J_matrix = calculate_tasumi_coupling_numpy(dipoles, gt_data['C_positions'])

        # Generate spectrum
        omega_grid, spectrum = generate_spectrum_numpy(
            gt_data['H_diag'],
            J_matrix,
            dipoles,
            omega_min, omega_max, omega_step, gamma
        )

        spectra_list.append(spectrum)

    all_spectra = np.stack(spectra_list, axis=0)  # [N, M]

    # Cluster
    labels, kmeans = cluster_spectra(all_spectra, omega_grid, n_clusters, random_state)

    # Sample
    sampled_indices = sample_from_clusters(labels, samples_per_cluster, random_state)

    # Map back to frame indices
    sampled_frame_indices = [frame_indices[i] for i in sampled_indices]

    print(f"Selected {len(sampled_frame_indices)} diverse frames for training")

    # Return labels to avoid re-clustering for visualization
    return sampled_frame_indices, all_spectra, omega_grid, labels, sampled_indices


def plot_cluster_summary(
    all_spectra: np.ndarray,
    omega_grid: np.ndarray,
    labels: np.ndarray,
    sampled_indices: np.ndarray,
    save_path: str = "cluster_summary.png"
):
    """
    Plot cluster summary showing representative spectra.

    Args:
        all_spectra: [N, M] all spectra
        omega_grid: [M] frequency grid
        labels: [N] cluster labels
        sampled_indices: [K] sampled frame indices
        save_path: Output path
    """

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Plot grid
    n_cols = min(5, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))

    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, label in enumerate(unique_labels):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get all spectra in this cluster
        cluster_mask = (labels == label)
        cluster_spectra = all_spectra[cluster_mask]

        # Plot all spectra in cluster (faint)
        for spectrum in cluster_spectra:
            ax.plot(omega_grid, spectrum, color='gray', alpha=0.2, linewidth=0.5)

        # Plot mean spectrum (bold)
        mean_spectrum = np.mean(cluster_spectra, axis=0)
        ax.plot(omega_grid, mean_spectrum, color='blue', linewidth=2, label='Mean')

        # Highlight sampled spectrum if any
        cluster_indices = np.where(cluster_mask)[0]
        sampled_in_cluster = [i for i in sampled_indices if i in cluster_indices]
        if sampled_in_cluster:
            for sample_idx in sampled_in_cluster:
                ax.plot(omega_grid, all_spectra[sample_idx], color='red', linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_title(f"Cluster {label} (n={np.sum(cluster_mask)})", fontsize=10)
        ax.set_xlabel("Frequency (cm⁻¹)", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.set_xlim(omega_grid[0], omega_grid[-1])
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)

    # Remove empty subplots
    for idx in range(n_clusters, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster summary to {save_path}")

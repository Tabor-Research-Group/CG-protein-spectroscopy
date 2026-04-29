"""
Memory-efficient data loading for large datasets.

Instead of loading all PKL files at once, this version:
1. Loads files one-by-one
2. Samples frames from each file
3. Only keeps the sampled frames in memory
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


def load_pkl_sampled(
    directory: str,
    frames_per_file: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Dict, Dict[str, str]]:
    """
    Load PKL files with MEMORY-EFFICIENT sampling.

    Instead of loading all files, randomly samples frames from each file.

    Args:
        directory: Path to directory containing PKL files
        frames_per_file: Number of frames to sample from each file
        seed: Random seed for reproducibility
        verbose: Print loading progress

    Returns:
        merged_data: Combined data from sampled frames
        file_mapping: Mapping of {protein_id: file_path}
    """
    np.random.seed(seed)

    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")

    pkl_files = sorted(dir_path.glob('*.pkl'))

    if len(pkl_files) == 0:
        raise ValueError(f"No PKL files found in {directory}")

    if verbose:
        print(f"\nMemory-efficient loading: {len(pkl_files)} PKL files from {directory}")
        print(f"Sampling {frames_per_file} frames per file")

    merged_data = defaultdict(list)
    file_mapping = {}
    frame_offset = 0

    total_sampled = 0

    for file_idx, pkl_file in enumerate(pkl_files):
        if verbose:
            print(f"  [{file_idx+1}/{len(pkl_files)}] Loading {pkl_file.name}...", end=' ')

        protein_id = pkl_file.stem
        file_mapping[protein_id] = str(pkl_file)

        # Load file
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Find all frame indices in this file
        all_frames = set()
        for aa_type, oscillators in data.items():
            for osc in oscillators:
                all_frames.add(osc['frame'])

        all_frames = sorted(all_frames)
        n_frames_in_file = len(all_frames)

        # Sample frames
        if n_frames_in_file <= frames_per_file:
            # Use all frames if file is small
            sampled_frames = set(all_frames)
        else:
            # Randomly sample
            sampled_indices = np.random.choice(
                len(all_frames),
                size=frames_per_file,
                replace=False
            )
            sampled_frames = set(all_frames[i] for i in sampled_indices)

        # Keep only oscillators from sampled frames
        n_kept = 0
        for aa_type, oscillators in data.items():
            for osc in oscillators:
                if osc['frame'] in sampled_frames:
                    # Add protein_id and adjust frame index
                    osc['protein_id'] = protein_id
                    osc['original_frame'] = osc['frame']
                    osc['frame'] = osc['frame'] + frame_offset
                    merged_data[aa_type].append(osc)
                    n_kept += 1

        # Update offset
        frame_offset += n_frames_in_file
        total_sampled += len(sampled_frames)

        if verbose:
            print(f"sampled {len(sampled_frames)}/{n_frames_in_file} frames, {n_kept:,} oscillators")

    # Convert to regular dict
    merged_data = dict(merged_data)

    if verbose:
        total_oscillators = sum(len(oscs) for oscs in merged_data.values())
        print(f"\n  TOTAL: {total_sampled} frames, {total_oscillators:,} oscillators")
        print(f"  Memory saved: ~{(len(pkl_files) * 1000 - total_sampled) / 1000:.1f}k frames not loaded")

    return merged_data, file_mapping


def estimate_memory_usage(directory: str) -> dict:
    """
    Estimate memory requirements for loading data.

    Returns:
        dict with memory estimates in GB
    """
    dir_path = Path(directory)
    pkl_files = list(dir_path.glob('*.pkl'))

    if len(pkl_files) == 0:
        return {'error': 'No files found'}

    # Check first file to estimate
    sample_file = pkl_files[0]
    file_size_mb = sample_file.stat().st_size / (1024**2)

    # Rule of thumb: in-memory size is ~5x file size for PKL files
    memory_per_file_gb = (file_size_mb * 5) / 1024

    total_files = len(pkl_files)
    total_memory_gb = memory_per_file_gb * total_files

    return {
        'num_files': total_files,
        'avg_file_size_mb': file_size_mb,
        'estimated_memory_per_file_gb': memory_per_file_gb,
        'estimated_total_memory_gb': total_memory_gb,
        'recommendation': (
            'Use load_pkl_sampled() with frames_per_file=50-100'
            if total_memory_gb > 10
            else 'Safe to use regular load_pkl_from_directory()'
        )
    }


if __name__ == '__main__':
    # Test memory estimation
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_utils_lazy.py <directory>")
        print("Example: python data_utils_lazy.py denoise_results/train_denoise_results")
        sys.exit(1)

    directory = sys.argv[1]

    print("="*80)
    print("MEMORY USAGE ESTIMATION")
    print("="*80)

    estimates = estimate_memory_usage(directory)

    for key, value in estimates.items():
        print(f"{key}: {value}")

    print("\n" + "="*80)
    print("TESTING SAMPLED LOADING")
    print("="*80)

    # Test with 10 frames per file
    data, mapping = load_pkl_sampled(directory, frames_per_file=10)

    print(f"\nLoaded {len(mapping)} proteins")
    print(f"Total oscillators: {sum(len(v) for v in data.values()):,}")

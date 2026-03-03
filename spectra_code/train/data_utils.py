"""
Data loading and preprocessing utilities.

Handles:
1. Loading PKL files
2. Extracting oscillator data
3. Organizing by frames
4. Computing CO/CN vectors for dipole calculation
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


def load_pkl_data(pkl_path: str) -> Dict:
    """Load pickle file containing oscillator data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_pkl_from_directory(directory: str, verbose: bool = True) -> Tuple[Dict, Dict[str, str]]:
    """
    Load all PKL files from a directory and merge them.

    Args:
        directory: Path to directory containing PKL files
        verbose: Print loading progress

    Returns:
        merged_data: Combined data from all files {amino_acid_type: [oscillators]}
        file_mapping: Mapping of {protein_id: file_path} for per-protein evaluation
    """

    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")

    pkl_files = sorted(dir_path.glob('*.pkl'))

    if len(pkl_files) == 0:
        raise ValueError(f"No PKL files found in {directory}")

    if verbose:
        print(f"\nLoading {len(pkl_files)} PKL files from {directory}")

    merged_data = defaultdict(list)
    file_mapping = {}
    frame_offset = 0  # Global frame offset to avoid collisions

    for file_idx, pkl_file in enumerate(pkl_files):
        if verbose:
            print(f"  [{file_idx+1}/{len(pkl_files)}] Loading {pkl_file.name}...")

        # Extract protein ID from filename (e.g., "1knt_A.pkl" -> "1knt_A")
        protein_id = pkl_file.stem
        file_mapping[protein_id] = str(pkl_file)

        # Load data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Find max frame index in this file
        max_frame_in_file = 0
        for aa_type, oscillators in data.items():
            for osc in oscillators:
                max_frame_in_file = max(max_frame_in_file, osc['frame'])

        n_frames_in_file = max_frame_in_file + 1

        # Merge with unique frame indices
        for aa_type, oscillators in data.items():
            for osc in oscillators:
                # Add protein_id and adjust frame index to be globally unique
                osc['protein_id'] = protein_id
                osc['original_frame'] = osc['frame']  # Keep original for reference
                osc['frame'] = osc['frame'] + frame_offset  # Make globally unique
            merged_data[aa_type].extend(oscillators)

        if verbose:
            n_osc = sum(len(oscs) for oscs in data.values())
            print(f"      {n_frames_in_file} frames, {n_osc:,} oscillators (frame offset: {frame_offset})")

        # Update offset for next file
        frame_offset += n_frames_in_file

    # Convert to regular dict
    merged_data = dict(merged_data)

    if verbose:
        total_oscillators = sum(len(oscs) for oscs in merged_data.values())
        total_frames = frame_offset
        print(f"\n  TOTAL: {total_frames} frames, {total_oscillators:,} oscillators")
        print(f"  Files loaded: {len(pkl_files)}")
        print(f"  Amino acid types: {len(merged_data)}")

    return merged_data, file_mapping


def load_individual_files_from_directory(directory: str, verbose: bool = True) -> Dict[str, Dict]:
    """
    Load individual PKL files from a directory without merging.

    Args:
        directory: Path to directory containing PKL files
        verbose: Print loading progress

    Returns:
        files_data: Dictionary of {protein_id: data_dict}
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")

    pkl_files = sorted(dir_path.glob('*.pkl'))

    if len(pkl_files) == 0:
        raise ValueError(f"No PKL files found in {directory}")

    if verbose:
        print(f"\nLoading {len(pkl_files)} individual PKL files from {directory}")

    files_data = {}

    for pkl_file in pkl_files:
        protein_id = pkl_file.stem

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Add protein_id to each oscillator
        for aa_type, oscillators in data.items():
            for osc in oscillators:
                osc['protein_id'] = protein_id

        files_data[protein_id] = data

        if verbose:
            n_osc = sum(len(oscs) for oscs in data.values())
            print(f"  {protein_id}: {n_osc:,} oscillators")

    return files_data


def extract_atoms_for_dipole(osc: Dict, use_predicted: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract C, O, N atom positions for dipole calculation.

    Args:
        osc: Oscillator dictionary
        use_predicted: If True, use predicted_atoms; else use atoms (ground truth)

    Returns:
        C, O, N: Atom positions [3] each
    """
    atoms = osc['predicted_atoms'] if use_predicted else osc['atoms']
    osc_type = osc.get('oscillator_type', osc.get('type'))
    residue_name = osc.get('residue_name', osc['residue_key'][1])

    if osc_type == 'backbone':
        C = atoms['C_prev']
        O = atoms['O_prev']
        N = atoms['N_curr']
    elif osc_type == 'sidechain':
        if 'GLN' in residue_name:
            C = atoms['CD']
            O = atoms['OE1']
            N = atoms['NE2']
        elif 'ASN' in residue_name:
            C = atoms['CG']
            O = atoms['OD1']
            N = atoms['ND2']
        else:
            raise ValueError(f"Unknown sidechain type: {residue_name}")
    else:
        raise ValueError(f"Unknown oscillator type: {osc_type}")

    return np.array(C), np.array(O), np.array(N)


def get_oscillator_type_encoding(osc: Dict) -> int:
    """
    Get oscillator type encoding.

    Returns:
        0: Regular backbone
        1: PRO backbone
        2: Sidechain (GLN/ASN)
    """
    osc_type = osc.get('oscillator_type', osc.get('type'))
    residue_name = osc.get('residue_name', osc['residue_key'][1])

    if osc_type == 'backbone':
        if 'PRO' in residue_name:
            return 1  # PRO backbone
        else:
            return 0  # Regular backbone
    elif osc_type == 'sidechain':
        return 2  # Sidechain
    else:
        return 0  # Default


def get_oscillator_charge(osc: Dict) -> float:
    """
    Get residue charge for oscillator.

    Returns approximate charge based on residue type.
    """
    residue_name = osc.get('residue_name', osc['residue_key'][1])

    # Remove -SC suffix for sidechains
    if '-SC' in residue_name:
        residue_name = residue_name.replace('-SC', '')

    # Charge assignments (simplified)
    charges = {
        'ARG': 1.0, 'LYS': 1.0,  # Positive
        'ASP': -1.0, 'GLU': -1.0,  # Negative
        'HIS': 0.5,  # Partially positive at neutral pH
    }

    return charges.get(residue_name, 0.0)


def organize_by_frames(pkl_data: Dict) -> Dict[int, List[Dict]]:
    """
    Organize oscillators by frame number.

    Args:
        pkl_data: Dictionary with amino acid keys

    Returns:
        frames_dict: {frame_idx: [list of oscillators]}
    """
    frames_dict = {}

    for aa_type, oscillators in pkl_data.items():
        for osc in oscillators:
            frame_idx = osc['frame']

            if frame_idx not in frames_dict:
                frames_dict[frame_idx] = []

            frames_dict[frame_idx].append(osc)

    # Sort oscillators within each frame by oscillator_index
    for frame_idx in frames_dict:
        frames_dict[frame_idx] = sorted(frames_dict[frame_idx], key=lambda x: x['oscillator_index'])

    return frames_dict


def extract_ground_truth_data(frame_oscillators: List[Dict]) -> Dict:
    """
    Extract ground truth data for a frame.

    Args:
        frame_oscillators: List of oscillators in one frame

    Returns:
        Dictionary with:
            - H_diag: Site energies [N]
            - C_positions: Carbon positions [N, 3]
            - O_positions: Oxygen positions [N, 3]
            - N_positions: Nitrogen positions [N, 3]
            - dipoles: Dipole vectors [N, 3] (from ground truth atoms)
            - oscillator_types: Type encoding [N]
            - charges: Residue charges [N]
    """
    N = len(frame_oscillators)

    H_diag = np.zeros(N, dtype=np.float32)
    C_positions = np.zeros((N, 3), dtype=np.float32)
    O_positions = np.zeros((N, 3), dtype=np.float32)
    N_positions = np.zeros((N, 3), dtype=np.float32)
    oscillator_types = np.zeros(N, dtype=np.int64)
    charges = np.zeros(N, dtype=np.float32)

    for i, osc in enumerate(frame_oscillators):
        # Hamiltonian (site energy)
        H_diag[i] = osc['hamiltonian']

        # Atom positions (ground truth)
        C, O, N_atom = extract_atoms_for_dipole(osc, use_predicted=False)
        C_positions[i] = C
        O_positions[i] = O
        N_positions[i] = N_atom

        # Type and charge
        oscillator_types[i] = get_oscillator_type_encoding(osc)
        charges[i] = get_oscillator_charge(osc)

    return {
        'H_diag': H_diag,
        'C_positions': C_positions,
        'O_positions': O_positions,
        'N_positions': N_positions,
        'oscillator_types': oscillator_types,
        'charges': charges,
        'frame_oscillators': frame_oscillators,
    }


def extract_predicted_data(frame_oscillators: List[Dict]) -> Dict:
    """
    Extract predicted atom data for a frame (for model input).

    Args:
        frame_oscillators: List of oscillators in one frame

    Returns:
        Dictionary with predicted atom positions
    """
    N = len(frame_oscillators)

    C_positions = np.zeros((N, 3), dtype=np.float32)
    O_positions = np.zeros((N, 3), dtype=np.float32)
    N_positions = np.zeros((N, 3), dtype=np.float32)
    oscillator_types = np.zeros(N, dtype=np.int64)
    charges = np.zeros(N, dtype=np.float32)

    # Rama angles (NNFS)
    rama_angles = np.zeros((N, 4), dtype=np.float32)  # phi_N, psi_N, phi_C, psi_C

    for i, osc in enumerate(frame_oscillators):
        # Predicted atom positions
        C, O, N_atom = extract_atoms_for_dipole(osc, use_predicted=True)
        C_positions[i] = C
        O_positions[i] = O
        N_positions[i] = N_atom

        # Type and charge
        oscillator_types[i] = get_oscillator_type_encoding(osc)
        charges[i] = get_oscillator_charge(osc)

        # Rama angles (from predicted_rama_nnfs)
        if 'predicted_rama_nnfs' in osc:
            rama = osc['predicted_rama_nnfs']
            rama_angles[i, 0] = rama.get('phi_N', 0.0)
            rama_angles[i, 1] = rama.get('psi_N', 0.0)
            rama_angles[i, 2] = rama.get('phi_C', 0.0)
            rama_angles[i, 3] = rama.get('psi_C', 0.0)

    rama_angles = np.nan_to_num(rama_angles)
    
    return {
        'C_positions': C_positions,
        'O_positions': O_positions,
        'N_positions': N_positions,
        'oscillator_types': oscillator_types,
        'charges': charges,
        'rama_angles': rama_angles,
        'frame_oscillators': frame_oscillators,
    }


def get_secondary_structure_from_rama(phi: float, psi: float) -> int:
    """
    Simple secondary structure classification from Ramachandran angles.

    Returns:
        0: Coil
        1: Alpha helix
        2: Beta sheet
        3: Turn
    """
    # Convert to radians
    phi_rad = np.deg2rad(phi)
    psi_rad = np.deg2rad(psi)

    # Alpha helix region: phi ≈ -60°, psi ≈ -45°
    if -100 < phi < -30 and -70 < psi < -20:
        return 1

    # Beta sheet region: phi ≈ -120°, psi ≈ +120°
    if -180 < phi < -90 and 90 < psi < 180:
        return 2

    # Extended beta region
    if -180 < phi < -90 and -180 < psi < -90:
        return 2

    # Turn regions
    if -100 < phi < 0 and -50 < psi < 50:
        return 3

    # Default: coil
    return 0


def compute_secondary_structure_batch(rama_angles: np.ndarray) -> np.ndarray:
    """
    Compute secondary structure for all oscillators.

    Args:
        rama_angles: [N, 4] array of (phi_N, psi_N, phi_C, psi_C)

    Returns:
        ss: [N] array of secondary structure codes
    """
    N = len(rama_angles)
    ss = np.zeros(N, dtype=np.int64)

    for i in range(N):
        phi = rama_angles[i, 0]  # Use phi_N
        psi = rama_angles[i, 3]  # Use psi_C
        ss[i] = get_secondary_structure_from_rama(phi, psi)

    return ss


def print_data_summary(pkl_data: Dict, name: str = "Data"):
    """Print summary statistics of loaded data."""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")

    total_oscillators = sum(len(oscs) for oscs in pkl_data.values())
    print(f"Total oscillators: {total_oscillators}")

    print(f"\nOscillators per residue type:")
    for aa_type in sorted(pkl_data.keys()):
        count = len(pkl_data[aa_type])
        print(f"  {aa_type:10s}: {count:6d}")

    # Get frame info
    frames_dict = organize_by_frames(pkl_data)
    num_frames = len(frames_dict)
    oscillators_per_frame = [len(oscs) for oscs in frames_dict.values()]

    print(f"\nFrame statistics:")
    print(f"  Total frames: {num_frames}")
    print(f"  Oscillators per frame: {np.mean(oscillators_per_frame):.1f} ± {np.std(oscillators_per_frame):.1f}")
    print(f"  Min/Max: {np.min(oscillators_per_frame)} / {np.max(oscillators_per_frame)}")

    # Check for ground truth hamiltonian
    sample_osc = list(pkl_data.values())[0][0]
    has_hamiltonian = 'hamiltonian' in sample_osc
    has_atoms = 'atoms' in sample_osc
    has_predicted = 'predicted_atoms' in sample_osc

    print(f"\nAvailable fields:")
    print(f"  Ground truth hamiltonian: {has_hamiltonian}")
    print(f"  Ground truth atoms: {has_atoms}")
    print(f"  Predicted atoms: {has_predicted}")

    if has_hamiltonian:
        all_hamiltonians = [osc['hamiltonian'] for oscs in pkl_data.values() for osc in oscs]
        print(f"\nHamiltonian statistics (cm^-1):")
        print(f"  Mean: {np.mean(all_hamiltonians):.1f}")
        print(f"  Std: {np.std(all_hamiltonians):.1f}")
        print(f"  Min/Max: {np.min(all_hamiltonians):.1f} / {np.max(all_hamiltonians):.1f}")

    print(f"{'='*60}\n")


# ============================================================================
# FRAME QUALITY FILTERING (Added in v9)
# ============================================================================

def check_oscillator_quality(oscillator: Dict, min_bond: float = 0.8, max_bond: float = 2.0) -> tuple[bool, dict]:
    """
    Check if a single oscillator has valid geometry.

    Args:
        oscillator: Oscillator dictionary
        min_bond: Minimum acceptable bond length (Angstroms)
        max_bond: Maximum acceptable bond length (Angstroms)

    Returns:
        is_valid: True if oscillator is valid
        stats: Dictionary with bond length info
    """
    stats = {'CO_dist': None, 'CN_dist': None, 'has_nan': False}

    try:
        # Get predicted positions (what the model will see)
        C, O, N_atom = extract_atoms_for_dipole(oscillator, use_predicted=True)

        # Check for NaN/Inf
        if (np.isnan(C).any() or np.isnan(O).any() or np.isnan(N_atom).any() or
            np.isinf(C).any() or np.isinf(O).any() or np.isinf(N_atom).any()):
            stats['has_nan'] = True
            return False, stats

        # Compute bond lengths
        CO_dist = np.linalg.norm(O - C)
        CN_dist = np.linalg.norm(N_atom - C)

        stats['CO_dist'] = float(CO_dist)
        stats['CN_dist'] = float(CN_dist)

        # Check if bonds are in valid range
        if (CO_dist < min_bond or CO_dist > max_bond or
            CN_dist < min_bond or CN_dist > max_bond):
            return False, stats

        return True, stats

    except Exception as e:
        # If extraction fails, mark as invalid
        return False, stats


def check_frame_quality(frame_oscillators: List[Dict], min_bond: float = 0.8, max_bond: float = 2.0) -> tuple[bool, dict]:
    """
    Check if all oscillators in a frame have valid geometry.

    Args:
        frame_oscillators: List of oscillators in one frame
        min_bond: Minimum acceptable bond length
        max_bond: Maximum acceptable bond length

    Returns:
        is_valid: True if ALL oscillators are valid
        stats: Dictionary with frame statistics
    """
    frame_stats = {
        'n_oscillators': len(frame_oscillators),
        'n_valid': 0,
        'n_invalid': 0,
        'has_nan': 0,
        'bad_bonds': 0,
        'CO_dists': [],
        'CN_dists': []
    }

    all_valid = True

    for osc in frame_oscillators:
        is_valid, osc_stats = check_oscillator_quality(osc, min_bond, max_bond)

        if is_valid:
            frame_stats['n_valid'] += 1
        else:
            frame_stats['n_invalid'] += 1
            all_valid = False

            if osc_stats['has_nan']:
                frame_stats['has_nan'] += 1
            else:
                frame_stats['bad_bonds'] += 1

        if osc_stats['CO_dist'] is not None:
            frame_stats['CO_dists'].append(osc_stats['CO_dist'])
        if osc_stats['CN_dist'] is not None:
            frame_stats['CN_dists'].append(osc_stats['CN_dist'])

    return all_valid, frame_stats


def filter_frames_by_quality(
    frames_dict: Dict[int, List[Dict]],
    min_bond: float = 0.8,
    max_bond: float = 2.0,
    verbose: bool = True
) -> tuple[Dict[int, List[Dict]], int, dict]:
    """
    Filter frames dictionary to keep only frames with valid geometry.

    Args:
        frames_dict: Dictionary of {frame_idx: [oscillators]}
        min_bond: Minimum acceptable bond length (Angstroms)
        max_bond: Maximum acceptable bond length (Angstroms)
        verbose: Print filtering statistics

    Returns:
        filtered_frames_dict: Dictionary with only valid frames
        excluded_count: Number of frames excluded
        stats: Overall statistics
    """
    filtered_dict = {}
    excluded_count = 0
    excluded_frames = []

    overall_stats = {
        'total_frames': len(frames_dict),
        'total_oscillators': 0,
        'excluded_frames': 0,
        'excluded_oscillators': 0,
        'frames_with_nan': 0,
        'frames_with_bad_bonds': 0
    }

    for frame_idx in sorted(frames_dict.keys()):
        frame_oscillators = frames_dict[frame_idx]
        overall_stats['total_oscillators'] += len(frame_oscillators)

        is_valid, frame_stats = check_frame_quality(frame_oscillators, min_bond, max_bond)

        if is_valid:
            filtered_dict[frame_idx] = frame_oscillators
        else:
            excluded_count += 1
            excluded_frames.append(frame_idx)
            overall_stats['excluded_frames'] += 1
            overall_stats['excluded_oscillators'] += len(frame_oscillators)

            if frame_stats['has_nan'] > 0:
                overall_stats['frames_with_nan'] += 1
            if frame_stats['bad_bonds'] > 0:
                overall_stats['frames_with_bad_bonds'] += 1

            # Print first few excluded frames
            if verbose and excluded_count <= 5:
                print(f"  Excluding frame {frame_idx}: {frame_stats['n_invalid']}/{frame_stats['n_oscillators']} invalid oscillators")
                if frame_stats['has_nan'] > 0:
                    print(f"    - {frame_stats['has_nan']} with NaN/Inf")
                if frame_stats['bad_bonds'] > 0:
                    print(f"    - {frame_stats['bad_bonds']} with bad bond lengths")

    if verbose:
        print(f"\nFrame filtering summary:")
        print(f"  Total frames: {overall_stats['total_frames']}")
        print(f"  Valid frames: {len(filtered_dict)} ({100*len(filtered_dict)/overall_stats['total_frames']:.1f}%)")
        print(f"  Excluded frames: {excluded_count} ({100*excluded_count/overall_stats['total_frames']:.1f}%)")
        if excluded_count > 0:
            print(f"    - With NaN/Inf: {overall_stats['frames_with_nan']}")
            print(f"    - With bad bonds: {overall_stats['frames_with_bad_bonds']}")

    return filtered_dict, excluded_count, overall_stats

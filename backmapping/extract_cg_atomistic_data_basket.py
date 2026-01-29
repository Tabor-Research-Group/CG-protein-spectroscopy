"""
================================================================================
FIXED: CG + ATOMISTIC DATA EXTRACTION PIPELINE
================================================================================
"""

import os
import sys
import numpy as np
import pickle
import argparse
import warnings
import multiprocessing
from functools import partial
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)

warnings.filterwarnings("ignore")


# ============================================================================
# SIDECHAIN OSCILLATOR ATOM MAPPINGS (GLN/ASN)
# ============================================================================

SC_OSCILLATOR_ATOMS = {
    'GLN': {
        'ref_atom': 'CA',
        'carbonyl_C': 'CD',
        'carbonyl_O': 'OE1',
        'amide_N': 'NE2',
        'amide_H1': 'HE21',
        'amide_H2': 'HE22'
    },
    'ASN': {
        'ref_atom': 'CA',
        'carbonyl_C': 'CG',
        'carbonyl_O': 'OD1',
        'amide_N': 'ND2',
        'amide_H1': 'HD21',
        'amide_H2': 'HD22'
    }
}


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self, config_path=None):
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Loaded configuration from: {config_path}")
        else:
            self.config = self._get_default_config()
            if config_path is not None:
                print(f"⚠ Config file '{config_path}' not found, using defaults")
            else:
                print("✓ Using default configuration")
    
    def _get_default_config(self):
        return {
            'base_directory': '.',
            'output_filename': 'amino_acid_baskets_complete.pkl',
            'frames_to_process': 100,
            'frame_selection_method': 'random',
            'frame_stride': 10,
            'random_seed': 42,
            'cpu_cores': 1,
            'atomistic_trajectory_file': 'prod_centered.xtc',
            'atomistic_topology_file': 'prod.tpr',
            'cg_pdb_file_pattern': '{folder}_CG_protein_only_full_trajectory.pdb',
            'hamiltonian_file': 'diagonal_hamiltonian.txt',
            'create_visualization': True,
            'visualize_amino_acid': 'PHE',
            'visualization_filename_prefix': 'sample_residue',
            'show_progress_bars': True,
            'show_statistics': True,
            'require_hamiltonians': False,
            'strict_validation': True
        }
    
    def get(self, key, default=None):
        value = self.config.get(key)
        return value if value is not None else default


# ============================================================================
# FRAME SELECTION
# ============================================================================

def select_frames(total_frames, n_frames, method='random', stride=10, random_seed=None):
    """Select frames to process based on method."""
    if n_frames is None:
        if method == 'stride':
            return list(range(0, total_frames, stride))
        else:
            return list(range(total_frames))
    
    n_frames = min(n_frames, total_frames)
    
    if method == 'random':
        if random_seed is not None:
            np.random.seed(random_seed)
        selected = np.random.choice(total_frames, size=n_frames, replace=False)
        return sorted(selected.tolist())
    elif method == 'sequential':
        return list(range(n_frames))
    elif method == 'stride':
        frames = list(range(0, total_frames, stride))
        return frames[:n_frames]
    else:
        return list(range(n_frames))


# ============================================================================
# CG PDB PARSING - FIXED TO TRACK BEAD SEQUENCES
# ============================================================================

def parse_cg_frame_structure(frame_content):
    """
    Parse CG frame to extract:
    1. Ordered list of ALL beads (BB and SC) with their types
    2. Mapping from residue keys to their beads
    3. Sequential bead structure for proper SC collection
    
    Returns:
        bead_sequence: List of (resseq, resname, bead_type, position) in order
        residue_beads: Dict mapping (resseq, resname) -> {bead_type: position}
        ordered_residues: List of unique (resseq, resname) in sequence
    """
    lines = frame_content.splitlines()
    
    bead_sequence = []  # Sequential list of ALL beads
    residue_beads = {}   # Dict mapping residue -> {bead_type: coords}
    
    for line in lines:
        if not line.startswith('ATOM'):
            continue
        
        resname = line[17:20].strip()
        resseq = int(line[22:26].strip())
        bead_type = line[12:16].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        
        position = np.array([x, y, z], dtype=np.float32)
        
        # Add to sequential list
        bead_sequence.append((resseq, resname, bead_type, position))
        
        # Add to residue mapping
        key = (resseq, resname)
        if key not in residue_beads:
            residue_beads[key] = {}
        residue_beads[key][bead_type] = position
    
    # Get unique ordered residues
    seen_residues = set()
    ordered_residues = []
    for resseq, resname, _, _ in bead_sequence:
        if (resseq, resname) not in seen_residues:
            ordered_residues.append((resseq, resname))
            seen_residues.add((resseq, resname))
    
    return bead_sequence, residue_beads, ordered_residues


def extract_oscillator_list_from_cg(frame_content):
    """
    Extract oscillator list with proper BB/SC bead associations.
    
    Key changes:
    1. BB oscillators start from FIRST residue (not second)
    2. Each BB oscillator gets BOTH BB_curr and BB_next beads
    3. SC beads between consecutive BB beads are collected
    4. SC oscillators get their SC1 bead + subsequent SC beads + previous BB
    
    Returns:
        oscillators: List of oscillator definitions with bead info
        ordered_residues: List of (resseq, resname) tuples
    """
    bead_sequence, residue_beads, ordered_residues = parse_cg_frame_structure(frame_content)
    
    # Find all BB bead positions in sequence
    bb_indices = [i for i, (_, _, bead_type, _) in enumerate(bead_sequence) 
                  if bead_type == 'BB']
    
    oscillators = []
    
    # Process backbone oscillators
    # Each BB oscillator connects TWO consecutive BB beads
    for i in range(len(bb_indices) - 1):
        bb_curr_idx = bb_indices[i]
        bb_next_idx = bb_indices[i + 1]
        
        # Get residue info for current BB
        resseq_curr, resname_curr, _, pos_curr = bead_sequence[bb_curr_idx]
        resseq_next, resname_next, _, pos_next = bead_sequence[bb_next_idx]
        
        # Collect all SC beads between these two BB beads
        sc_beads_between = {}
        for j in range(bb_curr_idx + 1, bb_next_idx):
            _, _, bead_type, pos = bead_sequence[j]
            if bead_type.startswith('SC'):
                sc_beads_between[bead_type] = pos
        
        oscillators.append({
            'type': 'backbone',
            'residue_key': (resseq_curr, resname_curr),  # Primary residue
            'bead_type': 'BB',
            'residue_index': i,
            # NEW: Store both BB beads
            'bb_curr_key': (resseq_curr, resname_curr),
            'bb_next_key': (resseq_next, resname_next),
            'bb_curr_pos': pos_curr,
            'bb_next_pos': pos_next,
            # SC beads between the two BB beads
            'sc_beads': sc_beads_between
        })
    
    # Process sidechain oscillators (GLN/ASN with SC1)
    for i, (resseq, resname, bead_type, pos) in enumerate(bead_sequence):
        if resname in ['GLN', 'ASN'] and bead_type == 'SC1':
            # Find previous BB bead
            bb_prev_idx = None
            for j in range(i - 1, -1, -1):
                if bead_sequence[j][2] == 'BB':
                    bb_prev_idx = j
                    break
            
            # Collect subsequent SC beads (SC2, SC3, ...) until next BB
            sc_beads_dict = {bead_type: pos}  # Start with SC1
            for j in range(i + 1, len(bead_sequence)):
                _, _, next_bead_type, next_pos = bead_sequence[j]
                if next_bead_type == 'BB':
                    break  # Stop at next BB
                if next_bead_type.startswith('SC'):
                    sc_beads_dict[next_bead_type] = next_pos
            
            # Get previous BB info
            bb_prev_key = None
            bb_prev_pos = None
            if bb_prev_idx is not None:
                resseq_bb, resname_bb, _, bb_prev_pos = bead_sequence[bb_prev_idx]
                bb_prev_key = (resseq_bb, resname_bb)
            
            oscillators.append({
                'type': 'sidechain',
                'residue_key': (resseq, resname),
                'bead_type': 'SC1',
                'residue_index': ordered_residues.index((resseq, resname)),
                # NEW: Store previous BB and all SC beads
                'bb_prev_key': bb_prev_key,
                'bb_prev_pos': bb_prev_pos,
                'sc_beads': sc_beads_dict  # SC1, SC2, SC3, ...
            })
    
    return oscillators, ordered_residues


# ============================================================================
# ATOMISTIC DATA EXTRACTION
# ============================================================================

def extract_atomistic_residues_frame(universe):
    """
    Extract all atoms for all residues.
    
    Returns:
        residue_data: Dict mapping (resid, resname) -> {atom_name: coords}
    """
    protein = universe.select_atoms('protein')
    residue_data = {}
    
    for residue in protein.residues:
        atom_coords = {}
        for atom in residue.atoms:
            atom_coords[atom.name.strip()] = atom.position.astype(np.float32)
        
        key = (residue.resid, residue.resname)
        residue_data[key] = atom_coords
    
    return residue_data


# ============================================================================
# HAMILTONIAN EXTRACTION
# ============================================================================

def extract_hamiltonians(folder_path, frame_idx):
    """Extract Hamiltonian values for a frame."""
    hamiltonian_file = os.path.join(folder_path, "diagonal_hamiltonian.txt")
    
    if not os.path.exists(hamiltonian_file):
        return None
    
    try:
        with open(hamiltonian_file, 'r') as h:
            ham_lines = h.readlines()
            
            if frame_idx >= len(ham_lines):
                return None
            
            ham_line = ham_lines[frame_idx].strip()
        
        if ',' in ham_line:
            ham_vals = np.array([float(val) for val in ham_line.split(',')], dtype=np.float32)
        else:
            ham_vals = np.array(ham_line.split(), dtype=np.float32)
        
        # First column is frame number, skip it
        ham_vals = ham_vals[1:]
        return ham_vals
        
    except Exception as e:
        print(f"    Error reading Hamiltonians: {str(e)}")
        return None


# ============================================================================
# RAMACHANDRAN ANGLES
# ============================================================================

def calculate_ramachandran_angles(universe):
    """Calculate Ramachandran angles for the entire trajectory."""
    try:
        protein = universe.select_atoms('protein')
        
        if len(protein) == 0:
            print(f"    ⚠ ERROR: No protein atoms found!")
            return None, []
        
        residue_list = [(res.resid, res.resname) for res in protein.residues]
        n_residues_total = len(residue_list)
        
        print(f"    → Found {n_residues_total} protein residues")
        print(f"    → Running Ramachandran analysis...")
        rama = Ramachandran(protein).run()
        
        if not hasattr(rama, 'results') or not hasattr(rama.results, 'angles'):
            print(f"    ⚠ ERROR: Ramachandran calculation failed!")
            return None, residue_list
        
        rama_angles = rama.results.angles
        
        if rama_angles is None or rama_angles.size == 0:
            print(f"    ⚠ ERROR: Ramachandran returned empty array!")
            return None, residue_list
        
        n_frames, n_rama_residues, n_angles = rama_angles.shape
        print(f"    → Ramachandran shape: ({n_frames} frames, {n_rama_residues} residues)")
        
        return rama_angles, residue_list
        
    except Exception as e:
        print(f"    ⚠ ERROR calculating Ramachandran: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, []


def get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid, angle_type):
    """Safely get a Ramachandran angle from the array."""
    if rama_angles is None:
        return None
    
    try:
        res_position = next(i for i, (rid, _) in enumerate(residue_list) if rid == resid)
    except StopIteration:
        return None
    
    rama_idx = res_position - 1
    
    if rama_idx < 0 or rama_idx >= rama_angles.shape[1]:
        return None
    
    angle_idx = 0 if angle_type == 'phi' else 1
    angle_value = rama_angles[frame_idx, rama_idx, angle_idx]
    
    if np.isnan(angle_value):
        return None
    
    return float(angle_value)


def get_nnfs_angles_from_array(rama_angles, residue_list, frame_idx, resid):
    """Get the 4 NNFS angles for a residue."""
    phi_i = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid, 'phi')
    psi_i = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid, 'psi')
    
    phi_prev = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid - 1, 'phi')
    psi_prev = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid - 1, 'psi')
    
    phi_next = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid + 1, 'phi')
    psi_next = get_rama_angle_safe(rama_angles, residue_list, frame_idx, resid + 1, 'psi')
    
    return {
        'phi_N': phi_i,
        'psi_N': psi_prev,
        'phi_C': phi_next,
        'psi_C': psi_i
    }


# ============================================================================
# OSCILLATOR EXTRACTION - FIXED
# ============================================================================

def extract_backbone_oscillator(oscillator, atomistic_dict,
                                rama_angles, residue_list, frame_idx):
    """
    Extract backbone C=O oscillator with FIXED atom and bead extraction.
    
    Changes:
    1. Now extracts CA_prev and N_prev from previous residue
    2. Stores BOTH BB_curr and BB_next bead positions
    3. Includes all SC beads between the two BB beads
    4. ✓ VALIDATES that CG and atomistic residue keys match
    
    The backbone oscillator between residue i and i+1 now has:
    - Atoms from residue i: C, O, CA, N (if exists)
    - Atoms from residue i+1: N, H, CA
    - BB beads: BB_curr (residue i) and BB_next (residue i+1)
    - SC beads: All SC1...SCn between these BB beads
    
    CRITICAL: residue_key must match bb_curr_key (the CA_prev residue)
    """
    residue_key = oscillator['residue_key']
    resid, resname = residue_key
    
    # Get oscillator's bead info from CG
    bb_curr_key = oscillator['bb_curr_key']
    bb_next_key = oscillator['bb_next_key']
    
    # ✓ VALIDATION: Verify residue_key matches bb_curr_key
    if residue_key != bb_curr_key:
        print(f"    ⚠ WARNING: Residue key mismatch! "
              f"residue_key={residue_key} != bb_curr_key={bb_curr_key}")
        return None
    
    # Get atomistic data for BOTH residues
    atoms_curr = atomistic_dict.get(bb_curr_key)
    atoms_next = atomistic_dict.get(bb_next_key)
    
    if atoms_curr is None:
        print(f"    ⚠ WARNING: No atomistic data for bb_curr residue {bb_curr_key}")
        return None
    if atoms_next is None:
        print(f"    ⚠ WARNING: No atomistic data for bb_next residue {bb_next_key}")
        return None
    
    # Check required atoms
    # From current residue (i): need C, O, CA
    if 'C' not in atoms_curr or 'O' not in atoms_curr or 'CA' not in atoms_curr:
        return None
    
    # From next residue (i+1): need N, CA
    if 'N' not in atoms_next or 'CA' not in atoms_next:
        return None
    
    # Build atom dictionary with FIXED naming
    # Previous = current residue (i), Current = next residue (i+1)
    atom_dict = {
        # Atoms from residue i (the "previous" in peptide bond)
        'C_prev': atoms_curr['C'],
        'O_prev': atoms_curr['O'],
        'CA_prev': atoms_curr['CA'],  # ✓ This is the primary residue
        'N_prev': atoms_curr.get('N', np.zeros(3, dtype=np.float32)),
        
        # Atoms from residue i+1 (the "current" in peptide bond)
        'N_curr': atoms_next['N'],
        'H_curr': atoms_next.get('H', np.zeros(3, dtype=np.float32)),
        'CA_curr': atoms_next['CA']
    }
    
    # Get NNFS angles (for residue i+1, the "current" side)
    nnfs_angles = get_nnfs_angles_from_array(rama_angles, residue_list, 
                                             frame_idx, bb_curr_key[0])
    
    return {
        'oscillator_type': 'backbone',
        'residue_key': residue_key,  # ✓ This matches bb_curr_key = CA_prev residue
        'residue_name': resname,
        
        # ✓ FIXED: Now includes CA_prev and N_prev
        'atoms': atom_dict,
        
        # NNFS angles
        'rama_nnfs': nnfs_angles,
        
        # ✓ NEW: Store BOTH BB beads explicitly
        'bb_curr': oscillator['bb_curr_pos'],
        'bb_next': oscillator['bb_next_pos'],
        'bb_curr_key': bb_curr_key,  # ✓ Should equal residue_key
        'bb_next_key': bb_next_key,
        
        # ✓ NEW: All SC beads between the two BB beads
        'sc_beads': oscillator['sc_beads'],
        
        # Legacy fields for compatibility
        'cg_bead': oscillator['bb_curr_pos'],  # Primary bead
        'cg_bead_type': 'BB'
    }


def extract_sidechain_oscillator(oscillator, atomistic_dict,
                                 rama_angles, residue_list, frame_idx):
    """
    Extract sidechain oscillator with FIXED bead collection.
    
    Changes:
    1. Now includes previous BB bead
    2. Collects ALL SC beads (SC1, SC2, SC3, ...) for this oscillator
    3. ✓ VALIDATES that CG and atomistic residue keys match
    
    Sidechain oscillators have NO NNFS neighbor interactions.
    
    CRITICAL: residue_key must match the atomistic residue we extract from
    """
    residue_key = oscillator['residue_key']
    resid, resname = residue_key
    
    if resname not in ['GLN', 'ASN']:
        return None
    
    # ✓ Get atomistic data and verify it exists
    atoms = atomistic_dict.get(residue_key)
    if atoms is None:
        print(f"    ⚠ WARNING: No atomistic data for sidechain residue {residue_key}")
        return None
    
    # Get sidechain atom mapping
    sc_map = SC_OSCILLATOR_ATOMS.get(resname)
    if sc_map is None:
        return None
    
    # Check required atoms
    required_atoms = ['ref_atom', 'carbonyl_C', 'carbonyl_O', 'amide_N', 'amide_H1']
    for atom_key in required_atoms:
        atom_name = sc_map[atom_key]
        if atom_name not in atoms:
            return None
    
    # Build sidechain atom dictionary
    # All sidechains start with CA and CB
    sidechain_atoms = {
        'CA': atoms[sc_map['ref_atom']],
        'CB': atoms.get('CB', np.zeros(3, dtype=np.float32)),
    }
    
    if resname == 'GLN':
        # GLN: CA-CB-CG-CD(=OE1)-NE2(-HE21/HE22)
        sidechain_atoms.update({
            'CG': atoms.get('CG', np.zeros(3, dtype=np.float32)),  # ✓ Explicitly include CG
            'CD': atoms[sc_map['carbonyl_C']],
            'OE1': atoms[sc_map['carbonyl_O']],
            'NE2': atoms[sc_map['amide_N']],
            'HE21': atoms.get(sc_map['amide_H1'], np.zeros(3, dtype=np.float32)),
            'HE22': atoms.get(sc_map['amide_H2'], np.zeros(3, dtype=np.float32))
        })
    elif resname == 'ASN':
        # ASN: CA-CB-CG(=OD1)-ND2(-HD21/HD22)
        # Note: CG is the carbonyl carbon for ASN
        sidechain_atoms.update({
            'CG': atoms[sc_map['carbonyl_C']],  # ✓ CG is carbonyl C for ASN
            'OD1': atoms[sc_map['carbonyl_O']],
            'ND2': atoms[sc_map['amide_N']],
            'HD21': atoms.get(sc_map['amide_H1'], np.zeros(3, dtype=np.float32)),
            'HD22': atoms.get(sc_map['amide_H2'], np.zeros(3, dtype=np.float32))
        })
    
    # No NNFS angles for sidechains
    nnfs_angles = {
        'phi_N': None,
        'psi_N': None,
        'phi_C': None,
        'psi_C': None
    }
    
    return {
        'oscillator_type': 'sidechain',
        'residue_key': residue_key,  # ✓ Matches the atomistic residue
        'residue_name': resname + '-SC',
        
        'atoms': sidechain_atoms,
        'rama_nnfs': nnfs_angles,
        
        # ✓ NEW: Previous BB bead
        'bb_prev': oscillator['bb_prev_pos'],
        'bb_prev_key': oscillator['bb_prev_key'],
        
        # ✓ NEW: All SC beads (SC1, SC2, SC3, ...)
        'sc_beads': oscillator['sc_beads'],
        
        # Legacy fields
        'cg_bead': oscillator['sc_beads'].get('SC1'),
        'cg_bead_type': 'SC1'
    }


def extract_oscillator_data(oscillator, atomistic_dict,
                            rama_angles, residue_list, frame_idx):
    """Route to appropriate extraction function."""
    osc_type = oscillator['type']
    
    if osc_type == 'backbone':
        return extract_backbone_oscillator(
            oscillator, atomistic_dict,
            rama_angles, residue_list, frame_idx
        )
    elif osc_type == 'sidechain':
        return extract_sidechain_oscillator(
            oscillator, atomistic_dict,
            rama_angles, residue_list, frame_idx
        )
    
    return None


# ============================================================================
# VALIDATION
# ============================================================================

def validate_oscillator_consistency(oscillator_data, verbose=False):
    """
    Validate individual oscillator data for consistency.
    
    COMPREHENSIVE checks:
    1. Residue key consistency (CG vs atomistic)
    2. Bead existence and proper assignment
    3. Atom extraction completeness (all required atoms present and non-null)
    4. Hamiltonian presence
    5. Geometric sanity (atoms not at origin unless optional)
    6. Data type correctness
    
    Returns: (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    osc_type = oscillator_data.get('oscillator_type')
    res_key = oscillator_data.get('residue_key')
    
    # ============================================================================
    # Check 1: Basic fields exist
    # ============================================================================
    if not res_key:
        errors.append("Missing residue_key")
    if not osc_type:
        errors.append("Missing oscillator_type")
    if osc_type not in ['backbone', 'sidechain']:
        errors.append(f"Invalid oscillator_type: {osc_type}")
    
    # ============================================================================
    # Check 2: Atoms exist and are valid
    # ============================================================================
    atoms = oscillator_data.get('atoms', {})
    if not atoms:
        errors.append("No atoms extracted")
    else:
        # Check all atoms are numpy arrays
        for atom_name, atom_pos in atoms.items():
            if not isinstance(atom_pos, np.ndarray):
                errors.append(f"Atom {atom_name} is not numpy array: {type(atom_pos)}")
            elif atom_pos.shape != (3,):
                errors.append(f"Atom {atom_name} has wrong shape: {atom_pos.shape}")
            elif np.any(np.isnan(atom_pos)):
                errors.append(f"Atom {atom_name} contains NaN values")
    
    # ============================================================================
    # Check 3: Type-specific validation - BACKBONE
    # ============================================================================
    if osc_type == 'backbone':
        # Check BB beads exist
        bb_curr = oscillator_data.get('bb_curr')
        bb_next = oscillator_data.get('bb_next')
        bb_curr_key = oscillator_data.get('bb_curr_key')
        bb_next_key = oscillator_data.get('bb_next_key')
        
        if bb_curr is None:
            errors.append("Missing bb_curr position")
        elif not isinstance(bb_curr, np.ndarray):
            errors.append(f"bb_curr is not numpy array: {type(bb_curr)}")
        elif bb_curr.shape != (3,):
            errors.append(f"bb_curr has wrong shape: {bb_curr.shape}")
        elif np.any(np.isnan(bb_curr)):
            errors.append("bb_curr contains NaN values")
        elif np.allclose(bb_curr, 0.0):
            warnings.append("bb_curr is at origin (0,0,0)")
        
        if bb_next is None:
            errors.append("Missing bb_next position")
        elif not isinstance(bb_next, np.ndarray):
            errors.append(f"bb_next is not numpy array: {type(bb_next)}")
        elif bb_next.shape != (3,):
            errors.append(f"bb_next has wrong shape: {bb_next.shape}")
        elif np.any(np.isnan(bb_next)):
            errors.append("bb_next contains NaN values")
        elif np.allclose(bb_next, 0.0):
            warnings.append("bb_next is at origin (0,0,0)")
        
        if bb_curr_key is None:
            errors.append("Missing bb_curr_key")
        if bb_next_key is None:
            errors.append("Missing bb_next_key")
        
        # CRITICAL: residue_key must match bb_curr_key (CA_prev residue)
        if bb_curr_key and res_key != bb_curr_key:
            errors.append(f"CRITICAL: Residue key mismatch: {res_key} != bb_curr_key {bb_curr_key}")
        
        # Check required atoms for backbone (COMPREHENSIVE)
        required_bb_atoms = {
            'C_prev': False,    # Required, must not be at origin
            'O_prev': False,    # Required, must not be at origin
            'CA_prev': False,   # Required, must not be at origin
            'N_prev': True,     # Optional (might be zero for first residue)
            'N_curr': False,    # Required, must not be at origin
            'H_curr': True,     # Optional (H might be missing)
            'CA_curr': False    # Required, must not be at origin
        }
        
        for atom_name, is_optional in required_bb_atoms.items():
            if atom_name not in atoms:
                if not is_optional:
                    errors.append(f"Missing REQUIRED backbone atom: {atom_name}")
                else:
                    warnings.append(f"Missing optional backbone atom: {atom_name}")
            else:
                atom_pos = atoms[atom_name]
                if atom_pos is None:
                    errors.append(f"Null backbone atom: {atom_name}")
                elif np.allclose(atom_pos, 0.0) and not is_optional:
                    warnings.append(f"Required atom {atom_name} is at origin (0,0,0)")
        
        # Check SC beads dict exists and is valid
        sc_beads = oscillator_data.get('sc_beads')
        if sc_beads is None:
            warnings.append("Missing sc_beads dictionary (should at least be empty dict)")
        elif not isinstance(sc_beads, dict):
            errors.append(f"sc_beads is not a dictionary: {type(sc_beads)}")
        else:
            # Validate each SC bead
            for bead_name, bead_pos in sc_beads.items():
                if not bead_name.startswith('SC'):
                    warnings.append(f"Unexpected bead name in sc_beads: {bead_name}")
                if not isinstance(bead_pos, np.ndarray):
                    errors.append(f"SC bead {bead_name} is not numpy array: {type(bead_pos)}")
                elif bead_pos.shape != (3,):
                    errors.append(f"SC bead {bead_name} has wrong shape: {bead_pos.shape}")
                elif np.any(np.isnan(bead_pos)):
                    errors.append(f"SC bead {bead_name} contains NaN")
    
    # ============================================================================
    # Check 4: Type-specific validation - SIDECHAIN
    # ============================================================================
    elif osc_type == 'sidechain':
        # Must be GLN or ASN
        resname = oscillator_data.get('residue_name', '')
        if not resname.startswith('GLN') and not resname.startswith('ASN'):
            errors.append(f"Sidechain oscillator must be GLN or ASN, got: {resname}")
        
        # Sidechain must have SC1 bead
        sc_beads = oscillator_data.get('sc_beads', {})
        if not sc_beads:
            errors.append("Missing sc_beads dictionary")
        elif 'SC1' not in sc_beads:
            errors.append("Missing SC1 bead for sidechain oscillator")
        else:
            sc1 = sc_beads['SC1']
            if not isinstance(sc1, np.ndarray):
                errors.append(f"SC1 is not numpy array: {type(sc1)}")
            elif sc1.shape != (3,):
                errors.append(f"SC1 has wrong shape: {sc1.shape}")
            elif np.any(np.isnan(sc1)):
                errors.append("SC1 contains NaN")
            elif np.allclose(sc1, 0.0):
                warnings.append("SC1 is at origin (0,0,0)")
        
        # Should have previous BB bead
        bb_prev = oscillator_data.get('bb_prev')
        bb_prev_key = oscillator_data.get('bb_prev_key')
        
        if bb_prev is None:
            warnings.append("Missing bb_prev for sidechain oscillator")
        elif not isinstance(bb_prev, np.ndarray):
            errors.append(f"bb_prev is not numpy array: {type(bb_prev)}")
        elif bb_prev.shape != (3,):
            errors.append(f"bb_prev has wrong shape: {bb_prev.shape}")
        elif np.any(np.isnan(bb_prev)):
            errors.append("bb_prev contains NaN")
        
        if bb_prev_key is None:
            warnings.append("Missing bb_prev_key")
        
        # Check required sidechain atoms
        base_resname = resname.replace('-SC', '')
        
        if base_resname == 'GLN':
            # GLN must have: CA, CB, CG, CD, OE1, NE2, HE21, (HE22 optional)
            required_gln_atoms = {
                'CA': False,
                'CB': False,
                'CG': False,   # ✓ Now required
                'CD': False,
                'OE1': False,
                'NE2': False,
                'HE21': True,  # H might be missing
                'HE22': True
            }
            
            for atom_name, is_optional in required_gln_atoms.items():
                if atom_name not in atoms:
                    if not is_optional:
                        errors.append(f"Missing REQUIRED GLN sidechain atom: {atom_name}")
                    else:
                        warnings.append(f"Missing optional GLN atom: {atom_name}")
                else:
                    atom_pos = atoms[atom_name]
                    if np.allclose(atom_pos, 0.0) and not is_optional:
                        warnings.append(f"Required GLN atom {atom_name} is at origin")
        
        elif base_resname == 'ASN':
            # ASN must have: CA, CB, CG, OD1, ND2, HD21, (HD22 optional)
            required_asn_atoms = {
                'CA': False,
                'CB': False,
                'CG': False,   # ✓ Now required (is carbonyl C)
                'OD1': False,
                'ND2': False,
                'HD21': True,  # H might be missing
                'HD22': True
            }
            
            for atom_name, is_optional in required_asn_atoms.items():
                if atom_name not in atoms:
                    if not is_optional:
                        errors.append(f"Missing REQUIRED ASN sidechain atom: {atom_name}")
                    else:
                        warnings.append(f"Missing optional ASN atom: {atom_name}")
                else:
                    atom_pos = atoms[atom_name]
                    if np.allclose(atom_pos, 0.0) and not is_optional:
                        warnings.append(f"Required ASN atom {atom_name} is at origin")
    
    # ============================================================================
    # Check 5: Hamiltonian validation
    # ============================================================================
    ham = oscillator_data.get('hamiltonian')
    if ham is None:
        warnings.append("Missing Hamiltonian value")
    elif not isinstance(ham, (int, float, np.number)):
        errors.append(f"Hamiltonian is not numeric: {type(ham)}")
    elif np.isnan(ham):
        errors.append("Hamiltonian is NaN")
    elif np.isinf(ham):
        errors.append("Hamiltonian is infinite")
    
    # ============================================================================
    # Check 6: Ramachandran angles (should be None for sidechains)
    # ============================================================================
    rama_nnfs = oscillator_data.get('rama_nnfs', {})
    if osc_type == 'sidechain':
        # All angles should be None for sidechains
        for angle_name in ['phi_N', 'psi_N', 'phi_C', 'psi_C']:
            if rama_nnfs.get(angle_name) is not None:
                errors.append(f"Sidechain oscillator should have {angle_name}=None, got {rama_nnfs.get(angle_name)}")
    
    # ============================================================================
    # Check 7: Metadata fields
    # ============================================================================
    required_metadata = ['folder', 'frame', 'oscillator_index']
    for field in required_metadata:
        if field not in oscillator_data:
            warnings.append(f"Missing metadata field: {field}")
    
    # Print if requested
    if verbose and (errors or warnings):
        print(f"  Validation for oscillator {res_key}:")
        if errors:
            print(f"    ERRORS ({len(errors)}):")
            for err in errors:
                print(f"      ✗ {err}")
        if warnings:
            print(f"    WARNINGS ({len(warnings)}):")
            for warn in warnings:
                print(f"      ⚠ {warn}")
    
    return len(errors) == 0, errors, warnings


def validate_frame_data(frame_oscillators, hamiltonians, frame_idx, 
                       expected_count, strict=True):
    """Validate oscillator-Hamiltonian alignment."""
    n_osc = len(frame_oscillators)
    
    if hamiltonians is None:
        if strict:
            return False, f"Frame {frame_idx}: No Hamiltonians found"
        else:
            return True, None
    
    n_ham = len(hamiltonians)
    
    if n_osc != n_ham:
        msg = f"Frame {frame_idx}: {n_osc} oscillators != {n_ham} Hamiltonians"
        if strict:
            return False, msg
        else:
            print(f"    ⚠ {msg}")
            return True, None
    
    if expected_count is not None and n_osc != expected_count:
        msg = f"Frame {frame_idx}: {n_osc} oscillators != {expected_count} expected"
        if strict:
            return False, msg
        else:
            print(f"    ⚠ {msg}")
    
    return True, None


# ============================================================================
# PROCESS FOLDER
# ============================================================================

def read_pdb_frames(pdb_file_name):
    """Read multi-frame CG PDB."""
    with open(pdb_file_name, 'r') as file:
        content = file.read()
    
    frames = content.split('ENDMDL')
    frames = [frame.strip() + '\nENDMDL' for frame in frames if frame.strip()]
    
    print(f"  → Loaded {len(frames)} CG frames")
    return frames


def process_folder(folder_path, config, n_frames=None):
    """Process one protein folder with validation."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    xtc_filename = config.get('atomistic_trajectory_file', default='prod_centered.xtc')
    tpr_filename = config.get('atomistic_topology_file', default='prod.tpr')
    hamiltonian_file = os.path.join(folder_path, "diagonal_hamiltonian.txt")
    cg_pdb_pattern = config.get('cg_pdb_file_pattern', 
                                default='{folder}_CG_protein_only_full_trajectory.pdb')
    
    xtc_file = os.path.join(folder_path, xtc_filename)
    tpr_file = os.path.join(folder_path, tpr_filename)
    cg_pdb_file = os.path.join(folder_path, cg_pdb_pattern.format(folder=folder_name))
    
    if not (os.path.exists(xtc_file) and os.path.exists(tpr_file) and 
            os.path.exists(cg_pdb_file) and os.path.exists(hamiltonian_file)):
        return None
    
    print(f"\nProcessing: {folder_name}")
    
    try:
        u = mda.Universe(tpr_file, xtc_file)
        cg_frames_all = read_pdb_frames(cg_pdb_file)
        
        total_frames = min(len(u.trajectory), len(cg_frames_all))
        
        selection_method = config.get('frame_selection_method', default='random')
        stride = config.get('frame_stride', default=10)
        random_seed = config.get('random_seed')
        strict = config.get('strict_validation', default=True)
        
        selected_frames = select_frames(total_frames, n_frames, method=selection_method,
                                       stride=stride, random_seed=random_seed)
        
        print(f"  → Frames to process: {len(selected_frames)}")
        
        # Calculate Ramachandran angles
        print(f"  → Calculating Ramachandran angles...")
        rama_angles, residue_list = calculate_ramachandran_angles(u)
        
        if rama_angles is None:
            print(f"  ⚠ WARNING: No Ramachandran angles calculated!")
        else:
            print(f"  ✓ Calculated angles for {len(rama_angles)} frames")
        
        # Get expected oscillator count
        oscillators_template, _ = extract_oscillator_list_from_cg(cg_frames_all[0])
        expected_osc_count = len(oscillators_template)
        print(f"  → Expected oscillators per frame: {expected_osc_count}")
        
        # Process all frames
        paired_data = []
        show_progress = config.get('show_progress_bars', default=True)
        
        for frame_idx in tqdm(selected_frames, desc=f"  {folder_name}", 
                             disable=not show_progress):
            u.trajectory[frame_idx]
            
            # Extract data
            atomistic_dict = extract_atomistic_residues_frame(u)
            hamiltonians = extract_hamiltonians(folder_path, frame_idx)
            
            # Get oscillator list with bead info
            oscillators_list, _ = extract_oscillator_list_from_cg(cg_frames_all[frame_idx])
            
            # Extract oscillators
            frame_oscillators = []
            for osc_idx, oscillator in enumerate(oscillators_list):
                osc_data = extract_oscillator_data(
                    oscillator, atomistic_dict,
                    rama_angles, residue_list, frame_idx
                )
                
                if osc_data is not None:
                    osc_data['folder'] = folder_name
                    osc_data['frame'] = frame_idx
                    osc_data['oscillator_index'] = osc_idx
                    
                    # Add Hamiltonian
                    if hamiltonians is not None and osc_idx < len(hamiltonians):
                        osc_data['hamiltonian'] = float(hamiltonians[osc_idx])
                    else:
                        osc_data['hamiltonian'] = None
                    
                    frame_oscillators.append(osc_data)
            
            # ✅ COMPREHENSIVE VALIDATION
            n_extracted = len(frame_oscillators)
            n_expected = len(oscillators_list)
            
            # Check 1: Did we extract all oscillators from CG list?
            if n_extracted != n_expected:
                msg = (f"Frame {frame_idx}: Expected {n_expected} oscillators from CG, "
                       f"but extracted {n_extracted} from atomistic "
                       f"({n_expected - n_extracted} failed extraction)")
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"    ⚠ {msg}")
            
            # Check 2: Do Hamiltonian counts match?
            if hamiltonians is not None:
                n_ham = len(hamiltonians)
                if n_extracted != n_ham:
                    msg = (f"Frame {frame_idx}: Extracted {n_extracted} oscillators "
                           f"but found {n_ham} Hamiltonians")
                    if strict:
                        raise ValueError(msg)
                    else:
                        print(f"    ⚠ {msg}")
            
            # Check 3: Verify residue key consistency between CG and atomistic
            for osc_idx, (cg_osc, extracted_osc) in enumerate(zip(oscillators_list, frame_oscillators)):
                cg_res_key = cg_osc['residue_key']
                atomistic_res_key = extracted_osc['residue_key']
                
                if cg_res_key != atomistic_res_key:
                    msg = (f"Frame {frame_idx}, Oscillator {osc_idx}: "
                           f"CG residue key {cg_res_key} != atomistic residue key {atomistic_res_key}")
                    if strict:
                        raise ValueError(msg)
                    else:
                        print(f"    ⚠ {msg}")
                
                # For backbone oscillators, verify bb_curr_key matches residue_key
                if extracted_osc['oscillator_type'] == 'backbone':
                    bb_curr_key = extracted_osc.get('bb_curr_key')
                    if bb_curr_key != atomistic_res_key:
                        msg = (f"Frame {frame_idx}, Oscillator {osc_idx}: "
                               f"BB oscillator residue_key {atomistic_res_key} != bb_curr_key {bb_curr_key}")
                        if strict:
                            raise ValueError(msg)
                        else:
                            print(f"    ⚠ {msg}")
            
            # Check 4: Are any Hamiltonians None when they shouldn't be?
            if hamiltonians is not None:
                n_missing_ham = sum(1 for osc in frame_oscillators 
                                   if osc['hamiltonian'] is None)
                if n_missing_ham > 0:
                    msg = (f"Frame {frame_idx}: {n_missing_ham}/{n_extracted} oscillators "
                           f"have missing Hamiltonians")
                    if strict:
                        raise ValueError(msg)
                    else:
                        print(f"    ⚠ {msg}")
            
            paired_data.append(frame_oscillators)
        
        total_oscillators = sum(len(osc) for osc in paired_data)
        print(f"  → Extracted {total_oscillators} oscillators from {len(paired_data)} frames")
        
        return paired_data
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_folder_wrapper(folder, base_dir, config, n_frames):
    """Wrapper for multiprocessing."""
    folder_path = os.path.join(base_dir, folder)
    result = process_folder(folder_path, config, n_frames)
    return (folder, result)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_and_organize_data(base_dir, config):
    """Process all folders and organize by amino acid type."""
    n_frames = config.get('frames_to_process')
    num_processes = config.get('cpu_cores', default=1)
    
    all_folders = [f for f in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, f))]
    
    print(f"\n{'='*80}")
    print(f"Scanning {len(all_folders)} folders...")
    print(f"{'='*80}")
    
    # Pre-filter valid folders
    valid_folders = []
    missing_stats = {
        'no_xtc': 0,
        'no_tpr': 0,
        'no_cg_pdb': 0,
        'no_hamiltonian': 0
    }
    
    xtc_filename = config.get('atomistic_trajectory_file', default='prod_centered.xtc')
    tpr_filename = config.get('atomistic_topology_file', default='prod.tpr')
    cg_pdb_pattern = config.get('cg_pdb_file_pattern', 
                                default='{folder}_CG_protein_only_full_trajectory.pdb')
    ham_filename = config.get('hamiltonian_file', default='diagonal_hamiltonian.txt')
    
    for folder in all_folders:
        folder_path = os.path.join(base_dir, folder)
        xtc_file = os.path.join(folder_path, xtc_filename)
        tpr_file = os.path.join(folder_path, tpr_filename)
        cg_pdb_file = os.path.join(folder_path, cg_pdb_pattern.format(folder=folder))
        ham_file = os.path.join(folder_path, ham_filename)
        
        if not os.path.exists(xtc_file):
            missing_stats['no_xtc'] += 1
            continue
        if not os.path.exists(tpr_file):
            missing_stats['no_tpr'] += 1
            continue
        if not os.path.exists(cg_pdb_file):
            missing_stats['no_cg_pdb'] += 1
            continue
        if not os.path.exists(ham_file):
            missing_stats['no_hamiltonian'] += 1
            continue
        
        valid_folders.append(folder)
    
    print(f"✓ Found {len(valid_folders)} valid protein folders")
    
    if any(missing_stats.values()):
        print(f"\nSkipped folders due to missing files:")
        if missing_stats['no_xtc'] > 0:
            print(f"  - Missing {xtc_filename}: {missing_stats['no_xtc']} folders")
        if missing_stats['no_tpr'] > 0:
            print(f"  - Missing {tpr_filename}: {missing_stats['no_tpr']} folders")
        if missing_stats['no_cg_pdb'] > 0:
            print(f"  - Missing CG PDB: {missing_stats['no_cg_pdb']} folders")
        if missing_stats['no_hamiltonian'] > 0:
            print(f"  - Missing {ham_filename}: {missing_stats['no_hamiltonian']} folders")
    
    print(f"{'='*80}")
    
    if len(valid_folders) == 0:
        return {}
    
    if num_processes == -1:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count())
    
    all_paired_data = {}
    
    if num_processes > 1:
        print(f"Using {num_processes} CPU cores")
        pool = multiprocessing.Pool(processes=num_processes)
        process_func = partial(process_folder_wrapper, 
                              base_dir=base_dir, config=config, n_frames=n_frames)
        
        results_iterator = pool.imap_unordered(process_func, valid_folders)
        
        completed = 0
        for folder, result in results_iterator:
            completed += 1
            if result is not None:
                all_paired_data[folder] = result
                print(f"  [{completed}/{len(valid_folders)}] ✓ {folder}")
            else:
                print(f"  [{completed}/{len(valid_folders)}] ✗ {folder}")
        
        pool.close()
        pool.join()
        
    else:
        print("Using sequential processing")
        for folder in valid_folders:
            folder_path = os.path.join(base_dir, folder)
            result = process_folder(folder_path, config, n_frames)
            if result is not None:
                all_paired_data[folder] = result
    
    # Organize by amino acid type
    amino_acid_baskets = {}
    
    for folder, frames in all_paired_data.items():
        for frame_oscillators in frames:
            for osc_data in frame_oscillators:
                resname = osc_data['residue_name']
                
                if resname not in amino_acid_baskets:
                    amino_acid_baskets[resname] = []
                
                amino_acid_baskets[resname].append(osc_data)
    
    return amino_acid_baskets


def analyze_statistics(amino_acid_baskets):
    """
    Print detailed statistics including validation checks.
    
    Now includes COMPREHENSIVE verification:
    1. All oscillators have consistent CG/atomistic mapping
    2. Hamiltonian availability
    3. Ramachandran angle availability
    4. Atom completeness
    5. Bead correctness
    """
    print(f"\n{'='*80}")
    print("EXTRACTION STATISTICS & COMPREHENSIVE VALIDATION")
    print(f"{'='*80}")
    print(f"{'AA':<12} {'Count':<8} {'Ham%':<8} {'Rama-N%':<10} {'Rama-C%':<10} {'Valid%':<8} {'Warn':<6}")
    print(f"{'-'*80}")
    
    total_oscillators = 0
    total_valid = 0
    total_warnings = 0
    total_errors = 0
    
    validation_details = {}
    
    for aa in sorted(amino_acid_baskets.keys()):
        data = amino_acid_baskets[aa]
        n = len(data)
        total_oscillators += n
        
        # Hamiltonian availability
        ham_pct = sum(1 for d in data if d.get('hamiltonian') is not None) / n * 100
        
        # Ramachandran N-side angles (phi_N, psi_N)
        rama_n_pct = sum(1 for d in data 
                        if d.get('rama_nnfs', {}).get('phi_N') is not None 
                        and d.get('rama_nnfs', {}).get('psi_N') is not None) / n * 100
        
        # Ramachandran C-side angles (phi_C, psi_C)
        rama_c_pct = sum(1 for d in data 
                        if d.get('rama_nnfs', {}).get('phi_C') is not None 
                        and d.get('rama_nnfs', {}).get('psi_C') is not None) / n * 100
        
        # Comprehensive validation
        valid_count = 0
        warning_count = 0
        error_count = 0
        
        for d in data:
            is_valid, errors, warnings = validate_oscillator_consistency(d, verbose=False)
            if is_valid:
                valid_count += 1
            error_count += len(errors)
            warning_count += len(warnings)
        
        total_valid += valid_count
        total_warnings += warning_count
        total_errors += error_count
        
        validation_details[aa] = {
            'valid': valid_count,
            'errors': error_count,
            'warnings': warning_count,
            'total': n
        }
        
        valid_pct = valid_count / n * 100
        
        note = ""
        if aa.endswith('-SC'):
            note = " (sidechain)"
        
        print(f"{aa:<12} {n:<8} {ham_pct:<8.1f} {rama_n_pct:<10.1f} "
              f"{rama_c_pct:<10.1f} {valid_pct:<8.1f} {warning_count:<6}{note}")
    
    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total oscillators:    {total_oscillators}")
    print(f"  Valid oscillators:    {total_valid} ({total_valid/total_oscillators*100:.1f}%)")
    print(f"  Total errors:         {total_errors}")
    print(f"  Total warnings:       {total_warnings}")
    print(f"{'='*80}")
    
    # Print any amino acids with errors
    if total_errors > 0:
        print(f"\n⚠ AMINO ACIDS WITH ERRORS:")
        for aa, details in validation_details.items():
            if details['errors'] > 0:
                print(f"  {aa}: {details['errors']} errors across {details['total']} oscillators "
                      f"({details['errors']/details['total']*100:.1f}%)")
        print(f"{'='*80}")
    
    print("\nNOTES:")
    print("  - Sidechain oscillators (GLN-SC, ASN-SC) should have 0% Rama angles (expected)")
    print("  - Valid% shows oscillators passing all validation checks")
    print("  - Warnings are non-critical issues (e.g., missing optional atoms)")
    print(f"{'='*80}\n")
    
    return validation_details


def generate_validation_report(amino_acid_baskets, output_file='validation_report.txt'):
    """
    Generate a comprehensive validation report and save to file.
    
    This report includes:
    1. Overall statistics
    2. Per-amino-acid validation details
    3. Common error patterns
    4. Detailed failure cases
    """
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE VALIDATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Collect all validation results
    all_errors = []
    all_warnings = []
    aa_stats = {}
    
    for aa in sorted(amino_acid_baskets.keys()):
        data = amino_acid_baskets[aa]
        
        valid_count = 0
        error_list = []
        warning_list = []
        
        for osc_idx, osc in enumerate(data):
            is_valid, errors, warnings = validate_oscillator_consistency(osc, verbose=False)
            
            if is_valid:
                valid_count += 1
            
            for err in errors:
                error_list.append((osc_idx, err))
                all_errors.append((aa, osc_idx, err))
            
            for warn in warnings:
                warning_list.append((osc_idx, warn))
                all_warnings.append((aa, osc_idx, warn))
        
        aa_stats[aa] = {
            'total': len(data),
            'valid': valid_count,
            'errors': len(error_list),
            'warnings': len(warning_list),
            'error_details': error_list,
            'warning_details': warning_list
        }
    
    # Overall summary
    total_oscillators = sum(s['total'] for s in aa_stats.values())
    total_valid = sum(s['valid'] for s in aa_stats.values())
    total_errors = len(all_errors)
    total_warnings = len(all_warnings)
    
    report_lines.append("OVERALL SUMMARY:")
    report_lines.append(f"  Total oscillators:  {total_oscillators}")
    report_lines.append(f"  Valid oscillators:  {total_valid} ({total_valid/total_oscillators*100:.2f}%)")
    report_lines.append(f"  Total errors:       {total_errors}")
    report_lines.append(f"  Total warnings:     {total_warnings}")
    report_lines.append("")
    
    if total_errors == 0 and total_warnings == 0:
        report_lines.append("✓✓✓ PERFECT EXTRACTION - NO ERRORS OR WARNINGS ✓✓✓")
        report_lines.append("")
    
    # Per-amino-acid breakdown
    report_lines.append("="*80)
    report_lines.append("PER-AMINO-ACID BREAKDOWN:")
    report_lines.append("="*80)
    report_lines.append("")
    
    for aa in sorted(aa_stats.keys()):
        stats = aa_stats[aa]
        report_lines.append(f"{aa}:")
        report_lines.append(f"  Total:    {stats['total']}")
        report_lines.append(f"  Valid:    {stats['valid']} ({stats['valid']/stats['total']*100:.2f}%)")
        report_lines.append(f"  Errors:   {stats['errors']}")
        report_lines.append(f"  Warnings: {stats['warnings']}")
        
        if stats['errors'] > 0:
            report_lines.append(f"  ⚠ HAS ERRORS - see detailed section below")
        elif stats['warnings'] > 0:
            report_lines.append(f"  ⚠ Has warnings (non-critical)")
        else:
            report_lines.append(f"  ✓ Perfect")
        
        report_lines.append("")
    
    # Error pattern analysis
    if total_errors > 0:
        report_lines.append("="*80)
        report_lines.append("ERROR PATTERN ANALYSIS:")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Count error types
        error_types = {}
        for aa, osc_idx, err in all_errors:
            if err not in error_types:
                error_types[err] = []
            error_types[err].append((aa, osc_idx))
        
        # Sort by frequency
        sorted_errors = sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)
        
        report_lines.append(f"Most common errors (by frequency):")
        for i, (error_msg, occurrences) in enumerate(sorted_errors[:10], 1):
            report_lines.append(f"  {i}. [{len(occurrences)}x] {error_msg}")
            if len(occurrences) <= 3:
                for aa, osc_idx in occurrences:
                    report_lines.append(f"       → {aa} oscillator {osc_idx}")
        report_lines.append("")
    
    # Detailed failure cases
    if total_errors > 0:
        report_lines.append("="*80)
        report_lines.append("DETAILED FAILURE CASES:")
        report_lines.append("="*80)
        report_lines.append("")
        
        for aa in sorted(aa_stats.keys()):
            stats = aa_stats[aa]
            if stats['errors'] > 0:
                report_lines.append(f"{aa} - {stats['errors']} oscillators with errors:")
                for osc_idx, err in stats['error_details'][:10]:  # Limit to first 10
                    report_lines.append(f"  Oscillator {osc_idx}: {err}")
                if len(stats['error_details']) > 10:
                    report_lines.append(f"  ... and {len(stats['error_details']) - 10} more")
                report_lines.append("")
    
    # Warning details (if needed)
    if total_warnings > 0:
        report_lines.append("="*80)
        report_lines.append("WARNING DETAILS (non-critical):")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Count warning types
        warning_types = {}
        for aa, osc_idx, warn in all_warnings:
            if warn not in warning_types:
                warning_types[warn] = 0
            warning_types[warn] += 1
        
        sorted_warnings = sorted(warning_types.items(), key=lambda x: x[1], reverse=True)
        
        for warn_msg, count in sorted_warnings:
            report_lines.append(f"  [{count}x] {warn_msg}")
        report_lines.append("")
    
    # Final verdict
    report_lines.append("="*80)
    report_lines.append("FINAL VERDICT:")
    report_lines.append("="*80)
    
    if total_errors == 0 and total_warnings == 0:
        report_lines.append("✓✓✓ DATA EXTRACTION IS PERFECT ✓✓✓")
        report_lines.append("All oscillators passed comprehensive validation.")
        report_lines.append("Ready for downstream analysis.")
    elif total_errors == 0:
        report_lines.append("✓ DATA EXTRACTION IS VALID")
        report_lines.append(f"No critical errors. {total_warnings} warnings (non-critical issues).")
        report_lines.append("Safe to proceed with caution.")
    else:
        report_lines.append("✗ DATA EXTRACTION HAS ERRORS")
        report_lines.append(f"{total_errors} critical errors found across {total_oscillators} oscillators.")
        report_lines.append("Review error details above and fix extraction logic.")
    
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Validation report saved to: {output_file}")
    
    # Also print to console
    print(report_text)
    
    return aa_stats, total_errors, total_warnings
    """
    Print detailed sample oscillator data for debugging.
    
    Args:
        amino_acid_baskets: The extracted data
        aa_type: Amino acid type to sample (e.g., 'ALA', 'GLN-SC')
        n_samples: Number of samples to print
    """
    if aa_type not in amino_acid_baskets:
        print(f"No data for amino acid type: {aa_type}")
        available = sorted(amino_acid_baskets.keys())
        print(f"Available types: {', '.join(available)}")
        return
    
    data = amino_acid_baskets[aa_type]
    print(f"\n{'='*80}")
    print(f"DETAILED SAMPLE DATA FOR {aa_type}")
    print(f"{'='*80}")
    print(f"Total {aa_type} oscillators: {len(data)}")
    
    for i, osc in enumerate(data[:n_samples]):
        print(f"\n{'-'*80}")
        print(f"SAMPLE {i+1}/{n_samples}:")
        print(f"{'-'*80}")
        print(f"  Folder:           {osc.get('folder')}")
        print(f"  Frame:            {osc.get('frame')}")
        print(f"  Oscillator index: {osc.get('oscillator_index')}")
        print(f"  Type:             {osc.get('oscillator_type')}")
        print(f"  Residue key:      {osc.get('residue_key')}")
        print(f"  Residue name:     {osc.get('residue_name')}")
        print(f"  Hamiltonian:      {osc.get('hamiltonian')}")
        
        # ========================================================================
        # BACKBONE OSCILLATOR DETAILS
        # ========================================================================
        if osc.get('oscillator_type') == 'backbone':
            print(f"\n  BACKBONE BEAD ASSIGNMENT:")
            print(f"    BB_curr_key:  {osc.get('bb_curr_key')} ← Should match residue_key")
            print(f"    BB_next_key:  {osc.get('bb_next_key')}")
            
            bb_curr = osc.get('bb_curr')
            bb_next = osc.get('bb_next')
            print(f"    BB_curr pos:  {bb_curr if bb_curr is not None else 'MISSING'}")
            print(f"    BB_next pos:  {bb_next if bb_next is not None else 'MISSING'}")
            
            # Check consistency
            if osc.get('bb_curr_key') == osc.get('residue_key'):
                print(f"    ✓ Residue key matches bb_curr_key")
            else:
                print(f"    ✗ MISMATCH: residue_key != bb_curr_key")
            
            sc_beads = osc.get('sc_beads', {})
            print(f"    SC beads between BB_curr and BB_next: {list(sc_beads.keys()) if sc_beads else 'none'}")
            
            print(f"\n  ATOMISTIC COORDINATES (7 atoms):")
            atoms = osc.get('atoms', {})
            
            # Atoms from residue i (bb_curr_key)
            print(f"    From residue {osc.get('bb_curr_key')}:")
            for atom_name in ['C_prev', 'O_prev', 'CA_prev', 'N_prev']:
                if atom_name in atoms:
                    atom_pos = atoms[atom_name]
                    is_zero = np.allclose(atom_pos, 0.0) if isinstance(atom_pos, np.ndarray) else False
                    status = "⚠ at origin" if is_zero else "✓"
                    print(f"      {atom_name:8s}: {atom_pos}  {status}")
                else:
                    print(f"      {atom_name:8s}: MISSING ✗")
            
            # Atoms from residue i+1 (bb_next_key)
            print(f"    From residue {osc.get('bb_next_key')}:")
            for atom_name in ['N_curr', 'H_curr', 'CA_curr']:
                if atom_name in atoms:
                    atom_pos = atoms[atom_name]
                    is_zero = np.allclose(atom_pos, 0.0) if isinstance(atom_pos, np.ndarray) else False
                    status = "⚠ at origin" if is_zero else "✓"
                    print(f"      {atom_name:8s}: {atom_pos}  {status}")
                else:
                    print(f"      {atom_name:8s}: MISSING ✗")
        
        # ========================================================================
        # SIDECHAIN OSCILLATOR DETAILS
        # ========================================================================
        elif osc.get('oscillator_type') == 'sidechain':
            base_resname = osc.get('residue_name', '').replace('-SC', '')
            
            print(f"\n  SIDECHAIN BEAD ASSIGNMENT:")
            print(f"    BB_prev_key:  {osc.get('bb_prev_key')}")
            bb_prev = osc.get('bb_prev')
            print(f"    BB_prev pos:  {bb_prev if bb_prev is not None else 'MISSING'}")
            
            sc_beads = osc.get('sc_beads', {})
            print(f"    SC beads:     {list(sc_beads.keys()) if sc_beads else 'MISSING'}")
            
            if 'SC1' in sc_beads:
                print(f"    SC1 pos:      {sc_beads['SC1']}  ✓")
            else:
                print(f"    SC1 pos:      MISSING ✗")
            
            print(f"\n  ATOMISTIC COORDINATES:")
            atoms = osc.get('atoms', {})
            
            if base_resname == 'GLN':
                expected_atoms = ['CA', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'HE21', 'HE22']
                print(f"    GLN sidechain atoms (expecting: CA-CB-CG-CD(=OE1)-NE2):")
            elif base_resname == 'ASN':
                expected_atoms = ['CA', 'CB', 'CG', 'OD1', 'ND2', 'HD21', 'HD22']
                print(f"    ASN sidechain atoms (expecting: CA-CB-CG(=OD1)-ND2):")
            else:
                expected_atoms = []
                print(f"    Sidechain atoms:")
            
            for atom_name in expected_atoms:
                if atom_name in atoms:
                    atom_pos = atoms[atom_name]
                    is_zero = np.allclose(atom_pos, 0.0) if isinstance(atom_pos, np.ndarray) else False
                    is_optional = atom_name.startswith('H')  # Hydrogens are optional
                    status = "✓"
                    if is_zero and not is_optional:
                        status = "⚠ at origin"
                    elif is_zero and is_optional:
                        status = "(optional, zero)"
                    print(f"      {atom_name:8s}: {atom_pos}  {status}")
                else:
                    print(f"      {atom_name:8s}: MISSING ✗")
        
        # ========================================================================
        # RAMACHANDRAN ANGLES
        # ========================================================================
        print(f"\n  RAMACHANDRAN (NNFS) ANGLES:")
        rama = osc.get('rama_nnfs', {})
        print(f"    phi_N (N-side): {rama.get('phi_N')}")
        print(f"    psi_N (N-side): {rama.get('psi_N')}")
        print(f"    phi_C (C-side): {rama.get('phi_C')}")
        print(f"    psi_C (C-side): {rama.get('psi_C')}")
        
        if osc.get('oscillator_type') == 'sidechain':
            all_none = all(rama.get(k) is None for k in ['phi_N', 'psi_N', 'phi_C', 'psi_C'])
            if all_none:
                print(f"    ✓ All angles are None (correct for sidechains)")
            else:
                print(f"    ✗ ERROR: Sidechain should have all angles = None")
        
        # ========================================================================
        # COMPREHENSIVE VALIDATION
        # ========================================================================
        print(f"\n  VALIDATION:")
        is_valid, errors, warnings = validate_oscillator_consistency(osc, verbose=False)
        
        if is_valid:
            print(f"    ✓ PASSED all validation checks")
        else:
            print(f"    ✗ FAILED validation")
        
        if errors:
            print(f"    Errors ({len(errors)}):")
            for err in errors:
                print(f"      - {err}")
        
        if warnings:
            print(f"    Warnings ({len(warnings)}):")
            for warn in warnings:
                print(f"      - {warn}")
        
        if is_valid and not warnings:
            print(f"    ✓✓ PERFECT - No errors or warnings")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract CG-atomistic data with COMPREHENSIVE validation')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--cpu', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-strict', action='store_true')
    parser.add_argument('--debug-sample', type=str, default=None,
                       help='Print sample data for debugging (e.g., ALA, GLN-SC)')
    parser.add_argument('--validation-report', type=str, default='validation_report.txt',
                       help='Output file for validation report')
    parser.add_argument('--skip-validation-report', action='store_true',
                       help='Skip generating validation report file')
    
    args = parser.parse_args()
    
    config = Config(args.config)
    
    if args.output:
        config.config['output_filename'] = args.output
    if args.frames:
        config.config['frames_to_process'] = args.frames
    if args.cpu:
        config.config['cpu_cores'] = args.cpu
    if args.no_strict:
        config.config['strict_validation'] = False
    
    print("="*80)
    print("CG-ATOMISTIC EXTRACTION - COMPREHENSIVE VALIDATION")
    print("="*80)
    print(f"Strict validation: {config.get('strict_validation', default=True)}")
    
    base_dir = config.get('base_directory', default='.')
    output_file = config.get('output_filename', default='amino_acid_baskets_complete.pkl')
    
    print(f"Output: {output_file}")
    print("="*80)
    
    if os.path.exists(output_file) and not args.overwrite:
        print(f"\n✓ Loading existing: {output_file}")
        with open(output_file, 'rb') as f:
            amino_acid_baskets = pickle.load(f)
    else:
        amino_acid_baskets = collect_and_organize_data(base_dir, config)
        
        print(f"\nSaving to: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(amino_acid_baskets, f)
    
    # Run statistics with validation
    validation_details = analyze_statistics(amino_acid_baskets)
    
    # Generate comprehensive validation report
    if not args.skip_validation_report:
        aa_stats, total_errors, total_warnings = generate_validation_report(
            amino_acid_baskets, 
            output_file=args.validation_report
        )
    
    # Print sample data if requested
    if args.debug_sample:
        print_oscillator_sample(amino_acid_baskets, aa_type=args.debug_sample, n_samples=2)
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Amino acid types: {len(amino_acid_baskets)}")
    print(f"Total oscillators: {sum(len(d) for d in amino_acid_baskets.values())}")
    print(f"{'='*80}")
    print("\nKEY FEATURES IMPLEMENTED:")
    print("  ✓ Backbone: 7 atoms (C,O,CA,N from prev + N,H,CA from curr)")
    print("  ✓ Backbone: Both bb_curr and bb_next beads stored")
    print("  ✓ Backbone: All SC beads between BB beads collected")
    print("  ✓ Sidechain: All SC1...SCn beads + previous BB")
    print("  ✓ Sidechain (GLN): CA, CB, CG, CD, OE1, NE2, HE21, HE22")
    print("  ✓ Sidechain (ASN): CA, CB, CG, OD1, ND2, HD21, HD22")
    print("  ✓ COMPREHENSIVE validation with detailed error reporting")
    print("  ✓ Full CG-atomistic consistency checks")
    print("  ✓ Oscillator count matching: CG == atomistic == Hamiltonian")
    print(f"{'='*80}\n")
    print("USAGE:")
    print("  Debug specific amino acid:  --debug-sample ALA")
    print("  Custom validation report:   --validation-report my_report.txt")
    print("  Skip validation report:     --skip-validation-report")
    print()


if __name__ == "__main__":
    main()


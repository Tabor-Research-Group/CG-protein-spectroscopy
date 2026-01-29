from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Atom name conventions used in the pickle:
# - backbone atoms are stored under names "N", "CA", "C", "O", "H" after reconstruction
# - GLN/ASN sidechain atoms follow standard PDB atom names (CA, CB, CG, ...)
# -----------------------------------------------------------------------------

# Backbone (peptide) bonds within a residue i (when atoms exist)
BACKBONE_BONDS_INTRA = [
    ("N", "CA"),
    ("CA", "C"),
    ("C", "O"),
    ("N", "H"),  # may be missing; will be masked
]

# Peptide bond between residue i and i+1
BACKBONE_BOND_INTER = ("C", "N")

# Sidechain GLN and ASN bonds
GLN_BONDS = [
    ("CA", "CB"),
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "OE1"),
    ("CD", "NE2"),
    ("NE2", "HE21"),
    ("NE2", "HE22"),
]

ASN_BONDS = [
    ("CA", "CB"),
    ("CB", "CG"),
    ("CG", "OD1"),
    ("CG", "ND2"),
    ("ND2", "HD21"),
    ("ND2", "HD22"),
]

# Backbone angles (triples)
BACKBONE_ANGLES_INTRA = [
    ("N", "CA", "C"),
    ("CA", "C", "O"),
    ("H", "N", "CA"),  # optional
]

# Angles involving peptide bond (inter-residue)
# At residue i: (CA_i, C_i, N_{i+1})
BACKBONE_ANGLE_C_N = ("CA", "C", "N")
# At residue i+1: (C_i, N_{i+1}, CA_{i+1})
BACKBONE_ANGLE_N_CA = ("C", "N", "CA")

# Sidechain angles
GLN_ANGLES = [
    ("CA", "CB", "CG"),
    ("CB", "CG", "CD"),
    ("CG", "CD", "OE1"),
    ("CG", "CD", "NE2"),
    ("CD", "NE2", "HE21"),
    ("CD", "NE2", "HE22"),
]
ASN_ANGLES = [
    ("CA", "CB", "CG"),
    ("CB", "CG", "OD1"),
    ("CB", "CG", "ND2"),
    ("CG", "ND2", "HD21"),
    ("CG", "ND2", "HD22"),
]


# Fixed partial charges for backbone atoms (AMBER-like, internal residues).
# Used only for long-range electrostatic restraint across non-neighbor residues.
BACKBONE_PARTIAL_CHARGES = {
    "C": 0.5973,
    "O": -0.5679,
    "N": -0.4157,
    "H": 0.2719,
}


@dataclass(frozen=True)
class TopologyIndices:
    # Indices refer to per-frame atom array indices
    bond_pairs: np.ndarray          # [Nbonds,2]
    angle_triples: np.ndarray       # [Nangles,3]
    # Coulomb (subset)
    charged_atom_indices: np.ndarray  # [Nq]
    charged_atom_charges: np.ndarray  # [Nq]
    charged_atom_res_index: np.ndarray  # [Nq] residue indices for exclusion mask

    # Dipoles per peptide bond i->i+1
    dipole_C_idx: np.ndarray  # [Nbonds_pep]
    dipole_O_idx: np.ndarray  # [Nbonds_pep]
    dipole_Nnext_idx: np.ndarray  # [Nbonds_pep]
    dipole_res_i: np.ndarray  # [Nbonds_pep] residue index i (for choosing local frame)

    # Ramachandran dihedrals for internal residues
    phi_indices: np.ndarray  # [Nphi,4] (C_{i-1}, N_i, CA_i, C_i)
    psi_indices: np.ndarray  # [Npsi,4] (N_i, CA_i, C_i, N_{i+1})
    rama_resid: np.ndarray   # [Nphi] residue indices i for phi/psi arrays alignment


def _idx(atom_map: Dict[Tuple[int, str], Dict[str, int]], res_i: Tuple[int, str], atom_name: str) -> Optional[int]:
    d = atom_map.get(res_i)
    if d is None:
        return None
    return d.get(atom_name)


def build_topology_indices(
    residue_ids: Sequence[int],
    residue_names: Sequence[str],
    atom_index_by_residue: Dict[Tuple[int, str], Dict[str, int]],
) -> TopologyIndices:
    """Build bond/angle/dihedral index arrays from reconstructed per-residue atoms.

    Parameters
    ----------
    residue_ids:
        List of residue sequence numbers (as in the pickle).
    residue_names:
        Same length; 3-letter residue names (e.g., 'ALA', 'GLN', ...).
    atom_index_by_residue:
        Maps (resid,resname)-> {atom_name -> global_atom_index} for this frame.
        Atom names are expected to be those used in the reconstructed atom table
        (N, CA, C, O, H, plus sidechain names for GLN/ASN).

    Returns
    -------
    TopologyIndices with fixed bond/angle/dihedral indices.
    """
    # Bonds
    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []

    # Helper: residue key objects
    res_keys = [(int(rid), str(rn)) for rid, rn in zip(residue_ids, residue_names)]
    n_res = len(res_keys)

    # Intra-residue backbone
    for rk in res_keys:
        for a, b in BACKBONE_BONDS_INTRA:
            ia = _idx(atom_index_by_residue, rk, a)
            ib = _idx(atom_index_by_residue, rk, b)
            if ia is not None and ib is not None:
                bonds.append((ia, ib))

        for a, b, c in BACKBONE_ANGLES_INTRA:
            ia = _idx(atom_index_by_residue, rk, a)
            ib = _idx(atom_index_by_residue, rk, b)
            ic = _idx(atom_index_by_residue, rk, c)
            if ia is not None and ib is not None and ic is not None:
                angles.append((ia, ib, ic))

    # Inter-residue peptide bonds and angles
    for i in range(n_res - 1):
        rk_i = res_keys[i]
        rk_ip1 = res_keys[i + 1]

        iC = _idx(atom_index_by_residue, rk_i, "C")
        iNn = _idx(atom_index_by_residue, rk_ip1, "N")
        iCA = _idx(atom_index_by_residue, rk_i, "CA")
        iCAn = _idx(atom_index_by_residue, rk_ip1, "CA")
        if iC is not None and iNn is not None:
            bonds.append((iC, iNn))
        # angle (CA_i, C_i, N_{i+1})
        if iCA is not None and iC is not None and iNn is not None:
            angles.append((iCA, iC, iNn))
        # angle (C_i, N_{i+1}, CA_{i+1})
        if iC is not None and iNn is not None and iCAn is not None:
            angles.append((iC, iNn, iCAn))

    # Sidechain bonds/angles for GLN and ASN (including -SC residue_name variants)
    for rk, resname in zip(res_keys, residue_names):
        res_upper = str(resname).upper()
        if res_upper.startswith("GLN"):
            for a, b in GLN_BONDS:
                ia = _idx(atom_index_by_residue, rk, a)
                ib = _idx(atom_index_by_residue, rk, b)
                if ia is not None and ib is not None:
                    bonds.append((ia, ib))
            for a, b, c in GLN_ANGLES:
                ia = _idx(atom_index_by_residue, rk, a)
                ib = _idx(atom_index_by_residue, rk, b)
                ic = _idx(atom_index_by_residue, rk, c)
                if ia is not None and ib is not None and ic is not None:
                    angles.append((ia, ib, ic))

        if res_upper.startswith("ASN"):
            for a, b in ASN_BONDS:
                ia = _idx(atom_index_by_residue, rk, a)
                ib = _idx(atom_index_by_residue, rk, b)
                if ia is not None and ib is not None:
                    bonds.append((ia, ib))
            for a, b, c in ASN_ANGLES:
                ia = _idx(atom_index_by_residue, rk, a)
                ib = _idx(atom_index_by_residue, rk, b)
                ic = _idx(atom_index_by_residue, rk, c)
                if ia is not None and ib is not None and ic is not None:
                    angles.append((ia, ib, ic))

    # Charged atoms for Coulomb term (backbone only)
    charged_idx: List[int] = []
    charged_q: List[float] = []
    charged_res_i: List[int] = []

    for res_i, rk in enumerate(res_keys):
        atom_map = atom_index_by_residue.get(rk, {})
        for atom_name, q in BACKBONE_PARTIAL_CHARGES.items():
            ia = atom_map.get(atom_name)
            if ia is not None:
                charged_idx.append(int(ia))
                charged_q.append(float(q))
                charged_res_i.append(int(res_i))

    # Dipole indices per peptide group i (C_i, O_i, N_{i+1})
    dip_C: List[int] = []
    dip_O: List[int] = []
    dip_Nn: List[int] = []
    dip_res: List[int] = []
    for i in range(n_res - 1):
        rk_i = res_keys[i]
        rk_ip1 = res_keys[i + 1]
        iC = _idx(atom_index_by_residue, rk_i, "C")
        iO = _idx(atom_index_by_residue, rk_i, "O")
        iNn = _idx(atom_index_by_residue, rk_ip1, "N")
        if iC is not None and iO is not None and iNn is not None:
            dip_C.append(int(iC))
            dip_O.append(int(iO))
            dip_Nn.append(int(iNn))
            dip_res.append(int(i))

    # Ramachandran indices for internal residues
    phi_list: List[Tuple[int, int, int, int]] = []
    psi_list: List[Tuple[int, int, int, int]] = []
    rama_res: List[int] = []
    for i in range(1, n_res - 1):
        rk_im1 = res_keys[i - 1]
        rk_i = res_keys[i]
        rk_ip1 = res_keys[i + 1]
        C_im1 = _idx(atom_index_by_residue, rk_im1, "C")
        N_i = _idx(atom_index_by_residue, rk_i, "N")
        CA_i = _idx(atom_index_by_residue, rk_i, "CA")
        C_i = _idx(atom_index_by_residue, rk_i, "C")
        N_ip1 = _idx(atom_index_by_residue, rk_ip1, "N")
        if C_im1 is not None and N_i is not None and CA_i is not None and C_i is not None:
            phi_list.append((int(C_im1), int(N_i), int(CA_i), int(C_i)))
        if N_i is not None and CA_i is not None and C_i is not None and N_ip1 is not None:
            psi_list.append((int(N_i), int(CA_i), int(C_i), int(N_ip1)))
        # Keep residue index for reporting (only if at least one angle exists)
        if (phi_list and phi_list[-1][1] == int(N_i)) or (psi_list and psi_list[-1][0] == int(N_i)):
            rama_res.append(int(i))

    # Note: phi_list and psi_list may not align in length if missing atoms.
    # We'll compute losses with independent masks.

    return TopologyIndices(
        bond_pairs=np.asarray(bonds, dtype=np.int64) if bonds else np.zeros((0, 2), dtype=np.int64),
        angle_triples=np.asarray(angles, dtype=np.int64) if angles else np.zeros((0, 3), dtype=np.int64),
        charged_atom_indices=np.asarray(charged_idx, dtype=np.int64) if charged_idx else np.zeros((0,), dtype=np.int64),
        charged_atom_charges=np.asarray(charged_q, dtype=np.float32) if charged_q else np.zeros((0,), dtype=np.float32),
        charged_atom_res_index=np.asarray(charged_res_i, dtype=np.int64) if charged_res_i else np.zeros((0,), dtype=np.int64),
        dipole_C_idx=np.asarray(dip_C, dtype=np.int64) if dip_C else np.zeros((0,), dtype=np.int64),
        dipole_O_idx=np.asarray(dip_O, dtype=np.int64) if dip_O else np.zeros((0,), dtype=np.int64),
        dipole_Nnext_idx=np.asarray(dip_Nn, dtype=np.int64) if dip_Nn else np.zeros((0,), dtype=np.int64),
        dipole_res_i=np.asarray(dip_res, dtype=np.int64) if dip_res else np.zeros((0,), dtype=np.int64),
        phi_indices=np.asarray(phi_list, dtype=np.int64) if phi_list else np.zeros((0, 4), dtype=np.int64),
        psi_indices=np.asarray(psi_list, dtype=np.int64) if psi_list else np.zeros((0, 4), dtype=np.int64),
        rama_resid=np.asarray(rama_res, dtype=np.int64) if rama_res else np.zeros((0,), dtype=np.int64),
    )

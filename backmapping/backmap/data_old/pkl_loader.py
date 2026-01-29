from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np


ResKey = Tuple[int, str]


def load_baskets(path: str) -> Dict[str, List[dict]]:
    """Load amino_acid_baskets dict from pickle.

    Top-level is expected to be dict[str, list[oscillator_dict]].
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level object in pickle must be a dict.")
    return data


def group_oscillators_by_frame(
    amino_acid_baskets: Mapping[str, Iterable[dict]],
) -> Dict[Tuple[str, int], Dict[str, List[dict]]]:
    """Flatten amino_acid_baskets into groups keyed by (folder, frame)."""
    frame_groups: Dict[Tuple[str, int], Dict[str, List[dict]]] = defaultdict(
        lambda: {"backbone": [], "sidechain": []}
    )

    for _, osc_list in amino_acid_baskets.items():
        if not isinstance(osc_list, (list, tuple)):
            continue
        for osc in osc_list:
            folder = osc.get("folder")
            frame = osc.get("frame")
            if folder is None or frame is None:
                continue

            osc_type = osc.get("oscillator_type")
            if osc_type == "backbone":
                frame_groups[(folder, int(frame))]["backbone"].append(osc)
            elif osc_type == "sidechain":
                frame_groups[(folder, int(frame))]["sidechain"].append(osc)
            else:
                continue

    return frame_groups


def _is_missing_coord(arr: np.ndarray) -> bool:
    return (arr.shape == (3,)) and np.allclose(arr, 0.0, atol=1e-6)


def add_coord(
    table: Dict[ResKey, Dict[str, np.ndarray]],
    res_key: ResKey,
    name: str,
    coord: Any,
    atol: float = 1e-3,
) -> None:
    """Add a coordinate to table[(resid,resname)][name] with de-duplication.

    - Ignores coordinates that are all zeros (used as placeholders for missing atoms).
    - If the same (res, atom) appears multiple times, enforces numerical consistency.
    """
    if coord is None:
        return
    arr = np.asarray(coord, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Coordinate for {res_key}, {name} has wrong shape {arr.shape}")
    if _is_missing_coord(arr):
        return

    entry = table.setdefault(res_key, {})
    if name in entry:
        old = entry[name]
        if not np.allclose(old, arr, atol=atol):
            entry[name] = 0.5 * (old + arr)
    else:
        entry[name] = arr


def build_per_residue_atomistic(
    backbone_oscillators: List[dict],
    sidechain_oscillators: List[dict],
) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Reconstruct per-residue atom coordinates from oscillators."""
    atoms_by_residue: Dict[ResKey, Dict[str, np.ndarray]] = {}

    # Backbone oscillators: residue i and i+1
    for osc in backbone_oscillators:
        atoms = osc.get("atoms", {})
        res_key_i = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        bb_next_key = tuple(osc.get("bb_next_key")) if osc.get("bb_next_key") is not None else None

        if res_key_i is not None:
            resid_i, resname_i = int(res_key_i[0]), str(res_key_i[1])
            key_i: ResKey = (resid_i, resname_i)
            add_coord(atoms_by_residue, key_i, "C",  atoms.get("C_prev"))
            add_coord(atoms_by_residue, key_i, "O",  atoms.get("O_prev"))
            add_coord(atoms_by_residue, key_i, "CA", atoms.get("CA_prev"))
            add_coord(atoms_by_residue, key_i, "N",  atoms.get("N_prev"))

        if bb_next_key is not None:
            resid_j, resname_j = int(bb_next_key[0]), str(bb_next_key[1])
            key_j: ResKey = (resid_j, resname_j)
            add_coord(atoms_by_residue, key_j, "N",  atoms.get("N_curr"))
            add_coord(atoms_by_residue, key_j, "CA", atoms.get("CA_curr"))
            add_coord(atoms_by_residue, key_j, "H",  atoms.get("H_curr"))

    # Sidechain oscillators: add all stored atoms
    for osc in sidechain_oscillators:
        res_key = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        side_atoms = osc.get("atoms", {})
        if res_key is None or side_atoms is None:
            continue
        resid, resname = int(res_key[0]), str(res_key[1])
        key: ResKey = (resid, resname)
        for atom_name, coord in side_atoms.items():
            add_coord(atoms_by_residue, key, str(atom_name), coord)

    return atoms_by_residue


def build_per_residue_cg(
    backbone_oscillators: List[dict],
    sidechain_oscillators: List[dict],
) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Reconstruct per-residue CG bead coordinates."""
    cg_by_residue: Dict[ResKey, Dict[str, np.ndarray]] = {}

    # Backbone oscillators provide BB beads and SC beads for residue i
    for osc in backbone_oscillators:
        res_key_i = tuple(osc.get("bb_curr_key")) if osc.get("bb_curr_key") is not None else None
        res_key_j = tuple(osc.get("bb_next_key")) if osc.get("bb_next_key") is not None else None

        if res_key_i is not None:
            resid_i, resname_i = int(res_key_i[0]), str(res_key_i[1])
            key_i: ResKey = (resid_i, resname_i)
            add_coord(cg_by_residue, key_i, "BB", osc.get("bb_curr"))
            sc_beads = osc.get("sc_beads", {}) or {}
            for bead_name, coord in sc_beads.items():
                add_coord(cg_by_residue, key_i, str(bead_name), coord)

        if res_key_j is not None:
            resid_j, resname_j = int(res_key_j[0]), str(res_key_j[1])
            key_j: ResKey = (resid_j, resname_j)
            add_coord(cg_by_residue, key_j, "BB", osc.get("bb_next"))

    # Sidechain oscillators provide SC beads and bb_prev (compatibility)
    for osc in sidechain_oscillators:
        res_key = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        if res_key is None:
            continue
        resid, resname = int(res_key[0]), str(res_key[1])
        key: ResKey = (resid, resname)

        sc_beads = osc.get("sc_beads", {}) or {}
        for bead_name, coord in sc_beads.items():
            add_coord(cg_by_residue, key, str(bead_name), coord)

        bb_prev_key = osc.get("bb_prev_key")
        if bb_prev_key is not None and osc.get("bb_prev") is not None:
            bbk = (int(bb_prev_key[0]), str(bb_prev_key[1]))
            add_coord(cg_by_residue, bbk, "BB", osc.get("bb_prev"))

    return cg_by_residue

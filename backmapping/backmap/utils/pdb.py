from __future__ import annotations

"""PDB writing and structure aggregation utilities.

The training/inference code deals with oscillator-local graphs, but for
visualization (VMD, PyMOL) you usually want a *whole-frame* structure.

This module provides:
- helpers to aggregate atomistic atoms and CG beads per residue from a list of
  oscillator dictionaries
- helpers to aggregate predicted atom coordinates (from oscillator predictions)
- a minimal PDB writer that writes multiple chains so you can overlay:
    chain A : ground truth atomistic
    chain B : predicted atomistic
    chain C : CG beads

We intentionally keep PDB output minimal but valid enough for VMD.
"""

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


ResKey = Tuple[int, str]  # (resid, resname)


# -----------------------------------------------------------------------------
# Table aggregation helpers
# -----------------------------------------------------------------------------


def add_coord(
    table: MutableMapping[ResKey, Dict[str, np.ndarray]],
    res_key: ResKey,
    name: str,
    coord: Optional[np.ndarray],
    *,
    atol: float = 1e-3,
    ignore_zeros: bool = True,
) -> None:
    """Add a coordinate to table[(resid,resname)][name] with simple de-dup.

    The pickle sometimes repeats the same atom/bead coordinates across multiple
    oscillators. We keep the first coordinate unless a repeated occurrence differs
    beyond `atol`, in which case we store the average.

    Parameters
    ----------
    ignore_zeros:
        If True, coordinates that are exactly (0,0,0) (within a small tolerance)
        are treated as missing and ignored.
    """
    if coord is None:
        return

    arr = np.asarray(coord, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Coordinate for {res_key}, {name} has wrong shape {arr.shape}")

    if ignore_zeros and np.allclose(arr, 0.0, atol=1e-6):
        return

    entry = table.setdefault(res_key, {})
    if name in entry:
        old = entry[name]
        if not np.allclose(old, arr, atol=atol):
            entry[name] = 0.5 * (old + arr)
    else:
        entry[name] = arr


def canonical_backbone_atom_name(aname: str) -> Tuple[str, int]:
    """Return canonical PDB atom name + residue offset for backbone keys.

    The oscillator graph uses names like "N_prev" and "CA_curr".
    For PDB writing we map these to standard atom names ("N", "CA", ...),
    and tell you whether they belong to residue 0 (prev) or residue 1 (curr).

    Returns
    -------
    (pdb_name, resid_local)
    """
    m = {
        "N_prev": ("N", 0),
        "CA_prev": ("CA", 0),
        "C_prev": ("C", 0),
        "O_prev": ("O", 0),
        "N_curr": ("N", 1),
        "H_curr": ("H", 1),
        "CA_curr": ("CA", 1),
    }
    if aname not in m:
        # fallback: strip suffix if present
        if aname.endswith("_prev"):
            return aname.replace("_prev", "")[:4], 0
        if aname.endswith("_curr"):
            return aname.replace("_curr", "")[:4], 1
        return aname[:4], 0
    return m[aname]


# -----------------------------------------------------------------------------
# Aggregation from oscillator dicts
# -----------------------------------------------------------------------------


def build_per_residue_atomistic(
    backbone_oscillators: Iterable[Mapping],
    sidechain_oscillators: Iterable[Mapping],
) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Aggregate ground-truth atomistic coordinates per residue."""
    atoms_by_residue: Dict[ResKey, Dict[str, np.ndarray]] = {}

    # Backbone oscillators provide backbone atoms for residue i and i+1
    for osc in backbone_oscillators:
        atoms = osc.get("atoms", {}) or {}
        res_key_i = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        bb_next_key = tuple(osc.get("bb_next_key")) if osc.get("bb_next_key") is not None else None

        if res_key_i is not None:
            add_coord(atoms_by_residue, res_key_i, "C", atoms.get("C_prev"))
            add_coord(atoms_by_residue, res_key_i, "O", atoms.get("O_prev"))
            add_coord(atoms_by_residue, res_key_i, "CA", atoms.get("CA_prev"))
            add_coord(atoms_by_residue, res_key_i, "N", atoms.get("N_prev"))

        if bb_next_key is not None:
            add_coord(atoms_by_residue, bb_next_key, "N", atoms.get("N_curr"))
            add_coord(atoms_by_residue, bb_next_key, "CA", atoms.get("CA_curr"))
            add_coord(atoms_by_residue, bb_next_key, "H", atoms.get("H_curr"))

    # Sidechain oscillators provide sidechain atoms for GLN/ASN etc.
    for osc in sidechain_oscillators:
        res_key = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        side_atoms = osc.get("atoms", {}) or {}
        if res_key is None:
            continue

        for atom_name, coord in side_atoms.items():
            add_coord(atoms_by_residue, res_key, str(atom_name), coord)

    return atoms_by_residue


def aggregate_atomistic_from_oscillators(oscillators: Iterable[Mapping]) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Convenience wrapper: aggregate GT atomistic atoms from a mixed oscillator list.

    Parameters
    ----------
    oscillators:
        Iterable containing both backbone and sidechain oscillator dicts.
    """
    backbone = [o for o in oscillators if str(o.get("oscillator_type", "")) == "backbone"]
    sidechain = [o for o in oscillators if str(o.get("oscillator_type", "")) == "sidechain"]
    return build_per_residue_atomistic(backbone, sidechain)


def build_per_residue_cg(
    backbone_oscillators: Iterable[Mapping],
    sidechain_oscillators: Iterable[Mapping],
) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Aggregate CG bead coordinates per residue."""
    cg_by_residue: Dict[ResKey, Dict[str, np.ndarray]] = {}

    # Backbone oscillators provide BB beads for residue i and i+1
    for osc in backbone_oscillators:
        res_key_i = tuple(osc.get("bb_curr_key")) if osc.get("bb_curr_key") is not None else None
        res_key_ip1 = tuple(osc.get("bb_next_key")) if osc.get("bb_next_key") is not None else None

        if res_key_i is not None:
            add_coord(cg_by_residue, res_key_i, "BB", osc.get("bb_curr"))
        if res_key_ip1 is not None:
            add_coord(cg_by_residue, res_key_ip1, "BB", osc.get("bb_next"))

    # Sidechain oscillators provide SC beads for the sidechain residue
    for osc in sidechain_oscillators:
        res_key = tuple(osc.get("residue_key")) if osc.get("residue_key") is not None else None
        sc_beads = osc.get("sc_beads", {}) or {}
        if res_key is None:
            continue

        for bead_name, coord in sc_beads.items():
            add_coord(cg_by_residue, res_key, str(bead_name), coord)

        # Ensure BB for this residue using bb_prev
        bb_prev_key = osc.get("bb_prev_key")
        if bb_prev_key is not None and osc.get("bb_prev") is not None:
            add_coord(cg_by_residue, tuple(bb_prev_key), "BB", osc.get("bb_prev"))

    return cg_by_residue


def aggregate_cg_from_oscillators(oscillators: Iterable[Mapping]) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Convenience wrapper: aggregate CG beads from a mixed oscillator list."""
    backbone = [o for o in oscillators if str(o.get("oscillator_type", "")) == "backbone"]
    sidechain = [o for o in oscillators if str(o.get("oscillator_type", "")) == "sidechain"]
    return build_per_residue_cg(backbone, sidechain)


def aggregate_predicted_from_oscillator_predictions(
    samples: List[Mapping],
    pred_atom_pos_global: np.ndarray,
    *,
    atol: float = 1e-3,
) -> Dict[ResKey, Dict[str, np.ndarray]]:
    """Aggregate predicted atom positions into a per-residue table.

    This function assumes `pred_atom_pos_global` is ordered exactly like the
    concatenation performed by :func:`backmap.data.collate.collate_graph_samples`:
    atoms are appended sample-by-sample, preserving each sample's internal atom order.
    """
    pred_atom_pos_global = np.asarray(pred_atom_pos_global, dtype=float)
    pred_atoms: Dict[ResKey, Dict[str, np.ndarray]] = {}
    off = 0
    for s in samples:
        na = int(s["x0_local"].shape[0])
        block = pred_atom_pos_global[off : off + na]
        off += na

        residue_keys = list(s.get("meta_residue_keys", []))
        atom_names = list(s.get("meta_atom_names", []))
        atom_res_local = np.asarray(s["atom_res"].detach().cpu().numpy(), dtype=int)
        osc_type = str(s.get("oscillator_type", ""))
        add_predicted_atoms_from_graph(
            pred_atoms,
            residue_keys=residue_keys,
            atom_names=atom_names,
            atom_res_local=atom_res_local,
            atom_pos=block,
            osc_type=osc_type,
            atol=atol,
        )
    return pred_atoms


# -----------------------------------------------------------------------------
# Aggregation from model predictions
# -----------------------------------------------------------------------------


def add_predicted_atoms_from_graph(
    pred_atoms_by_residue: MutableMapping[ResKey, Dict[str, np.ndarray]],
    *,
    residue_keys: List[ResKey],
    atom_names: List[str],
    atom_res_local: np.ndarray,
    atom_pos: np.ndarray,
    osc_type: str,
    atol: float = 1e-3,
) -> None:
    """Merge one oscillator prediction into a per-residue table."""
    atom_pos = np.asarray(atom_pos, dtype=float)
    atom_res_local = np.asarray(atom_res_local, dtype=int)
    if atom_pos.ndim != 2 or atom_pos.shape[1] != 3:
        raise ValueError(f"atom_pos must be [Na,3], got {atom_pos.shape}")

    for i in range(atom_pos.shape[0]):
        resid_local = int(atom_res_local[i])
        res_key = residue_keys[resid_local]

        aname = str(atom_names[i])
        if osc_type == "backbone":
            pdb_name, _ = canonical_backbone_atom_name(aname)
        else:
            pdb_name = aname

        add_coord(pred_atoms_by_residue, res_key, pdb_name, atom_pos[i], atol=atol, ignore_zeros=False)


# -----------------------------------------------------------------------------
# PDB writing
# -----------------------------------------------------------------------------


def pdb_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain_id: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    *,
    record: str = "ATOM",
    element: Optional[str] = None,
) -> str:
    """Format a PDB ATOM/HETATM line.

    This is not a full PDB writer, but is sufficient for VMD and most viewers.
    """
    if element is None:
        element = (name.strip()[0] if name.strip() else "C").upper()

    atom_name = name[:4].rjust(4)
    resname = (resname or "UNK")[:3].upper()
    chain_id = (chain_id or "A")[:1]

    line = (
        f"{record:<6}{serial:5d} {atom_name:s} "
        f"{resname:>3s} {chain_id:1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          "
        f"{element:>2s}\n"
    )
    return line


def write_multichain_pdb(
    *,
    out_path: str,
    atoms_A: Optional[Dict[ResKey, Dict[str, np.ndarray]]] = None,
    atoms_B: Optional[Dict[ResKey, Dict[str, np.ndarray]]] = None,
    beads_C: Optional[Dict[ResKey, Dict[str, np.ndarray]]] = None,
    chain_A: str = "A",
    chain_B: str = "B",
    chain_C: str = "C",
) -> None:
    """Write a PDB with up to three chains.

    Chain usage:
    - chain_A: ground truth atomistic
    - chain_B: predicted atomistic
    - chain_C: CG beads
    """
    serial = 1
    with open(out_path, "w") as f:
        if atoms_A is not None:
            for (resid, resname) in sorted(atoms_A.keys(), key=lambda k: k[0]):
                for atom_name, coord in sorted(atoms_A[(resid, resname)].items()):
                    x, y, z = coord
                    f.write(
                        pdb_atom_line(
                            serial,
                            str(atom_name),
                            str(resname),
                            chain_A,
                            int(resid),
                            float(x),
                            float(y),
                            float(z),
                            record="ATOM",
                        )
                    )
                    serial += 1

        if atoms_B is not None:
            for (resid, resname) in sorted(atoms_B.keys(), key=lambda k: k[0]):
                for atom_name, coord in sorted(atoms_B[(resid, resname)].items()):
                    x, y, z = coord
                    f.write(
                        pdb_atom_line(
                            serial,
                            str(atom_name),
                            str(resname),
                            chain_B,
                            int(resid),
                            float(x),
                            float(y),
                            float(z),
                            record="ATOM",
                        )
                    )
                    serial += 1

        if beads_C is not None:
            for (resid, resname) in sorted(beads_C.keys(), key=lambda k: k[0]):
                for bead_name, coord in sorted(beads_C[(resid, resname)].items()):
                    x, y, z = coord
                    f.write(
                        pdb_atom_line(
                            serial,
                            str(bead_name),
                            str(resname),
                            chain_C,
                            int(resid),
                            float(x),
                            float(y),
                            float(z),
                            record="HETATM",
                            element="C",
                        )
                    )
                    serial += 1

        f.write("END\n")

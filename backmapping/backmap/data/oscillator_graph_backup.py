from __future__ import annotations

"""Graph construction from a single oscillator dictionary.

This module converts **one oscillator entry** from your pickle into a graph that
can be batched and consumed by :class:`backmap.model.gnn.BackmapGNN`.

Why oscillator-local graphs?
----------------------------
Your earlier training runs produced *all-zero* losses. The most common reason is
that the model/loss pipeline is accidentally operating on an *empty atom set*
(e.g., Na==0) or an empty mask. This builder enforces strong invariants:

- At least one supervised atom must exist after filtering (Na>0)
- Returned indices (bonds/angles/dihedrals/dipoles) are deterministic and stable

Returned tensors
----------------
Each graph sample dict contains:

- Local atom targets (x0_local) and global atom targets (atom_pos0)
- Residue anchor BB positions + residue-local frames (bb_pos, bb_frames)
- Bead nodes + atom nodes, and a fully-connected edge list by default
- Convenience topology indices for losses/metrics:
    bond_index     [2, Nbonds]
    angle_index    [3, Nangles]
    dihedral_index [4, Ndihedrals]
    dipole_index   [3, Ndipoles]  columns are (C_idx, O_idx, N_idx) with N_idx=-1 if missing

Metadata (python objects)
-------------------------
We also include metadata needed for debugging and PDB writing:

- meta_folder, meta_frame, meta_oscillator_index
- meta_residue_keys: list[(resid:int, resname:str)] for each local residue
- meta_atom_names: list[str] of original pickle atom keys
- meta_atom_names_pdb: list[str] canonical atom names for PDB output

"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from backmap.geometry.frames import compute_residue_local_frames, global_to_local


def _as_float_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert array-like to a 1D/2D float tensor without copying more than needed."""
    arr = np.asarray(x, dtype=np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _is_zero_pos(x: np.ndarray, tol: float = 1e-8) -> bool:
    """True if a position is effectively the origin (used in the dataset for "missing")."""
    return bool(np.all(np.abs(x) < tol))


def _tuple_reskey(x: Any) -> Tuple[int, str] | None:
    """Convert (resid, resname) like values to a typed tuple."""
    if x is None:
        return None
    if isinstance(x, tuple) and len(x) == 2:
        return (int(x[0]), str(x[1]))
    if isinstance(x, list) and len(x) == 2:
        return (int(x[0]), str(x[1]))
    return None


# Canonical mapping for backbone oscillator atom keys → PDB atom names and residue slot.
# Slot 0 == residue i ("prev"), Slot 1 == residue i+1 ("curr")
_BACKBONE_PDB_MAP: Dict[str, Tuple[str, int]] = {
    "N_prev": ("N", 0),
    "CA_prev": ("CA", 0),
    "C_prev": ("C", 0),
    "O_prev": ("O", 0),
    "N_curr": ("N", 1),
    "H_curr": ("H", 1),
    "CA_curr": ("CA", 1),
}


def canonical_pdb_atom_name(osc_type: str, atom_key: str) -> str:
    """Return a PDB-friendly atom name.

    For backbone oscillators, the pickle uses "*_prev"/"*_curr" names; we map
    those to standard backbone atom names.

    For sidechain oscillators, we keep the original atom name (CA, CB, CG, ...).
    """
    if osc_type == "backbone" and atom_key in _BACKBONE_PDB_MAP:
        return _BACKBONE_PDB_MAP[atom_key][0]
    return str(atom_key)


@dataclass(frozen=True)
class GraphVocab:
    """Vocabularies for categorical embeddings."""

    resname_to_id: Dict[str, int]
    name_to_id: Dict[str, int]
    atom_group_to_id: Dict[str, int]

    @property
    def num_resnames(self) -> int:
        return len(self.resname_to_id)

    @property
    def num_names(self) -> int:
        return len(self.name_to_id)

    @property
    def num_atom_groups(self) -> int:
        return len(self.atom_group_to_id)

    @property
    def num_node_types(self) -> int:
        """Number of node type IDs.

        We use a simple 2-type scheme:
        0 = bead node (BB/SC beads)
        1 = atom node
        """
        return 2

    @staticmethod
    def from_sets(resnames: Sequence[str], names: Sequence[str], atom_groups: Sequence[str]) -> "GraphVocab":
        return GraphVocab(
            resname_to_id={s: i for i, s in enumerate(resnames)},
            name_to_id={s: i for i, s in enumerate(names)},
            atom_group_to_id={s: i for i, s in enumerate(atom_groups)},
        )

    def resname_id(self, s: str) -> int:
        return self.resname_to_id.get(s, self.resname_to_id.get("UNK", 0))

    def name_id(self, s: str) -> int:
        return self.name_to_id.get(s, self.name_to_id.get("UNK", 0))

    def atom_group_id(self, s: str) -> int:
        return self.atom_group_to_id.get(s, self.atom_group_to_id.get("UNK", 0))


# -----------------------------------------------------------------------------
# Topology helpers
# -----------------------------------------------------------------------------


def _maybe_add_pair(out: List[Tuple[int, int]], name_to_idx: Dict[str, int], a: str, b: str) -> None:
    ia = name_to_idx.get(a)
    ib = name_to_idx.get(b)
    if ia is None or ib is None:
        return
    out.append((ia, ib))


def _maybe_add_triple(out: List[Tuple[int, int, int]], name_to_idx: Dict[str, int], a: str, b: str, c: str) -> None:
    ia = name_to_idx.get(a)
    ib = name_to_idx.get(b)
    ic = name_to_idx.get(c)
    if ia is None or ib is None or ic is None:
        return
    out.append((ia, ib, ic))


def _maybe_add_quad(
    out: List[Tuple[int, int, int, int]], name_to_idx: Dict[str, int], a: str, b: str, c: str, d: str
) -> None:
    ia = name_to_idx.get(a)
    ib = name_to_idx.get(b)
    ic = name_to_idx.get(c)
    id_ = name_to_idx.get(d)
    if ia is None or ib is None or ic is None or id_ is None:
        return
    out.append((ia, ib, ic, id_))


def _build_topology(
    *,
    osc_type: str,
    residue_name: str,
    name_to_idx: Dict[str, int],
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int]],
]:
    """Return (bonds, angles, dihedrals, dipoles) in atom-index space.

    - dipoles are stored as (C_idx, O_idx, N_idx) where N_idx may be -1
      (we fill -1 later).
    """
    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    diheds: List[Tuple[int, int, int, int]] = []
    dipoles: List[Tuple[int, int, int]] = []

    if osc_type == "backbone":
        # Bonds
        _maybe_add_pair(bonds, name_to_idx, "N_prev", "CA_prev")
        _maybe_add_pair(bonds, name_to_idx, "CA_prev", "C_prev")
        _maybe_add_pair(bonds, name_to_idx, "C_prev", "O_prev")
        _maybe_add_pair(bonds, name_to_idx, "C_prev", "N_curr")
        _maybe_add_pair(bonds, name_to_idx, "N_curr", "H_curr")
        _maybe_add_pair(bonds, name_to_idx, "N_curr", "CA_curr")

        # Angles (a few informative ones)
        _maybe_add_triple(angles, name_to_idx, "N_prev", "CA_prev", "C_prev")
        _maybe_add_triple(angles, name_to_idx, "CA_prev", "C_prev", "O_prev")
        _maybe_add_triple(angles, name_to_idx, "CA_prev", "C_prev", "N_curr")
        _maybe_add_triple(angles, name_to_idx, "O_prev", "C_prev", "N_curr")
        _maybe_add_triple(angles, name_to_idx, "C_prev", "N_curr", "CA_curr")
        _maybe_add_triple(angles, name_to_idx, "C_prev", "N_curr", "H_curr")

        # Dihedrals available within the peptide group
        _maybe_add_quad(diheds, name_to_idx, "N_prev", "CA_prev", "C_prev", "N_curr")   # psi(i)
        _maybe_add_quad(diheds, name_to_idx, "CA_prev", "C_prev", "N_curr", "CA_curr")  # omega(i)

        # Dipole: carbonyl + (optional) CN
        c = name_to_idx.get("C_prev")
        o = name_to_idx.get("O_prev")
        n = name_to_idx.get("N_curr", -1)
        if c is not None and o is not None:
            dipoles.append((c, o, int(n)))

    elif osc_type == "sidechain":
        # The sidechain residue_name is expected to be GLN-SC or ASN-SC.
        rn = residue_name.upper()
        if rn.startswith("GLN"):
            # GLN: CA-CB-CG-CD(=OE1)-NE2(-HE21/HE22)
            _maybe_add_pair(bonds, name_to_idx, "CA", "CB")
            _maybe_add_pair(bonds, name_to_idx, "CB", "CG")
            _maybe_add_pair(bonds, name_to_idx, "CG", "CD")
            _maybe_add_pair(bonds, name_to_idx, "CD", "OE1")
            _maybe_add_pair(bonds, name_to_idx, "CD", "NE2")
            _maybe_add_pair(bonds, name_to_idx, "NE2", "HE21")
            _maybe_add_pair(bonds, name_to_idx, "NE2", "HE22")

            _maybe_add_triple(angles, name_to_idx, "CA", "CB", "CG")
            _maybe_add_triple(angles, name_to_idx, "CB", "CG", "CD")
            _maybe_add_triple(angles, name_to_idx, "CG", "CD", "OE1")
            _maybe_add_triple(angles, name_to_idx, "CG", "CD", "NE2")
            _maybe_add_triple(angles, name_to_idx, "OE1", "CD", "NE2")
            _maybe_add_triple(angles, name_to_idx, "CD", "NE2", "HE21")
            _maybe_add_triple(angles, name_to_idx, "CD", "NE2", "HE22")

            _maybe_add_quad(diheds, name_to_idx, "CA", "CB", "CG", "CD")
            _maybe_add_quad(diheds, name_to_idx, "CB", "CG", "CD", "NE2")

            c = name_to_idx.get("CD")
            o = name_to_idx.get("OE1")
            n = name_to_idx.get("NE2", -1)
            if c is not None and o is not None:
                dipoles.append((c, o, int(n)))

        elif rn.startswith("ASN"):
            # ASN: CA-CB-CG(=OD1)-ND2(-HD21/HD22)
            _maybe_add_pair(bonds, name_to_idx, "CA", "CB")
            _maybe_add_pair(bonds, name_to_idx, "CB", "CG")
            _maybe_add_pair(bonds, name_to_idx, "CG", "OD1")
            _maybe_add_pair(bonds, name_to_idx, "CG", "ND2")
            _maybe_add_pair(bonds, name_to_idx, "ND2", "HD21")
            _maybe_add_pair(bonds, name_to_idx, "ND2", "HD22")

            _maybe_add_triple(angles, name_to_idx, "CA", "CB", "CG")
            _maybe_add_triple(angles, name_to_idx, "CB", "CG", "OD1")
            _maybe_add_triple(angles, name_to_idx, "CB", "CG", "ND2")
            _maybe_add_triple(angles, name_to_idx, "OD1", "CG", "ND2")
            _maybe_add_triple(angles, name_to_idx, "CG", "ND2", "HD21")
            _maybe_add_triple(angles, name_to_idx, "CG", "ND2", "HD22")

            _maybe_add_quad(diheds, name_to_idx, "CA", "CB", "CG", "ND2")

            c = name_to_idx.get("CG")
            o = name_to_idx.get("OD1")
            n = name_to_idx.get("ND2", -1)
            if c is not None and o is not None:
                dipoles.append((c, o, int(n)))

        else:
            # Unknown sidechain type. We still return empty topology and rely on
            # denoising loss only.
            pass

    return bonds, angles, diheds, dipoles


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------


def build_graph_from_oscillator(
    osc: Mapping[str, Any],
    *,
    vocab: GraphVocab,
    drop_zero_atoms: bool = True,
    max_sidechain_beads: int = 4,
    fully_connected_edges: bool = True,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Convert one oscillator dict into a training graph sample."""

    osc_type = str(osc.get("oscillator_type", ""))
    residue_name = str(osc.get("residue_name", "UNK"))

    if osc_type not in {"backbone", "sidechain"}:
        raise ValueError(f"Unknown oscillator_type: {osc_type}")

    # ----------------
    # Metadata (folder/frame/index)
    # ----------------
    meta_folder = osc.get("folder")
    meta_frame = osc.get("frame")
    meta_oscillator_index = osc.get("oscillator_index")

    # ----------------
    # Residue BB beads (anchors)
    # ----------------
    residue_keys: List[Tuple[int, str]] = []
    # For sidechain oscillators we keep a copy of the *true* BB bead position
    # (bb_prev) even if we later choose to anchor the residue-local frame at SC1.
    bb_prev_raw: Optional[torch.Tensor] = None

    if osc_type == "backbone":
        # residue i and i+1
        bb_curr = _as_float_tensor(osc["bb_curr"], device=device, dtype=dtype)
        bb_next = _as_float_tensor(osc["bb_next"], device=device, dtype=dtype)
        bb_pos = torch.stack([bb_curr, bb_next], dim=0)  # [2,3]

        k0 = _tuple_reskey(osc.get("bb_curr_key")) or _tuple_reskey(osc.get("residue_key"))
        k1 = _tuple_reskey(osc.get("bb_next_key"))
        if k0 is None or k1 is None:
            # We can still train without residue ids, but cannot write full-frame PDBs.
            # Keep placeholders.
            k0 = k0 or (-1, "UNK")
            k1 = k1 or (-1, "UNK")
        residue_keys = [k0, k1]
        Nres = 2
    else:
        bb_prev_raw = _as_float_tensor(osc["bb_prev"], device=device, dtype=dtype)
        bb_pos = bb_prev_raw.unsqueeze(0)  # [1,3] (may be overridden to SC1 anchor below)

        k0 = _tuple_reskey(osc.get("bb_prev_key")) or _tuple_reskey(osc.get("residue_key"))
        if k0 is None:
            k0 = (-1, "UNK")
        residue_keys = [k0]
        Nres = 1

    # Optional sidechain beads
    sc_beads: Mapping[str, Any] = osc.get("sc_beads", {}) or {}
    sc1_any = sc_beads.get("SC1")
    sc1_t: Optional[torch.Tensor] = None
    if sc1_any is not None:
        try:
            sc1_t = _as_float_tensor(sc1_any, device=device, dtype=dtype)
        except Exception:
            sc1_t = None

    # --- Choose a stable residue-local origin + frame ---
    #
    # Backbone oscillators:
    #   origin is BB bead(s) as usual.
    # Sidechain oscillators:
    #   If SC1 exists, use **SC1 as the local origin**. This keeps sidechain atom
    #   radii within the configured max_atom_radius (usually 6Å) and matches the
    #   physical intent: sidechain atoms are naturally local to SC1.
    #   We still include the BB bead as an input node, but we do not use it as
    #   the origin for sidechain atom local coordinates.
    if osc_type == "sidechain" and sc1_t is not None and torch.isfinite(sc1_t).all().item():
        # Anchor at SC1 for sidechains
        assert bb_prev_raw is not None
        bb_pos = sc1_t.unsqueeze(0)
        # For N==1, compute_residue_local_frames will use this reference direction
        # to build a deterministic frame.
        sc1_pos = bb_prev_raw.unsqueeze(0)
    else:
        # Anchor at BB (default)
        if sc1_t is None:
            sc1_pos = None
        else:
            if Nres == 2:
                nan = torch.full_like(sc1_t, float("nan"))
                sc1_pos = torch.stack([sc1_t, nan], dim=0)
            else:
                sc1_pos = sc1_t.unsqueeze(0)

    bb_frames = compute_residue_local_frames(bb_pos, sc1_pos=sc1_pos)  # [Nres,3,3]

    # -----------------
    # Atomistic targets
    # -----------------
    atoms: Mapping[str, Any] = osc.get("atoms", {}) or {}
    if len(atoms) == 0:
        raise ValueError("osc['atoms'] is empty")

    backbone_order = ["N_prev", "CA_prev", "C_prev", "O_prev", "N_curr", "H_curr", "CA_curr"]

    atom_names: List[str] = []
    atom_names_pdb: List[str] = []
    atom_pos_global: List[torch.Tensor] = []
    atom_res_local: List[int] = []
    atom_group: List[int] = []

    name_to_atom_index: Dict[str, int] = {}

    def _add_atom(aname: str, pos_any: Any, resid_local: int):
        nonlocal atom_names, atom_names_pdb, atom_pos_global, atom_res_local, atom_group, name_to_atom_index
        pos_np = np.asarray(pos_any, dtype=np.float32)
        if pos_np.shape != (3,):
            pos_np = pos_np.reshape(3)
        if drop_zero_atoms and _is_zero_pos(pos_np):
            return
        idx = len(atom_names)
        atom_names.append(aname)
        atom_names_pdb.append(canonical_pdb_atom_name(osc_type, aname))
        atom_pos_global.append(torch.as_tensor(pos_np, device=device, dtype=dtype))
        atom_res_local.append(int(resid_local))
        atom_group.append(vocab.atom_group_id(aname))
        name_to_atom_index[aname] = idx

    if osc_type == "backbone":
        for aname in backbone_order:
            if aname not in atoms:
                continue
            # residue slot is encoded in the key
            resid_local = 0 if aname.endswith("_prev") else 1
            _add_atom(aname, atoms[aname], resid_local)
    else:
        for aname in sorted(atoms.keys()):
            _add_atom(aname, atoms[aname], resid_local=0)

    if len(atom_names) == 0:
        raise RuntimeError(
            "After filtering missing/zero atoms, this oscillator has Na=0 atoms. "
            "Set drop_zero_atoms=False or inspect the oscillator entry."
        )

    atom_pos0 = torch.stack(atom_pos_global, dim=0)  # [Na,3]
    atom_res_local_t = torch.tensor(atom_res_local, device=device, dtype=torch.long)
    atom_group_t = torch.tensor(atom_group, device=device, dtype=torch.long)

    # Convert global → residue-local.
    # - Backbone oscillators: origin is the BB bead for each residue slot.
    # - Sidechain oscillators: origin is SC1 when available (preferred), else BB.
    origin = bb_pos[atom_res_local_t]
    R = bb_frames[atom_res_local_t]
    x0_local = global_to_local(atom_pos0, origin, R)  # [Na,3]

    # ----------------
    # Bead nodes (CG)
    # ----------------
    bead_names: List[str] = []
    bead_pos: List[torch.Tensor] = []
    bead_res_local: List[int] = []

    def _add_bead(bname: str, pos_any: Any, resid_local: int):
        bead_names.append(str(bname))
        bead_pos.append(_as_float_tensor(pos_any, device=device, dtype=dtype))
        bead_res_local.append(int(resid_local))

    if osc_type == "backbone":
        _add_bead("BB", bb_pos[0], 0)
        _add_bead("BB", bb_pos[1], 1)
        for i in range(1, max_sidechain_beads + 1):
            key = f"SC{i}"
            if key in sc_beads:
                _add_bead(key, sc_beads[key], 0)
    else:
        # For sidechain oscillators, bb_pos[0] may be anchored to SC1. Always add
        # the actual BB bead position as the "BB" input node.
        _add_bead("BB", bb_prev_raw if bb_prev_raw is not None else bb_pos[0], 0)
        for i in range(1, max_sidechain_beads + 1):
            key = f"SC{i}"
            if key in sc_beads:
                _add_bead(key, sc_beads[key], 0)

    bead_pos_t = torch.stack(bead_pos, dim=0)
    bead_res_local_t = torch.tensor(bead_res_local, device=device, dtype=torch.long)
    Nb = int(bead_pos_t.shape[0])

    # ----------------
    # Nodes list
    # ----------------
    Na = int(x0_local.shape[0])
    num_nodes = Nb + Na

    bead_node_indices = torch.arange(0, Nb, device=device, dtype=torch.long)
    atom_node_indices = torch.arange(Nb, Nb + Na, device=device, dtype=torch.long)

    node_type = torch.empty((num_nodes,), device=device, dtype=torch.long)
    node_type[:Nb] = 0  # bead
    node_type[Nb:] = 1  # atom

    node_name = torch.empty((num_nodes,), device=device, dtype=torch.long)
    for i, nm in enumerate(bead_names):
        node_name[i] = vocab.name_id(nm)
    for i, nm in enumerate(atom_names):
        node_name[Nb + i] = vocab.name_id(nm)

    node_resname = torch.full((num_nodes,), vocab.resname_id(residue_name), device=device, dtype=torch.long)

    node_res = torch.empty((num_nodes,), device=device, dtype=torch.long)
    node_res[:Nb] = bead_res_local_t
    node_res[Nb:] = atom_res_local_t

    # In oscillator-local graphs, residue indices already start at 0.
    node_res_in_frame = node_res.clone()

    # ----------------
    # Edge construction
    # ----------------
    if fully_connected_edges:
        # All directed edges excluding self loops.
        src, dst = torch.meshgrid(
            torch.arange(num_nodes, device=device),
            torch.arange(num_nodes, device=device),
            indexing="ij",
        )
        mask = src != dst
        src = src[mask].reshape(-1)
        dst = dst[mask].reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        # Minimal connectivity: connect everything to beads (bidirectional).
        src_list: List[int] = []
        dst_list: List[int] = []
        for i in range(num_nodes):
            for j in range(Nb):
                if i == j:
                    continue
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
        edge_index = torch.tensor([src_list, dst_list], device=device, dtype=torch.long)

    # Edge types used by BackmapGNN (small integer categories)
    src = edge_index[0]
    dst = edge_index[1]
    src_is_atom = src >= Nb
    dst_is_atom = dst >= Nb

    edge_type = torch.empty((edge_index.shape[1],), device=device, dtype=torch.long)
    edge_type[(~src_is_atom) & (~dst_is_atom)] = 0  # bead-bead
    edge_type[(src_is_atom) & (dst_is_atom)] = 4    # atom-atom
    edge_type[(src_is_atom) ^ (dst_is_atom)] = 3    # atom-any

    # ----------------
    # Topology indices (atom-index space 0..Na-1)
    # ----------------
    bonds, angles, diheds, dipoles = _build_topology(
        osc_type=osc_type,
        residue_name=residue_name,
        name_to_idx=name_to_atom_index,
    )

    if len(bonds) > 0:
        bond_index = torch.tensor(bonds, device=device, dtype=torch.long).t().contiguous()
    else:
        bond_index = torch.zeros((2, 0), device=device, dtype=torch.long)

    if len(angles) > 0:
        angle_index = torch.tensor(angles, device=device, dtype=torch.long).t().contiguous()
    else:
        angle_index = torch.zeros((3, 0), device=device, dtype=torch.long)

    if len(diheds) > 0:
        dihedral_index = torch.tensor(diheds, device=device, dtype=torch.long).t().contiguous()
    else:
        dihedral_index = torch.zeros((4, 0), device=device, dtype=torch.long)

    if len(dipoles) > 0:
        dipole_index = torch.tensor(dipoles, device=device, dtype=torch.long).t().contiguous()
    else:
        dipole_index = torch.zeros((3, 0), device=device, dtype=torch.long)

    # ----------------
    # Package output
    # ----------------
    return {
        # Core tensors
        "x0_local": x0_local,
        "atom_pos0": atom_pos0,
        "atom_res": atom_res_local_t,  # residue slot 0..Nres-1
        "atom_group": atom_group_t,
        "bb_pos": bb_pos,
        "bb_frames": bb_frames,
        "bead_pos": bead_pos_t,
        "bead_node_indices": bead_node_indices,
        "atom_node_indices": atom_node_indices,
        "node_type": node_type,
        "node_name": node_name,
        "node_resname": node_resname,
        "node_res": node_res,
        "node_res_in_frame": node_res_in_frame,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_nodes": int(num_nodes),
        # Topology
        "bond_index": bond_index,
        "angle_index": angle_index,
        "dihedral_index": dihedral_index,
        "dipole_index": dipole_index,
        # Metadata
        "oscillator_type": osc_type,
        "residue_name": residue_name,
        "meta_folder": meta_folder,
        "meta_frame": meta_frame,
        "meta_oscillator_index": meta_oscillator_index,
        "meta_residue_keys": residue_keys,
        "meta_atom_names": atom_names,
        "meta_atom_names_pdb": atom_names_pdb,
    }

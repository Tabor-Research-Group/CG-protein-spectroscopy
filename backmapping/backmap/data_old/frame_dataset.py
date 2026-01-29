from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from backmap.config import DataConfig
from backmap.data.pkl_loader import (
    load_baskets,
    group_oscillators_by_frame,
    build_per_residue_atomistic,
    build_per_residue_cg,
)
from backmap.data.vocab import Vocab, build_vocab_from_baskets
from backmap.geometry.frames import compute_residue_local_frames
from backmap.geometry.spherical import cartesian_to_spherical_sincos
from backmap.physics.topology import build_topology_indices, TopologyIndices


@dataclass(frozen=True)
class FrameMeta:
    folder: str
    frame: int


def _torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {dtype_str}")


def _as_float_tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def _as_long_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long)


def _build_candidate_edges(
    bb_pos: np.ndarray,                 # [Nres,3]
    bead_node_res_index: np.ndarray,    # [Nb]
    bead_node_is_bb: np.ndarray,        # [Nb]
    atom_res_index: np.ndarray,         # [Na]
    atom_bb_anchor_node: np.ndarray,    # [Na] indices into bead nodes for BB anchor
    max_atom_radius: float,
    bead_edge_cutoff: float,
    atom_edge_cutoff: float,
    bond_pairs: np.ndarray,             # [Nbonds,2] atom indices
) -> Tuple[np.ndarray, np.ndarray]:
    """Build candidate directed edges for one frame.

    Node order within frame:
      - [0..Nb-1] are CG beads
      - [Nb..Nb+Na-1] are atoms

    Returns
    -------
    edge_index: [2,E] int64
    edge_type:  [E] int64
    """
    Nres = bb_pos.shape[0]
    Nb = bead_node_res_index.shape[0]
    Na = atom_res_index.shape[0]

    # Map residue index -> BB bead node index (within bead nodes)
    bb_node_by_res = -np.ones((Nres,), dtype=np.int64)
    for bn in range(Nb):
        if bead_node_is_bb[bn]:
            r = int(bead_node_res_index[bn])
            bb_node_by_res[r] = bn
    if np.any(bb_node_by_res < 0):
        # Should not happen; but if it does, we cannot build neighbor candidates robustly.
        raise ValueError("Missing BB bead node for at least one residue")

    # Precompute residue-residue distances (BB-BB) for candidate filtering
    # Nres is typically <= a few hundred; full matrix is OK.
    dmat = np.sqrt(((bb_pos[:, None, :] - bb_pos[None, :, :]) ** 2).sum(axis=-1))  # [Nres,Nres]

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edges_type: List[int] = []

    def add_edge(u: int, v: int, t: int) -> None:
        edges_src.append(int(u))
        edges_dst.append(int(v))
        edges_type.append(int(t))

    # 0) BB-BB conditioning edges within bead_edge_cutoff, plus chain edges
    for i in range(Nres):
        for j in range(i + 1, Nres):
            if dmat[i, j] <= bead_edge_cutoff:
                u = int(bb_node_by_res[i])
                v = int(bb_node_by_res[j])
                add_edge(u, v, 0)
                add_edge(v, u, 0)
    for i in range(Nres - 1):
        u = int(bb_node_by_res[i])
        v = int(bb_node_by_res[i + 1])
        add_edge(u, v, 0)
        add_edge(v, u, 0)

    # 1) BB-SC edges inside each residue (always)
    # 2) SC-SC edges inside each residue (optional; always)
    sc_nodes_by_res: Dict[int, List[int]] = {}
    for bn in range(Nb):
        r = int(bead_node_res_index[bn])
        if not bead_node_is_bb[bn]:
            sc_nodes_by_res.setdefault(r, []).append(bn)
    for r, sc_nodes in sc_nodes_by_res.items():
        bb = int(bb_node_by_res[r])
        for sc in sc_nodes:
            add_edge(bb, sc, 1)
            add_edge(sc, bb, 1)
        # SC-SC fully connected within residue
        if len(sc_nodes) >= 2:
            for i in range(len(sc_nodes)):
                for j in range(i + 1, len(sc_nodes)):
                    u = sc_nodes[i]
                    v = sc_nodes[j]
                    add_edge(u, v, 2)
                    add_edge(v, u, 2)

    # 3) Atom-bead edges (candidates based on BB anchor distance)
    res_bead_margin = atom_edge_cutoff + max_atom_radius
    for a in range(Na):
        ra = int(atom_res_index[a])
        # residues whose BB is within margin of anchor BB
        close_res = np.where(dmat[ra] <= res_bead_margin)[0]
        atom_node = Nb + a
        for r in close_res:
            # connect to BB node of r and all SC nodes of r
            bb = int(bb_node_by_res[int(r)])
            add_edge(atom_node, bb, 3)
            add_edge(bb, atom_node, 3)
            for sc in sc_nodes_by_res.get(int(r), []):
                add_edge(atom_node, sc, 3)
                add_edge(sc, atom_node, 3)

    # 4) Atom-atom edges (candidates based on anchor BB distance)
    res_res_margin = atom_edge_cutoff + 2.0 * max_atom_radius
    atoms_by_res: Dict[int, List[int]] = {}
    for a in range(Na):
        atoms_by_res.setdefault(int(atom_res_index[a]), []).append(a)

    for i in range(Nres):
        ai = atoms_by_res.get(i, [])
        if not ai:
            continue
        # within same residue: fully connected (excluding self)
        for u_i in ai:
            for v_i in ai:
                if u_i != v_i:
                    add_edge(Nb + u_i, Nb + v_i, 4)

        # between residues
        close_res = np.where(dmat[i] <= res_res_margin)[0]
        for j in close_res:
            if j <= i:
                continue
            aj = atoms_by_res.get(int(j), [])
            if not aj:
                continue
            for u_i in ai:
                for v_j in aj:
                    add_edge(Nb + u_i, Nb + v_j, 4)
                    add_edge(Nb + v_j, Nb + u_i, 4)

    # 5) Covalent bond edges (always, no cutoff)
    if bond_pairs.shape[0] > 0:
        for a_idx, b_idx in bond_pairs.tolist():
            u = Nb + int(a_idx)
            v = Nb + int(b_idx)
            add_edge(u, v, 5)
            add_edge(v, u, 5)

    edge_index = np.stack([np.asarray(edges_src, dtype=np.int64), np.asarray(edges_dst, dtype=np.int64)], axis=0)
    edge_type = np.asarray(edges_type, dtype=np.int64)
    return edge_index, edge_type


def _frame_cache_path(cache_dir: Path, folder: str, frame: int) -> Path:
    safe_folder = folder.replace(os.sep, "_")
    return cache_dir / f"{safe_folder}__frame{int(frame)}.pt"


def preprocess_to_cache(cfg: DataConfig) -> Tuple[List[FrameMeta], Vocab]:
    """Preprocess the pickle into per-frame cached tensors."""
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    vocab_path = cache_dir / "vocab.json"

    if manifest_path.exists() and vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        manifest = json.loads(manifest_path.read_text())
        frames = [FrameMeta(folder=str(m["folder"]), frame=int(m["frame"])) for m in manifest["frames"]]
        return frames, vocab

    if not cfg.pickle_path or not Path(cfg.pickle_path).exists():
        raise FileNotFoundError(f"Pickle not found: {cfg.pickle_path}")

    baskets = load_baskets(cfg.pickle_path)
    vocab = build_vocab_from_baskets(baskets)
    vocab.save(vocab_path)

    groups = group_oscillators_by_frame(baskets)

    dtype = _torch_dtype(cfg.dtype)

    frames: List[FrameMeta] = []

    for (folder, frame), group in groups.items():
        backbone = group["backbone"]
        sidechain = group["sidechain"]
        if len(backbone) == 0:
            # no backbone -> skip (cannot define local frames reliably)
            continue

        atoms_by_res = build_per_residue_atomistic(backbone, sidechain)
        cg_by_res = build_per_residue_cg(backbone, sidechain)

        # Build ordered residue list based on BB presence
        res_keys = sorted([rk for rk, beads in cg_by_res.items() if "BB" in beads], key=lambda x: x[0])
        if len(res_keys) < 2:
            continue

        residue_ids = np.asarray([rk[0] for rk in res_keys], dtype=np.int64)
        residue_names = [str(rk[1]).upper() for rk in res_keys]

        bb_pos = np.stack([cg_by_res[rk]["BB"] for rk in res_keys], axis=0).astype(np.float32)  # [Nres,3]

        # Optional SC1 for frame stabilization
        sc1_pos = np.full_like(bb_pos, np.nan, dtype=np.float32)
        for i, rk in enumerate(res_keys):
            if "SC1" in cg_by_res.get(rk, {}):
                sc1_pos[i] = cg_by_res[rk]["SC1"].astype(np.float32)

        # Local frames (torch -> numpy)
        bb_pos_t = torch.as_tensor(bb_pos, dtype=dtype)
        sc1_pos_t = torch.as_tensor(sc1_pos, dtype=dtype)
        R = compute_residue_local_frames(bb_pos_t, sc1_pos_t, eps=1e-8).cpu().numpy().astype(np.float32)  # [Nres,3,3]

        # Bead nodes (BB + sidechain beads)
        bead_pos_list: List[np.ndarray] = []
        bead_name_list: List[str] = []
        bead_res_index_list: List[int] = []
        bead_is_bb_list: List[int] = []

        bb_node_index_by_res = {}

        for i, rk in enumerate(res_keys):
            bead_pos_list.append(cg_by_res[rk]["BB"].astype(np.float32))
            bead_name_list.append("BB")
            bead_res_index_list.append(i)
            bead_is_bb_list.append(1)
            bb_node_index_by_res[i] = len(bead_pos_list) - 1

        # Add SC beads for each residue (sorted by bead name for determinism)
        for i, rk in enumerate(res_keys):
            beads = cg_by_res.get(rk, {})
            for bead_name in sorted([k for k in beads.keys() if k != "BB"]):
                bead_pos_list.append(beads[bead_name].astype(np.float32))
                bead_name_list.append(str(bead_name))
                bead_res_index_list.append(i)
                bead_is_bb_list.append(0)

        bead_pos = np.stack(bead_pos_list, axis=0).astype(np.float32)  # [Nb,3]
        bead_res_index = np.asarray(bead_res_index_list, dtype=np.int64)
        bead_is_bb = np.asarray(bead_is_bb_list, dtype=np.bool_)

        # Atom nodes
        atom_pos_list: List[np.ndarray] = []
        atom_name_list: List[str] = []
        atom_res_index_list: List[int] = []
        atom_group_list: List[int] = []
        atom_bb_anchor_list: List[int] = []

        backbone_names = {"N", "CA", "C", "O", "H"}

        atom_index_by_residue: Dict[Tuple[int, str], Dict[str, int]] = {}
        for i, rk in enumerate(res_keys):
            # Standardize residue key
            resid, resname = int(rk[0]), str(rk[1]).upper()
            atoms = atoms_by_res.get((resid, resname), {})
            if not atoms:
                continue
            for atom_name, coord in sorted(atoms.items()):
                atom_pos_list.append(coord.astype(np.float32))
                atom_name_list.append(str(atom_name))
                atom_res_index_list.append(i)
                atom_group_list.append(0 if str(atom_name) in backbone_names else 1)
                atom_bb_anchor_list.append(int(bb_node_index_by_res[i]))
                # mapping for topology indices
                atom_index_by_residue.setdefault((resid, resname), {})[str(atom_name)] = len(atom_pos_list) - 1

        if len(atom_pos_list) == 0:
            continue

        atom_pos = np.stack(atom_pos_list, axis=0).astype(np.float32)  # [Na,3]
        atom_res_index = np.asarray(atom_res_index_list, dtype=np.int64)  # [Na]
        atom_group = np.asarray(atom_group_list, dtype=np.int64)  # [Na]
        atom_bb_anchor = np.asarray(atom_bb_anchor_list, dtype=np.int64)  # [Na], bead node index

        # Ground truth local vectors relative to BB (and in BB local frame)
        x0_local = np.zeros_like(atom_pos, dtype=np.float32)
        max_r = float(cfg.max_atom_radius)
        for a in range(atom_pos.shape[0]):
            r = int(atom_res_index[a])
            origin = bb_pos[r]
            v = atom_pos[a] - origin  # global
            vloc = R[r].T @ v         # local
            # clamp radius
            n = np.sqrt((vloc * vloc).sum() + 1e-12)
            if n > max_r:
                vloc = vloc * (max_r / n)
            x0_local[a] = vloc

        x0_sph = cartesian_to_spherical_sincos(torch.as_tensor(x0_local, dtype=dtype)).cpu().numpy().astype(np.float32)

        # Topology indices (bonds/angles/rama/dipole/coulomb)
        topo = build_topology_indices(residue_ids.tolist(), residue_names, atom_index_by_residue)

        # Candidate edges (directed)
        edge_index, edge_type = _build_candidate_edges(
            bb_pos=bb_pos,
            bead_node_res_index=bead_res_index,
            bead_node_is_bb=bead_is_bb,
            atom_res_index=atom_res_index,
            atom_bb_anchor_node=atom_bb_anchor,
            max_atom_radius=float(cfg.max_atom_radius),
            bead_edge_cutoff=float(cfg.bead_edge_cutoff),
            atom_edge_cutoff=float(cfg.atom_edge_cutoff),
            bond_pairs=topo.bond_pairs,
        )

        # Node attributes (for embeddings)
        Nb = bead_pos.shape[0]
        Na = atom_pos.shape[0]
        Nn = Nb + Na

        node_type = np.zeros((Nn,), dtype=np.int64)
        node_name = np.zeros((Nn,), dtype=np.int64)
        node_resname = np.zeros((Nn,), dtype=np.int64)
        node_res_index = np.zeros((Nn,), dtype=np.int64)

        # Beads
        for i in range(Nb):
            bead_nm = bead_name_list[i]
            node_type[i] = vocab.node_type_to_id["BB"] if bead_nm == "BB" else vocab.node_type_to_id["SC"]
            node_name[i] = vocab.name_to_id.get(bead_nm, 0)
            r = int(bead_res_index[i])
            node_res_index[i] = r
            node_resname[i] = vocab.resname_to_id.get(residue_names[r], 0)

        # Atoms
        for a in range(Na):
            idx = Nb + a
            node_type[idx] = vocab.node_type_to_id["ATOM"]
            node_name[idx] = vocab.name_to_id.get(atom_name_list[a], 0)
            r = int(atom_res_index[a])
            node_res_index[idx] = r
            node_resname[idx] = vocab.resname_to_id.get(residue_names[r], 0)

        # Frame cache
        payload: Dict[str, Any] = {
            "folder": folder,
            "frame": int(frame),
            "residue_ids": torch.as_tensor(residue_ids, dtype=torch.long),
            "residue_names": residue_names,

            "bb_pos": _as_float_tensor(bb_pos, dtype=dtype),
            "bb_frames": _as_float_tensor(R, dtype=dtype),

            "bead_pos": _as_float_tensor(bead_pos, dtype=dtype),
            "bead_res_index": _as_long_tensor(bead_res_index),
            "bead_is_bb": torch.as_tensor(bead_is_bb, dtype=torch.bool),

            "atom_pos": _as_float_tensor(atom_pos, dtype=dtype),
            "atom_names": atom_name_list,
            "atom_res_index": _as_long_tensor(atom_res_index),
            "atom_group": _as_long_tensor(atom_group),
            "atom_bb_anchor": _as_long_tensor(atom_bb_anchor),

            "x0_local": _as_float_tensor(x0_local, dtype=dtype),
            "x0_sph": _as_float_tensor(x0_sph, dtype=dtype),

            "node_type": _as_long_tensor(node_type),
            "node_name": _as_long_tensor(node_name),
            "node_resname": _as_long_tensor(node_resname),
            "node_res_index": _as_long_tensor(node_res_index),

            "edge_index": _as_long_tensor(edge_index),
            "edge_type": _as_long_tensor(edge_type),

            # Topology indices
            "bond_pairs": _as_long_tensor(topo.bond_pairs),
            "angle_triples": _as_long_tensor(topo.angle_triples),
            "charged_idx": _as_long_tensor(topo.charged_atom_indices),
            "charged_q": torch.as_tensor(topo.charged_atom_charges, dtype=dtype),
            "charged_res_index": _as_long_tensor(topo.charged_atom_res_index),
            "dip_C": _as_long_tensor(topo.dipole_C_idx),
            "dip_O": _as_long_tensor(topo.dipole_O_idx),
            "dip_Nn": _as_long_tensor(topo.dipole_Nnext_idx),
            "dip_res_i": _as_long_tensor(topo.dipole_res_i),
            "phi_idx": _as_long_tensor(topo.phi_indices),
            "psi_idx": _as_long_tensor(topo.psi_indices),
        }

        cache_path = _frame_cache_path(cache_dir, folder, int(frame))
        torch.save(payload, cache_path)

        frames.append(FrameMeta(folder=folder, frame=int(frame)))

    manifest = {"frames": [{"folder": f.folder, "frame": f.frame} for f in frames]}
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return frames, vocab


class OscillatorFrameDataset(Dataset):
    """Torch dataset returning preprocessed per-frame graphs."""

    def __init__(
        self,
        cfg: DataConfig,
        frames: Sequence[FrameMeta],
        vocab: Vocab,
    ):
        self.cfg = cfg
        self.vocab = vocab
        self.frames = list(frames)
        self.cache_dir = Path(cfg.cache_dir)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.frames[idx]
        cache_path = _frame_cache_path(self.cache_dir, meta.folder, meta.frame)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached frame file: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        return payload

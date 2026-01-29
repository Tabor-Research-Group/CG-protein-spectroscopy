from __future__ import annotations

"""Collation utilities for oscillator-local graphs.

The model operates on graphs containing both **bead nodes** and **atom nodes**.
Each dataset sample is a small graph (typically ~5-20 nodes). A DataLoader
batch concatenates these small graphs into one larger disjoint graph.

This is historically a frequent source of subtle bugs:
- edge_index offset mistakes
- mixing of atom index space (0..Na-1 per sample) vs node index space (0..Nn-1)
- forgetting to offset residue indices used to index bb_pos / bb_frames

To prevent silent failures (e.g. producing Na_total=0 and therefore zero losses),
this collate function performs strict sanity checks.

Returned batch keys
-------------------

Tensor fields (concatenated):
  x0_local       [Na,3]
  atom_pos0      [Na,3]   (ground truth global positions)
  atom_res       [Na]     (global residue index into bb_pos/bb_frames)
  atom_group     [Na]

  bb_pos         [Nres,3]
  bb_frames      [Nres,3,3]

  bead_pos       [Nb,3]
  bead_node_indices [Nb]
  atom_node_indices [Na]

  node_type/node_name/node_resname/node_res/node_res_in_frame
  edge_index     [2,E]
  edge_type      [E]

  bond_index     [2,Nbonds]
  angle_index    [3,Nangles]
  dihedral_index [4,Ndihed]
  dipole_index   [3,Ndip]

Batch bookkeeping:
  node_batch     [Nn]  (node -> sample id)
  atom_batch     [Na]  (atom -> sample id)
  node_ptr       [B+1]
  atom_ptr       [B+1]
  res_ptr        [B+1]

Metadata (python lists):
  meta_folder, meta_frame, meta_oscillator_index
  meta_residue_keys   (list[list[tuple]])
  meta_atom_names     (list[list[str]])
  meta_atom_names_pdb (list[list[str]])
  meta_residue_keys_flat (list[tuple]) length Nres (indexes match bb_pos rows)

"""

from typing import Any, Dict, List

import torch


def collate_graph_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Empty batch")

    # All samples are expected to already be on the intended device (dataset can move).
    device = samples[0]["x0_local"].device
    dtype = samples[0]["x0_local"].dtype

    # Running offsets
    node_off = 0
    res_off = 0
    atom_off = 0

    # Accumulators
    x0_local = []
    atom_pos0 = []
    atom_res = []
    atom_group = []

    bb_pos = []
    bb_frames = []

    bead_pos = []
    bead_node_indices = []
    atom_node_indices = []

    node_type = []
    node_name = []
    node_resname = []
    node_res = []
    node_res_in_frame = []

    edge_index = []
    edge_type = []

    bond_index = []
    angle_index = []
    dihedral_index = []
    dipole_index = []

    # Batch bookkeeping
    node_batch = []
    atom_batch = []
    node_ptr = [0]
    atom_ptr = [0]
    res_ptr = [0]

    # Metadata (python objects)
    meta_folder: List[str] = []
    meta_frame: List[int] = []
    meta_oscillator_index: List[int] = []
    meta_residue_keys: List[List[tuple]] = []
    meta_atom_names: List[List[str]] = []
    meta_atom_names_pdb: List[List[str]] = []
    meta_residue_keys_flat: List[tuple] = []

    for b, s in enumerate(samples):
        # --- counts ---
        Nn = int(s["num_nodes"])
        Na = int(s["x0_local"].shape[0])
        Nr = int(s["bb_pos"].shape[0])

        if Nn <= 0:
            raise RuntimeError("Sample has num_nodes<=0")
        if Na <= 0:
            raise RuntimeError("Sample has Na<=0 atoms; this would create zero loss")
        if Nr <= 0:
            raise RuntimeError("Sample has Nr<=0 residues")

        # --- atoms ---
        x0_local.append(s["x0_local"])
        atom_pos0.append(s["atom_pos0"])
        atom_res.append(s["atom_res"].to(dtype=torch.long) + res_off)
        atom_group.append(s["atom_group"].to(dtype=torch.long))

        # --- residues ---
        bb_pos.append(s["bb_pos"])
        bb_frames.append(s["bb_frames"])
        # Keep a global mapping from residue row -> residue_key for later frame-level PDB writing.
        rk = s.get("meta_residue_keys")
        if rk is None:
            rk = [(0, "UNK")] * Nr
        if len(rk) != Nr:
            raise RuntimeError(f"meta_residue_keys length {len(rk)} != Nr {Nr}")
        meta_residue_keys_flat.extend(list(rk))

        # --- beads / atom node indices ---
        bead_pos.append(s["bead_pos"])
        bead_node_indices.append(s["bead_node_indices"].to(dtype=torch.long) + node_off)
        atom_node_indices.append(s["atom_node_indices"].to(dtype=torch.long) + node_off)

        # --- node attrs ---
        node_type.append(s["node_type"].to(dtype=torch.long))
        node_name.append(s["node_name"].to(dtype=torch.long))
        node_resname.append(s["node_resname"].to(dtype=torch.long))
        node_res.append(s["node_res"].to(dtype=torch.long) + res_off)
        # Important: node_res_in_frame is used only for positional embedding.
        # It should remain *local per sample* (0..Nr-1).
        node_res_in_frame.append(s["node_res_in_frame"].to(dtype=torch.long))

        # --- edges ---
        edge_index.append(s["edge_index"].to(dtype=torch.long) + node_off)
        edge_type.append(s["edge_type"].to(dtype=torch.long))

        # --- topology indices in *atom index space* ---
        # These are optional / may be empty.
        if "bond_index" in s:
            bond_index.append(s["bond_index"].to(dtype=torch.long) + atom_off)
        if "angle_index" in s:
            angle_index.append(s["angle_index"].to(dtype=torch.long) + atom_off)
        if "dihedral_index" in s:
            dihedral_index.append(s["dihedral_index"].to(dtype=torch.long) + atom_off)
        if "dipole_index" in s:
            dipole_index.append(s["dipole_index"].to(dtype=torch.long) + atom_off)

        # --- bookkeeping ---
        node_batch.append(torch.full((Nn,), b, device=device, dtype=torch.long))
        atom_batch.append(torch.full((Na,), b, device=device, dtype=torch.long))

        node_off += Nn
        res_off += Nr
        atom_off += Na

        node_ptr.append(node_off)
        atom_ptr.append(atom_off)
        res_ptr.append(res_off)

        # --- metadata ---
        meta_folder.append(str(s.get("meta_folder", "")))
        meta_frame.append(int(s.get("meta_frame", -1)))
        meta_oscillator_index.append(int(s.get("meta_oscillator_index", -1)))
        meta_residue_keys.append(list(s.get("meta_residue_keys", [])))
        meta_atom_names.append(list(s.get("meta_atom_names", [])))
        meta_atom_names_pdb.append(list(s.get("meta_atom_names_pdb", [])))

    # Concatenate
    batch: Dict[str, Any] = {
        "x0_local": torch.cat(x0_local, dim=0).to(device=device, dtype=dtype),
        "atom_pos0": torch.cat(atom_pos0, dim=0).to(device=device, dtype=dtype),
        "atom_res": torch.cat(atom_res, dim=0).to(device=device, dtype=torch.long),
        "atom_group": torch.cat(atom_group, dim=0).to(device=device, dtype=torch.long),
        "bb_pos": torch.cat(bb_pos, dim=0).to(device=device, dtype=dtype),
        "bb_frames": torch.cat(bb_frames, dim=0).to(device=device, dtype=dtype),
        "bead_pos": torch.cat(bead_pos, dim=0).to(device=device, dtype=dtype),
        "bead_node_indices": torch.cat(bead_node_indices, dim=0).to(device=device, dtype=torch.long),
        "atom_node_indices": torch.cat(atom_node_indices, dim=0).to(device=device, dtype=torch.long),
        "node_type": torch.cat(node_type, dim=0).to(device=device, dtype=torch.long),
        "node_name": torch.cat(node_name, dim=0).to(device=device, dtype=torch.long),
        "node_resname": torch.cat(node_resname, dim=0).to(device=device, dtype=torch.long),
        "node_res": torch.cat(node_res, dim=0).to(device=device, dtype=torch.long),
        "node_res_in_frame": torch.cat(node_res_in_frame, dim=0).to(device=device, dtype=torch.long),
        "edge_index": torch.cat(edge_index, dim=1).to(device=device, dtype=torch.long),
        "edge_type": torch.cat(edge_type, dim=0).to(device=device, dtype=torch.long),
        "num_nodes": int(node_off),
        "node_batch": torch.cat(node_batch, dim=0).to(device=device, dtype=torch.long),
        "atom_batch": torch.cat(atom_batch, dim=0).to(device=device, dtype=torch.long),
        "node_ptr": torch.tensor(node_ptr, device=device, dtype=torch.long),
        "atom_ptr": torch.tensor(atom_ptr, device=device, dtype=torch.long),
        "res_ptr": torch.tensor(res_ptr, device=device, dtype=torch.long),
        # Python metadata
        "meta_folder": meta_folder,
        "meta_frame": meta_frame,
        "meta_oscillator_index": meta_oscillator_index,
        "meta_residue_keys": meta_residue_keys,
        "meta_atom_names": meta_atom_names,
        "meta_atom_names_pdb": meta_atom_names_pdb,
        "meta_residue_keys_flat": meta_residue_keys_flat,
        "batch_size": len(samples),
    }

    # Topology tensors (empty if none)
    batch["bond_index"] = (
        torch.cat(bond_index, dim=1).to(device=device, dtype=torch.long)
        if bond_index
        else torch.zeros((2, 0), device=device, dtype=torch.long)
    )
    batch["angle_index"] = (
        torch.cat(angle_index, dim=1).to(device=device, dtype=torch.long)
        if angle_index
        else torch.zeros((3, 0), device=device, dtype=torch.long)
    )
    batch["dihedral_index"] = (
        torch.cat(dihedral_index, dim=1).to(device=device, dtype=torch.long)
        if dihedral_index
        else torch.zeros((4, 0), device=device, dtype=torch.long)
    )
    batch["dipole_index"] = (
        torch.cat(dipole_index, dim=1).to(device=device, dtype=torch.long)
        if dipole_index
        else torch.zeros((3, 0), device=device, dtype=torch.long)
    )

    # Final sanity checks: Na must match atom_node_indices length.
    Na_total = int(batch["x0_local"].shape[0])
    if Na_total != int(batch["atom_node_indices"].numel()):
        raise RuntimeError(
            f"Inconsistent batch: Na_total={Na_total} but atom_node_indices has {int(batch['atom_node_indices'].numel())} entries"
        )
    if Na_total == 0:
        raise RuntimeError("Batch has Na_total=0 atoms; losses would be zero")

    return batch

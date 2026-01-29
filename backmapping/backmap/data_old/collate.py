from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_frames(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached frame payloads into a single batch.

    Outputs:
      - concatenated nodes/edges (graph is a disjoint union of per-frame graphs)
      - concatenated atoms and residues (for physics losses)
      - index maps that let you place bead/atom positions into the node-position array
    """
    if len(samples) == 0:
        raise ValueError("Empty batch")

    node_offset = 0
    atom_offset = 0
    res_offset = 0

    node_type = []
    node_name = []
    node_resname = []
    node_res_global = []
    node_res_frame = []

    edge_index = []
    edge_type = []

    bead_pos = []
    bead_is_bb = []
    bead_res_global = []
    bead_res_frame = []
    bead_node_indices = []

    atom_pos0 = []
    x0_local = []
    x0_sph = []
    atom_res_global = []
    atom_res_frame = []
    atom_group = []
    atom_bb_anchor_node = []
    atom_node_indices = []

    # topology
    bond_pairs = []
    angle_triples = []
    charged_idx = []
    charged_q = []
    charged_res_index = []
    dip_C = []
    dip_O = []
    dip_Nn = []
    dip_res_i = []
    phi_idx = []
    psi_idx = []

    bb_pos = []
    bb_frames = []

    frame_sizes = []
    residue_names_per_frame = []
    folder_per_frame = []
    frame_id_per_frame = []

    for s in samples:
        Nb = int(s["bead_pos"].shape[0])
        Na = int(s["atom_pos"].shape[0])
        Nn = int(s["node_type"].shape[0])
        Nr = int(s["bb_pos"].shape[0])

        if Nn != Nb + Na:
            raise ValueError("Node ordering assumption violated (beads then atoms)")

        # node arrays
        node_type.append(s["node_type"])
        node_name.append(s["node_name"])
        node_resname.append(s["node_resname"])
        node_res_frame.append(s["node_res_index"])
        node_res_global.append(s["node_res_index"] + res_offset)

        # edges
        ei = s["edge_index"].clone() + node_offset
        edge_index.append(ei)
        edge_type.append(s["edge_type"])

        # beads and their node indices
        bead_pos.append(s["bead_pos"])
        bead_is_bb.append(s["bead_is_bb"])
        bead_res_frame.append(s["bead_res_index"])
        bead_res_global.append(s["bead_res_index"] + res_offset)
        bead_node_indices.append(node_offset + torch.arange(Nb, dtype=torch.long))

        # atoms and their node indices
        atom_pos0.append(s["atom_pos"])
        x0_local.append(s["x0_local"])
        x0_sph.append(s["x0_sph"])
        atom_res_frame.append(s["atom_res_index"])
        atom_res_global.append(s["atom_res_index"] + res_offset)
        atom_group.append(s["atom_group"])
        atom_node_indices.append(node_offset + Nb + torch.arange(Na, dtype=torch.long))

        # anchor BB bead indices are bead-node indices within frame; convert to global node index
        atom_bb_anchor_node.append(s["atom_bb_anchor"].clone() + node_offset)

        # residue geometry
        bb_pos.append(s["bb_pos"])
        bb_frames.append(s["bb_frames"])

        # topology indices (atoms-only)
        if s["bond_pairs"].numel() > 0:
            bond_pairs.append(s["bond_pairs"] + atom_offset)
        if s["angle_triples"].numel() > 0:
            angle_triples.append(s["angle_triples"] + atom_offset)

        if s["charged_idx"].numel() > 0:
            charged_idx.append(s["charged_idx"] + atom_offset)
            charged_q.append(s["charged_q"])
            charged_res_index.append(s["charged_res_index"] + res_offset)

        if s["dip_C"].numel() > 0:
            dip_C.append(s["dip_C"] + atom_offset)
            dip_O.append(s["dip_O"] + atom_offset)
            dip_Nn.append(s["dip_Nn"] + atom_offset)
            dip_res_i.append(s["dip_res_i"] + res_offset)

        if s["phi_idx"].numel() > 0:
            phi_idx.append(s["phi_idx"] + atom_offset)
        if s["psi_idx"].numel() > 0:
            psi_idx.append(s["psi_idx"] + atom_offset)

        residue_names_per_frame.append(s["residue_names"])
        folder_per_frame.append(str(s["folder"]))
        frame_id_per_frame.append(int(s["frame"]))

        frame_sizes.append(
            {"Nb": Nb, "Na": Na, "Nn": Nn, "Nr": Nr, "node_offset": node_offset, "atom_offset": atom_offset, "res_offset": res_offset}
        )

        node_offset += Nn
        atom_offset += Na
        res_offset += Nr

    dtype_float = samples[0]["atom_pos"].dtype

    batch: Dict[str, Any] = {
        "node_type": torch.cat(node_type, dim=0),
        "node_name": torch.cat(node_name, dim=0),
        "node_resname": torch.cat(node_resname, dim=0),
        "node_res": torch.cat(node_res_global, dim=0),
        "node_res_in_frame": torch.cat(node_res_frame, dim=0),

        "edge_index": torch.cat(edge_index, dim=1),
        "edge_type": torch.cat(edge_type, dim=0),

        # beads
        "bead_pos": torch.cat(bead_pos, dim=0),
        "bead_is_bb": torch.cat(bead_is_bb, dim=0),
        "bead_res": torch.cat(bead_res_global, dim=0),
        "bead_res_in_frame": torch.cat(bead_res_frame, dim=0),
        "bead_node_indices": torch.cat(bead_node_indices, dim=0),

        # atoms
        "atom_pos0": torch.cat(atom_pos0, dim=0),
        "x0_local": torch.cat(x0_local, dim=0),
        "x0_sph": torch.cat(x0_sph, dim=0),
        "atom_res": torch.cat(atom_res_global, dim=0),
        "atom_res_in_frame": torch.cat(atom_res_frame, dim=0),
        "atom_group": torch.cat(atom_group, dim=0),
        "atom_bb_anchor_node": torch.cat(atom_bb_anchor_node, dim=0),
        "atom_node_indices": torch.cat(atom_node_indices, dim=0),

        # residues
        "bb_pos": torch.cat(bb_pos, dim=0),
        "bb_frames": torch.cat(bb_frames, dim=0),

        # topology
        "bond_pairs": torch.cat(bond_pairs, dim=0) if bond_pairs else torch.zeros((0, 2), dtype=torch.long),
        "angle_triples": torch.cat(angle_triples, dim=0) if angle_triples else torch.zeros((0, 3), dtype=torch.long),
        "charged_idx": torch.cat(charged_idx, dim=0) if charged_idx else torch.zeros((0,), dtype=torch.long),
        "charged_q": torch.cat(charged_q, dim=0) if charged_q else torch.zeros((0,), dtype=dtype_float),
        "charged_res_index": torch.cat(charged_res_index, dim=0) if charged_res_index else torch.zeros((0,), dtype=torch.long),
        "dip_C": torch.cat(dip_C, dim=0) if dip_C else torch.zeros((0,), dtype=torch.long),
        "dip_O": torch.cat(dip_O, dim=0) if dip_O else torch.zeros((0,), dtype=torch.long),
        "dip_Nn": torch.cat(dip_Nn, dim=0) if dip_Nn else torch.zeros((0,), dtype=torch.long),
        "dip_res_i": torch.cat(dip_res_i, dim=0) if dip_res_i else torch.zeros((0,), dtype=torch.long),
        "phi_idx": torch.cat(phi_idx, dim=0) if phi_idx else torch.zeros((0, 4), dtype=torch.long),
        "psi_idx": torch.cat(psi_idx, dim=0) if psi_idx else torch.zeros((0, 4), dtype=torch.long),

        # meta
        "frame_sizes": frame_sizes,
        "residue_names_per_frame": residue_names_per_frame,
        "folder_per_frame": folder_per_frame,
        "frame_id_per_frame": frame_id_per_frame,
        "num_nodes": node_offset,
        "num_atoms": atom_offset,
        "num_residues": res_offset,
    }
    return batch

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import json
from pathlib import Path


@dataclass(frozen=True)
class Vocab:
    resname_to_id: Dict[str, int]
    name_to_id: Dict[str, int]  # atom/bead names
    node_type_to_id: Dict[str, int]  # e.g., "BB", "SC", "ATOM"
    atom_group_to_id: Dict[str, int]  # "backbone", "sidechain"

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(
                {
                    "resname_to_id": self.resname_to_id,
                    "name_to_id": self.name_to_id,
                    "node_type_to_id": self.node_type_to_id,
                    "atom_group_to_id": self.atom_group_to_id,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        with Path(path).open("r") as f:
            d = json.load(f)
        return Vocab(
            resname_to_id={str(k): int(v) for k, v in d["resname_to_id"].items()},
            name_to_id={str(k): int(v) for k, v in d["name_to_id"].items()},
            node_type_to_id={str(k): int(v) for k, v in d["node_type_to_id"].items()},
            atom_group_to_id={str(k): int(v) for k, v in d["atom_group_to_id"].items()},
        )


def build_vocab_from_baskets(amino_acid_baskets: Mapping[str, Iterable[dict]]) -> Vocab:
    resnames: Set[str] = set()
    names: Set[str] = set()

    # fixed tokens
    node_types = {"BB": 0, "SC": 1, "ATOM": 2}
    atom_groups = {"backbone": 0, "sidechain": 1}

    for aa_type, osc_list in amino_acid_baskets.items():
        if not isinstance(osc_list, (list, tuple)):
            continue
        for osc in osc_list:
            rk = osc.get("residue_key")
            if rk is not None and len(rk) >= 2:
                resnames.add(str(rk[1]).upper())
            # beads
            if osc.get("oscillator_type") == "backbone":
                names.add("BB")
                sc = osc.get("sc_beads", {}) or {}
                for k in sc.keys():
                    names.add(str(k))
            elif osc.get("oscillator_type") == "sidechain":
                names.add("BB")
                sc = osc.get("sc_beads", {}) or {}
                for k in sc.keys():
                    names.add(str(k))
            # atoms
            atoms = osc.get("atoms", {}) or {}
            for k in atoms.keys():
                # backbone keys are like C_prev, etc, but after reconstruction we'll use standard PDB names
                names.add(str(k))
    # Also include reconstructed backbone names
    for k in ["N", "CA", "C", "O", "H"]:
        names.add(k)

    res_list = sorted(resnames)
    name_list = sorted(names)

    resname_to_id = {r: i for i, r in enumerate(res_list)}
    name_to_id = {n: i for i, n in enumerate(name_list)}

    return Vocab(
        resname_to_id=resname_to_id,
        name_to_id=name_to_id,
        node_type_to_id=node_types,
        atom_group_to_id=atom_groups,
    )

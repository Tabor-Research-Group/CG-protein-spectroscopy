from __future__ import annotations

"""PyTorch dataset for oscillator-local backmapping.

Each entry in your pickle is an *oscillator dictionary* (either backbone or
sidechain). This dataset treats each oscillator as an independent sample and
builds a small graph suitable for the BackmapGNN.

Key robustness features
-----------------------
- NumPy 1.x / 2.x pickle compatibility (see :func:`backmap.data.io.load_pickle_numpy_compat`)
- Deterministic ordering of oscillators (by folder/frame/oscillator_index when available)
- Optional filtering / truncation for debug runs
- Strong sanity checks to prevent silent Na==0 batches (which would yield zero loss)

This dataset does **not** do the train/val/test split itself. Splitting is
implemented at the DataLoader level (see :mod:`backmap.data.splits`).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from backmap.data.io import load_pickle_numpy_compat
from backmap.data.oscillator_graph import build_graph_from_oscillator, GraphVocab


@dataclass(frozen=True)
class DatasetConfig:
    """Controls how oscillator samples are converted into graphs."""

    # If True, coordinates equal to (0,0,0) are treated as missing and dropped.
    drop_zero_atoms: bool = True

    # Maximum SC beads per residue to include (SC1..SCk). Extra beads are ignored.
    max_sidechain_beads: int = 4

    # If True, build a fully-connected directed graph (minus self loops).
    # This is simple and robust for small graphs.
    fully_connected_edges: bool = True

    # Optional cap for quick debug runs.
    max_oscillators: Optional[int] = None

    # If True, exclude sidechain oscillators (GLN-SC/ASN-SC). Useful for debugging.
    include_sidechains: bool = True


def _as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


class OscillatorDataset(Dataset):
    """Dataset over oscillator dictionaries."""

    def __init__(
        self,
        pickle_path: str,
        *,
        vocab: GraphVocab,
        cfg: DatasetConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.pickle_path = str(pickle_path)
        self.vocab = vocab
        self.cfg = cfg or DatasetConfig()
        self.device = torch.device(device)
        self.dtype = dtype

        raw = load_pickle_numpy_compat(self.pickle_path)
        if not isinstance(raw, dict):
            raise ValueError("Expected top-level dict in pickle")

        # Flatten all oscillators across residue types
        osc_list: List[Dict[str, Any]] = []
        for _, lst in raw.items():
            if isinstance(lst, (list, tuple)):
                osc_list.extend(list(lst))

        if not osc_list:
            raise ValueError(f"No oscillators found in {self.pickle_path}")

        # Optional filtering (drop sidechains)
        if not self.cfg.include_sidechains:
            osc_list = [o for o in osc_list if str(o.get("oscillator_type", "")) == "backbone"]

        # Deterministic sort when metadata exists
        def _sort_key(o: Dict[str, Any]):
            return (
                str(o.get("folder", "")),
                _as_int(o.get("frame", -1)),
                _as_int(o.get("oscillator_index", -1)),
                str(o.get("residue_name", "")),
            )

        osc_list = sorted(osc_list, key=_sort_key)

        # Optional truncation
        if self.cfg.max_oscillators is not None:
            osc_list = osc_list[: int(self.cfg.max_oscillators)]

        self.oscillators = osc_list

        # Precompute a few metadata arrays for fast splitting
        self._folders: List[str] = [str(o.get("folder", "")) for o in self.oscillators]
        self._frames: List[int] = [_as_int(o.get("frame", -1)) for o in self.oscillators]

    def __len__(self) -> int:
        return len(self.oscillators)

    def folders(self) -> List[str]:
        """Return folder name for each oscillator (parallel to dataset indices)."""
        return list(self._folders)

    def frames(self) -> List[int]:
        """Return frame number for each oscillator (parallel to dataset indices)."""
        return list(self._frames)

    def unique_folders(self) -> List[str]:
        return sorted(set(self._folders))

    def indices_for_folders(self, folders: Iterable[str]) -> List[int]:
        fset = set(str(f) for f in folders)
        return [i for i, f in enumerate(self._folders) if f in fset]

    def indices_for_folder_frame(self, folder: str, frame: int) -> List[int]:
        folder = str(folder)
        frame = int(frame)
        return [i for i, (f, fr) in enumerate(zip(self._folders, self._frames)) if f == folder and fr == frame]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        osc = self.oscillators[int(idx)]
        sample = build_graph_from_oscillator(
            osc,
            vocab=self.vocab,
            drop_zero_atoms=self.cfg.drop_zero_atoms,
            max_sidechain_beads=self.cfg.max_sidechain_beads,
            fully_connected_edges=self.cfg.fully_connected_edges,
            device=self.device,
            dtype=self.dtype,
        )

        # Hard stop: Na==0 would cause zero losses and will silently waste compute.
        Na = int(sample["x0_local"].shape[0])
        if Na <= 0:
            raise RuntimeError(
                "Dataset produced Na<=0 atoms for a sample. "
                "This would create zero loss. Inspect the oscillator entry and graph builder."
            )

        return sample


def build_default_vocab_from_pickle(pickle_path: str) -> GraphVocab:
    """Scan a pickle and build a stable vocabulary for node/residue/atom group IDs."""
    raw = load_pickle_numpy_compat(pickle_path)
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level dict in pickle")

    residue_names = set(["UNK"])
    node_names = set(["UNK", "BB", "SC1", "SC2", "SC3", "SC4"])
    atom_groups = set(["UNK"])

    # Backbone atom name set (fixed ordering)
    atom_groups.update(["N_prev", "CA_prev", "C_prev", "O_prev", "N_curr", "H_curr", "CA_curr"])

    for _, lst in raw.items():
        if not isinstance(lst, (list, tuple)):
            continue
        for osc in lst:
            residue_names.add(str(osc.get("residue_name", "UNK")))
            atoms = osc.get("atoms", {}) or {}
            for aname in atoms.keys():
                atom_groups.add(str(aname))
                node_names.add(str(aname))
            sc = osc.get("sc_beads", {}) or {}
            for bname in sc.keys():
                node_names.add(str(bname))

    return GraphVocab.from_sets(
        resnames=sorted(residue_names),
        names=sorted(node_names),
        atom_groups=sorted(atom_groups),
    )

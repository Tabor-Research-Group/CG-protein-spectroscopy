#!/usr/bin/env python3
"""
Validate the oscillator pickle dataset used for backmapping training.

This script is intentionally *structurally aware*:
- Backbone oscillators use bb_curr/bb_next anchors and atom names with _prev/_curr suffixes.
- Sidechain oscillators use bb_prev anchor (single-residue) and atom names without suffixes.

It checks:
- Non-finite coordinates (BB anchors and atoms)
- Zero "sentinel" coordinates (required vs optional)
- Atom radii relative to the correct BB anchor bead(s)
- Outliers beyond max_atom_radius and a hard absurd_radius cutoff

Outputs a JSON report for auditing data quality and for selecting a sensible max_atom_radius.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Prefer the project's numpy-compat loader when available
try:
    from backmap.data.io import load_pickle_numpy_compat  # type: ignore
except Exception:  # pragma: no cover
    load_pickle_numpy_compat = None  # type: ignore


def _as_np3(x: Any) -> np.ndarray:
    """Convert to np.float32 shape (3,) when possible; otherwise best-effort."""
    arr = np.asarray(x, dtype=np.float32)
    return arr


def _isfinite(a: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(a)))


def _is_origin(a: np.ndarray, eps: float = 1e-8) -> bool:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size != 3:
        return False
    return bool(np.linalg.norm(a) < eps)


def _is_optional_zero_atom(atom_name: str) -> bool:
    """
    Optional zero-coordinate atoms:
    - Any atom starting with 'H' (missing hydrogens are common)
    - N_prev for the first residue in a chain (may be zero sentinel)
    """
    if atom_name.startswith("H"):
        return True
    if atom_name == "N_prev":
        return True
    return False


def _infer_outer_res_name(outer_key: Any) -> str:
    if outer_key is None:
        return "?"
    if isinstance(outer_key, str):
        return outer_key
    # Sometimes residue_key is (resnum, resname); in that case prefer resname.
    if isinstance(outer_key, tuple) and len(outer_key) == 2:
        return str(outer_key[1])
    return str(outer_key)


def _infer_oscillator_type(entry: Dict[str, Any], outer_res_name: str) -> str:
    """
    Return "backbone" or "sidechain".

    We prefer entry["oscillator_type"] when present; otherwise use heuristics.
    """
    t = entry.get("oscillator_type")
    if isinstance(t, str) and t.strip():
        tl = t.strip().lower()
        if tl in {"backbone", "sidechain"}:
            return tl
    # Heuristics for older pickles
    if outer_res_name.endswith("-SC"):
        return "sidechain"
    if "bb_prev" in entry:
        return "sidechain"
    return "backbone"


def _atom_slot_backbone(atom_name: str) -> int:
    """
    Map backbone atom name -> residue slot:
      *_prev -> 0
      *_curr -> 1
    Unknown names default to 0 (and will be flagged).
    """
    if atom_name.endswith("_prev"):
        return 0
    if atom_name.endswith("_curr"):
        return 1
    return 0


@dataclass
class Anchors:
    bb_pos: List[np.ndarray]
    ok: bool
    osc_type: str
    anchor_field: str  # which field was used as the primary BB anchor for slot 0 (sidechain) or for bb_curr (backbone)


def _get_bb_anchors(entry: Dict[str, Any], osc_type: str, *, sidechain_anchor: str = "sc1") -> Anchors:
    """
    Return BB anchor positions for the oscillator.

    Backbone:
      bb_pos = [bb_curr, bb_next]  (two residues)
    Sidechain:
      bb_pos = [bb_prev]          (single residue)

    We also support limited fallbacks for older/variant pickles:
    - sidechain: bb_prev -> bb_curr -> cg_bead (only if cg_bead_type == "BB")
    """
    if osc_type == "backbone":
        bb_curr = _as_np3(entry.get("bb_curr"))
        bb_next = _as_np3(entry.get("bb_next"))
        ok = _isfinite(bb_curr) and _isfinite(bb_next)
        return Anchors(bb_pos=[bb_curr, bb_next], ok=ok, osc_type=osc_type, anchor_field="bb_curr/bb_next")

    # sidechain
    #
    # Prefer anchoring at SC1 (center bead for sidechain predictions) when available.
    # This matches how the training code constructs sidechain-local frames.
    sidechain_anchor = (sidechain_anchor or "sc1").strip().lower()
    if sidechain_anchor not in {"sc1", "bb_prev"}:
        sidechain_anchor = "sc1"

    if sidechain_anchor == "sc1":
        sc_beads = entry.get("sc_beads", {}) or {}
        sc1_raw = sc_beads.get("SC1", None)
        # Fallback: some variants store SC1 in cg_bead when cg_bead_type=='SC1'
        if sc1_raw is None and str(entry.get("cg_bead_type", "")).upper() == "SC1":
            sc1_raw = entry.get("cg_bead", None)
        if sc1_raw is not None:
            sc1 = _as_np3(sc1_raw)
            if _isfinite(sc1):
                return Anchors(bb_pos=[sc1], ok=True, osc_type=osc_type, anchor_field="sc1")

    # Legacy/default: anchor at bb_prev (BB bead for the residue)
    anchor_field = "bb_prev"
    bb_prev_raw = entry.get("bb_prev", None)
    if bb_prev_raw is None:
        # fallback 1: some pipelines store sidechain anchor as bb_curr
        bb_prev_raw = entry.get("bb_curr", None)
        anchor_field = "bb_curr(fallback)"
    if bb_prev_raw is None:
        # fallback 2: cg_bead if explicitly tagged as BB
        if str(entry.get("cg_bead_type", "")).upper() == "BB":
            bb_prev_raw = entry.get("cg_bead", None)
            anchor_field = "cg_bead(BB fallback)"

    bb_prev = _as_np3(bb_prev_raw)
    ok = _isfinite(bb_prev)
    return Anchors(bb_pos=[bb_prev], ok=ok, osc_type=osc_type, anchor_field=anchor_field)


def _iter_entries(data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (outer_res_name, entry) pairs for:
      - top-level dict: {residue_name: [entry, ...], ...}
      - top-level list: [entry, ...]
    """
    if isinstance(data, dict):
        for outer_key, entries in data.items():
            outer_res = _infer_outer_res_name(outer_key)
            if entries is None:
                continue
            for e in entries:
                if isinstance(e, dict):
                    yield outer_res, e
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                outer_res = _infer_outer_res_name(e.get("residue_name", "?"))
                yield outer_res, e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle_path", type=str, required=True)
    ap.add_argument("--max_atom_radius", type=float, default=6.0)
    ap.add_argument("--absurd_radius", type=float, default=25.0, help="Hard cutoff to flag extreme outliers")
    ap.add_argument(
        "--sidechain_anchor",
        type=str,
        default="sc1",
        choices=["sc1", "bb_prev"],
        help=(
            "Which anchor to use for sidechain oscillators when computing radii. "
            "'sc1' (recommended) anchors at sc_beads['SC1'] when available; "
            "'bb_prev' anchors at bb_prev (legacy)."
        ),
    )
    ap.add_argument("--report_path", type=str, default="dataset_report.json")
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    pkl_path = Path(args.pickle_path)

    # Load (prefer compat loader if present)
    if load_pickle_numpy_compat is not None:
        data = load_pickle_numpy_compat(pkl_path)
    else:  # pragma: no cover
        import pickle
        data = pickle.loads(pkl_path.read_bytes())

    # Top-level info
    if isinstance(data, dict):
        top_level_type = "dict"
        n_top_keys = len(data)
        # Estimate entries count without materializing everything twice
        n_entries = sum(len(v) for v in data.values() if isinstance(v, list))
    elif isinstance(data, list):
        top_level_type = "list"
        n_top_keys = 0
        n_entries = len(data)
    else:
        top_level_type = type(data).__name__
        n_top_keys = 0
        n_entries = 0

    # Global counters
    entries_with_any_nonfinite = 0
    entries_with_bb_nonfinite = 0
    entries_with_atom_nonfinite = 0
    entries_with_required_zero = 0
    entries_with_optional_zero = 0
    entries_outside_max = 0
    entries_outside_absurd = 0

    atoms_total = 0
    atoms_nonfinite_total = 0
    atoms_zero_required = 0
    atoms_zero_optional = 0
    atoms_outside_max_total = 0
    atoms_outside_absurd_total = 0

    top_zero_optional: Counter[str] = Counter()
    top_zero_required: Counter[str] = Counter()
    top_nonfinite_atoms: Counter[str] = Counter()

    radii: List[float] = []
    worst_by_max_r: List[Tuple[float, Dict[str, Any]]] = []

    # Sidechain-specific diagnostics to detect residue/anchor mismatches.
    # If the data are consistent, CA should be close to the BB bead, and SC1 should
    # be a few Å from BB (depending on residue).
    sc_bb_to_sc1: List[float] = []
    sc_ca_to_bb: List[float] = []
    sc_ca_to_sc1: List[float] = []
    sc_ca_to_anchor: List[float] = []

    # Per-residue counters (keyed by *outer residue name*, string)
    by_res = defaultdict(
        lambda: {
            "n_entries": 0,
            "n_any_nonfinite": 0,
            "n_bb_nonfinite": 0,
            "n_atom_nonfinite": 0,
            "n_required_zero": 0,
            "n_optional_zero": 0,
            "n_outside_max": 0,
            "n_outside_absurd": 0,
            "max_r": 0.0,
        }
    )

    for outer_res_name, entry in _iter_entries(data):
        # Identify metadata
        residue_name = str(entry.get("residue_name") or outer_res_name or "?")
        folder = str(entry.get("folder", "?"))
        frame = int(entry.get("frame", -1))
        osc_type = _infer_oscillator_type(entry, outer_res_name)

        anchors = _get_bb_anchors(entry, osc_type, sidechain_anchor=args.sidechain_anchor)

        atoms = entry.get("atoms", {}) or {}
        if not isinstance(atoms, dict):
            atoms = {}

        # --- Sidechain anchor sanity diagnostics ---
        # These help distinguish "max radius is big because the anchor is BB" vs
        # "max radius is big because atoms belong to a different residue".
        diag_bb_to_sc1: Optional[float] = None
        diag_ca_to_bb: Optional[float] = None
        diag_ca_to_sc1: Optional[float] = None
        diag_ca_to_anchor: Optional[float] = None

        if osc_type == "sidechain":
            # BB bead for this residue (may be stored as bb_prev, sometimes bb_curr).
            bb_prev_raw = entry.get("bb_prev", None)
            if bb_prev_raw is None:
                bb_prev_raw = entry.get("bb_curr", None)
            bb_prev = None
            if bb_prev_raw is not None:
                bb_prev_np = _as_np3(bb_prev_raw)
                if _isfinite(bb_prev_np):
                    bb_prev = bb_prev_np

            # SC1 bead for this residue (preferred sidechain anchor)
            sc1 = None
            sc_beads = entry.get("sc_beads", {}) or {}
            sc1_raw = sc_beads.get("SC1", None)
            if sc1_raw is None and str(entry.get("cg_bead_type", "")).upper() == "SC1":
                sc1_raw = entry.get("cg_bead", None)
            if sc1_raw is not None:
                sc1_np = _as_np3(sc1_raw)
                if _isfinite(sc1_np):
                    sc1 = sc1_np

            if bb_prev is not None and sc1 is not None:
                diag_bb_to_sc1 = float(np.linalg.norm(sc1 - bb_prev))
                sc_bb_to_sc1.append(diag_bb_to_sc1)

            # CA atom (if present) should be close to the BB bead; if it isn't,
            # the entry may be mismatched.
            ca = None
            for ca_key in ("CA", "CA_prev", "CA_curr"):
                if ca_key in atoms:
                    ca_np = _as_np3(atoms.get(ca_key))
                    if _isfinite(ca_np) and (not _is_origin(ca_np)):
                        ca = ca_np
                        break

            if ca is not None:
                if bb_prev is not None:
                    diag_ca_to_bb = float(np.linalg.norm(ca - bb_prev))
                    sc_ca_to_bb.append(diag_ca_to_bb)
                if sc1 is not None:
                    diag_ca_to_sc1 = float(np.linalg.norm(ca - sc1))
                    sc_ca_to_sc1.append(diag_ca_to_sc1)
                if anchors.ok:
                    diag_ca_to_anchor = float(np.linalg.norm(ca - anchors.bb_pos[0]))
                    sc_ca_to_anchor.append(diag_ca_to_anchor)

        by_res[residue_name]["n_entries"] += 1

        entry_has_any_nonfinite = False
        entry_has_bb_nonfinite = not anchors.ok
        entry_has_atom_nonfinite = False
        entry_has_required_zero = False
        entry_has_optional_zero = False
        entry_has_outside_max = False
        entry_has_outside_absurd = False

        if entry_has_bb_nonfinite:
            entry_has_any_nonfinite = True

        entry_max_r = 0.0

        for aname, apos_any in atoms.items():
            atoms_total += 1
            aname = str(aname)
            apos = _as_np3(apos_any)

            # Atom finite check
            if not _isfinite(apos):
                atoms_nonfinite_total += 1
                top_nonfinite_atoms[aname] += 1
                entry_has_any_nonfinite = True
                entry_has_atom_nonfinite = True
                continue

            # Zero sentinel (origin) check
            if _is_origin(apos):
                if _is_optional_zero_atom(aname):
                    atoms_zero_optional += 1
                    top_zero_optional[aname] += 1
                    entry_has_optional_zero = True
                else:
                    atoms_zero_required += 1
                    top_zero_required[aname] += 1
                    entry_has_required_zero = True
                # Do not compute radii for zero sentinel atoms
                continue

            # Radii checks (only if BB anchors are valid)
            if anchors.ok:
                if osc_type == "backbone":
                    ridx = _atom_slot_backbone(aname)
                    if ridx >= len(anchors.bb_pos):
                        # unexpected naming (shouldn't happen for well-formed backbone data)
                        entry_has_outside_max = True
                        continue
                else:
                    ridx = 0

                r = float(np.linalg.norm(apos - anchors.bb_pos[ridx]))
                radii.append(r)
                entry_max_r = max(entry_max_r, r)

                if r > args.max_atom_radius:
                    atoms_outside_max_total += 1
                    entry_has_outside_max = True
                if r > args.absurd_radius:
                    atoms_outside_absurd_total += 1
                    entry_has_outside_absurd = True

        # Update per-entry counters
        if entry_has_any_nonfinite:
            entries_with_any_nonfinite += 1
            by_res[residue_name]["n_any_nonfinite"] += 1
        if entry_has_bb_nonfinite:
            entries_with_bb_nonfinite += 1
            by_res[residue_name]["n_bb_nonfinite"] += 1
        if entry_has_atom_nonfinite:
            entries_with_atom_nonfinite += 1
            by_res[residue_name]["n_atom_nonfinite"] += 1
        if entry_has_required_zero:
            entries_with_required_zero += 1
            by_res[residue_name]["n_required_zero"] += 1
        if entry_has_optional_zero:
            entries_with_optional_zero += 1
            by_res[residue_name]["n_optional_zero"] += 1
        if entry_has_outside_max:
            entries_outside_max += 1
            by_res[residue_name]["n_outside_max"] += 1
        if entry_has_outside_absurd:
            entries_outside_absurd += 1
            by_res[residue_name]["n_outside_absurd"] += 1

        by_res[residue_name]["max_r"] = max(by_res[residue_name]["max_r"], float(entry_max_r))

        if entry_max_r > 0:
            worst_by_max_r.append(
                (
                    entry_max_r,
                    {
                        "folder": folder,
                        "frame": frame,
                        "residue_name": residue_name,
                        "oscillator_type": osc_type,
                        "anchor_field": anchors.anchor_field,
                        "max_r": float(entry_max_r),
                        "diag_bb_to_sc1": diag_bb_to_sc1,
                        "diag_ca_to_bb": diag_ca_to_bb,
                        "diag_ca_to_sc1": diag_ca_to_sc1,
                        "diag_ca_to_anchor": diag_ca_to_anchor,
                    },
                )
            )

    worst_by_max_r.sort(key=lambda x: x[0], reverse=True)
    worst_by_max_r = worst_by_max_r[: int(args.top_k)]

    radii_np = np.asarray(radii, dtype=np.float32)
    radii_np = radii_np[np.isfinite(radii_np)]

    def _stats_from_list(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {}
        a = np.asarray(values, dtype=np.float32)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {}
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    summary: Dict[str, Any] = {
        "pickle_path": str(pkl_path),
        "top_level_type": top_level_type,
        "n_top_level_keys": int(n_top_keys),
        "n_entries": int(n_entries),
        "max_atom_radius": float(args.max_atom_radius),
        "absurd_radius": float(args.absurd_radius),
        "entries_with_any_nonfinite": int(entries_with_any_nonfinite),
        "entries_with_bb_nonfinite": int(entries_with_bb_nonfinite),
        "entries_with_atom_nonfinite": int(entries_with_atom_nonfinite),
        "entries_with_required_zero_coords": int(entries_with_required_zero),
        "entries_with_optional_zero_coords": int(entries_with_optional_zero),
        "entries_with_atoms_outside_max_radius": int(entries_outside_max),
        "entries_with_atoms_outside_absurd_radius": int(entries_outside_absurd),
        "atoms_total": int(atoms_total),
        "atoms_nonfinite_total": int(atoms_nonfinite_total),
        "atoms_zero_required": int(atoms_zero_required),
        "atoms_zero_optional": int(atoms_zero_optional),
        "atoms_outside_max_total": int(atoms_outside_max_total),
        "atoms_outside_absurd_total": int(atoms_outside_absurd_total),
        "top_zero_optional_atom_names": dict(top_zero_optional.most_common(20)),
        "top_zero_required_atom_names": dict(top_zero_required.most_common(20)),
        "top_nonfinite_atom_names": dict(top_nonfinite_atoms.most_common(20)),
        "radii": {},
        "sidechain_diagnostics": {
            "sidechain_anchor_mode": str(args.sidechain_anchor),
            "bb_to_sc1": _stats_from_list(sc_bb_to_sc1),
            "ca_to_bb": _stats_from_list(sc_ca_to_bb),
            "ca_to_sc1": _stats_from_list(sc_ca_to_sc1),
            "ca_to_anchor": _stats_from_list(sc_ca_to_anchor),
        },
        "worst_by_max_r": [w[1] for w in worst_by_max_r],
        "by_residue_name": dict(by_res),
    }

    if radii_np.size:
        summary["radii"] = {
            "n": int(radii_np.size),
            "mean": float(radii_np.mean()),
            "std": float(radii_np.std()),
            "p50": float(np.percentile(radii_np, 50)),
            "p90": float(np.percentile(radii_np, 90)),
            "p99": float(np.percentile(radii_np, 99)),
            "max": float(radii_np.max()),
        }

    report_path = Path(args.report_path)
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote report: {report_path}")

    # Print a short console summary
    console = {
        "top_level_type": summary["top_level_type"],
        "n_top_level_keys": summary["n_top_level_keys"],
        "n_entries": summary["n_entries"],
        "entries_with_any_nonfinite": summary["entries_with_any_nonfinite"],
        "entries_with_bb_nonfinite": summary["entries_with_bb_nonfinite"],
        "entries_with_atom_nonfinite": summary["entries_with_atom_nonfinite"],
        "entries_with_required_zero_coords": summary["entries_with_required_zero_coords"],
        "entries_with_optional_zero_coords": summary["entries_with_optional_zero_coords"],
        "entries_with_atoms_outside_max_radius": summary["entries_with_atoms_outside_max_radius"],
        "entries_with_atoms_outside_absurd_radius": summary["entries_with_atoms_outside_absurd_radius"],
        "radii": summary["radii"],
    }
    print(json.dumps(console, indent=2))


if __name__ == "__main__":
    main()

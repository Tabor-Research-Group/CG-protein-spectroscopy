from __future__ import annotations

"""Plotting utilities.

This module writes a rich set of diagnostic plots at each epoch.

Why so many plots?
------------------
Backmapping quality can regress in specific geometric aspects even if a single
scalar loss decreases. The plots here focus on interpretable quantities:

- Dipole alignment distributions (cosine similarity)
- Bond length scatter and error distributions
- Bond angle and dihedral distributions
- Local radial distance distributions
- Nonbonded minimum-distance distributions and repulsion proxies

All plots are saved as PNGs under:
  <run_dir>/plots/epoch_XXXX/<split>/

The training script also writes a JSON summary with the same quantities.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_np(x) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=np.float32)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _hist_plot(
    path: Path,
    data: np.ndarray,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "count",
    bins: int = 60,
) -> None:
    """
    Safe histogram plotter.

    IMPORTANT: accepts xlabel/ylabel for backward compatibility because
    some call sites pass these kwargs. Skips plotting if data is empty
    or all non-finite, to avoid crashing training.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    if data is None:
        return

    path = Path(path)

    data = np.asarray(data).reshape(-1)
    data = data[np.isfinite(data)]

    if data.size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".skipped.txt", "w") as f:
            f.write(f"Skipped histogram '{title}': no finite values.\n")
        return

    plt.figure(figsize=(4, 3))
    plt.hist(data, bins=bins)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _scatter_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    identity_line: bool = True,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert and flatten
    if x is None or y is None:
        return

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # If caller accidentally filtered x/y independently, shapes may no longer match.
    if x.size != y.size:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + '.skipped.txt', 'w') as f:
            f.write(f"Skipped scatter '{title}': shape mismatch x={x.size} y={y.size}.\n")
        return

    # Keep only finite pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Nothing to plot → skip safely
    if x.size == 0 or y.size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".skipped.txt", "w") as f:
            f.write(f"Skipped scatter '{title}': no finite points.\n")
        return

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=4, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if identity_line:
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def plot_epoch_metrics(
    *,
    out_dir: str | Path,
    epoch: int,
    split: str,
    losses: Dict[str, float],
    metrics: Dict[str, np.ndarray],
    cfg=None,
    plot_cfg=None,
    bins: int = 60,
) -> None:
    """Write per-epoch plots.

    Parameters
    ----------
    out_dir:
        Run directory (contains metrics logs and checkpoints).
    epoch:
        Epoch number (1-indexed).
    split:
        "train" | "val" | "test".
    losses:
        Scalar mean losses for the split.
    metrics:
        Dict of metric arrays (NumPy). Expected keys are those produced by the
        trainer's evaluation routine.
    bins:
        Histogram bins.
    """
    out_dir = Path(out_dir)
    base = out_dir / "plots" / f"epoch_{epoch:04d}" / split
    _ensure_dir(base)

    # Save JSON summary for quick grepping
    summary = {
        "epoch": int(epoch),
        "split": str(split),
        "losses": {k: float(v) for k, v in losses.items()},
        "metrics": {},
    }
    for k, arr in metrics.items():
        arr = _to_np(arr).reshape(-1)
        if arr.size == 0:
            summary["metrics"][k] = {"n": 0}
        else:
            summary["metrics"][k] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # --- Dipole correlation ---
    dip = _to_np(metrics.get("dipole_cos"))
    _hist_plot(
        base / "dipole_correlation_hist.png",
        dip,
        bins=bins,
        title=f"Dipole correlation (cos) | {split} | epoch {epoch}",
        xlabel="cos(pred,true)",
    )

    # --- Radial distances ---
    r_true = _to_np(metrics.get("radial_true"))
    r_pred = _to_np(metrics.get("radial_pred"))
    _hist_plot(
        base / "radial_true_hist.png",
        r_true,
        bins=bins,
        title=f"Local radial distances | TRUE | {split} | epoch {epoch}",
        xlabel="r (Å)",
    )
    _hist_plot(
        base / "radial_pred_hist.png",
        r_pred,
        bins=bins,
        title=f"Local radial distances | PRED | {split} | epoch {epoch}",
        xlabel="r (Å)",
    )

    # --- Bond lengths ---
    b_true = _to_np(metrics.get("bond_true"))
    b_pred = _to_np(metrics.get("bond_pred"))
    _scatter_plot(
        base / "bond_length_scatter.png",
        b_true,
        b_pred,
        title=f"Bond length scatter | {split} | epoch {epoch}",
        xlabel="bond length TRUE (Å)",
        ylabel="bond length PRED (Å)",
    )
    if b_true.size and b_pred.size:
        _hist_plot(
            base / "bond_length_error_hist.png",
            b_pred - b_true,
            bins=bins,
            title=f"Bond length error | {split} | epoch {epoch}",
            xlabel="pred - true (Å)",
        )

    # --- Bond angles ---
    a_true = _to_np(metrics.get("angle_true"))
    a_pred = _to_np(metrics.get("angle_pred"))
    _hist_plot(
        base / "angle_true_hist.png",
        a_true,
        bins=bins,
        title=f"Bond angles | TRUE | {split} | epoch {epoch}",
        xlabel="angle (rad)",
    )
    _hist_plot(
        base / "angle_pred_hist.png",
        a_pred,
        bins=bins,
        title=f"Bond angles | PRED | {split} | epoch {epoch}",
        xlabel="angle (rad)",
    )
    if a_true.size and a_pred.size:
        _hist_plot(
            base / "angle_error_hist.png",
            a_pred - a_true,
            bins=bins,
            title=f"Bond angle error | {split} | epoch {epoch}",
            xlabel="pred - true (rad)",
        )

    # --- Dihedrals ---
    d_true = _to_np(metrics.get("dihedral_true"))
    d_pred = _to_np(metrics.get("dihedral_pred"))
    _hist_plot(
        base / "dihedral_true_hist.png",
        d_true,
        bins=bins,
        title=f"Dihedrals | TRUE | {split} | epoch {epoch}",
        xlabel="dihedral (rad)",
    )
    _hist_plot(
        base / "dihedral_pred_hist.png",
        d_pred,
        bins=bins,
        title=f"Dihedrals | PRED | {split} | epoch {epoch}",
        xlabel="dihedral (rad)",
    )

    # --- Nonbonded minimum distances ---
    nb_true = _to_np(metrics.get("nonbond_min_true"))
    nb_pred = _to_np(metrics.get("nonbond_min_pred"))
    _hist_plot(
        base / "nonbond_min_true_hist.png",
        nb_true,
        bins=bins,
        title=f"Minimum nonbonded distance | TRUE | {split} | epoch {epoch}",
        xlabel="min r (Å)",
    )
    _hist_plot(
        base / "nonbond_min_pred_hist.png",
        nb_pred,
        bins=bins,
        title=f"Minimum nonbonded distance | PRED | {split} | epoch {epoch}",
        xlabel="min r (Å)",
    )

    # --- Repulsion energy proxy ---
    e_true = _to_np(metrics.get("repulsion_energy_true"))
    e_pred = _to_np(metrics.get("repulsion_energy_pred"))
    _scatter_plot(
        base / "repulsion_energy_scatter.png",
        e_true,
        e_pred,
        title=f"Repulsion energy proxy scatter | {split} | epoch {epoch}",
        xlabel="energy TRUE",
        ylabel="energy PRED",
        identity_line=True,
    )
    _hist_plot(
        base / "repulsion_energy_true_hist.png",
        e_true,
        bins=bins,
        title=f"Repulsion energy proxy | TRUE | {split} | epoch {epoch}",
        xlabel="energy",
    )
    _hist_plot(
        base / "repulsion_energy_pred_hist.png",
        e_pred,
        bins=bins,
        title=f"Repulsion energy proxy | PRED | {split} | epoch {epoch}",
        xlabel="energy",
    )

    # A short text file with scalar losses for convenience
    lines = [f"epoch: {epoch}", f"split: {split}"]
    for k, v in losses.items():
        lines.append(f"{k}: {float(v):.8f}")
    (base / "losses.txt").write_text("\n".join(lines) + "\n")

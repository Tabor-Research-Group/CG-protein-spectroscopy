#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
# -----------------------------------------------------------------------------
# Make the repository importable when running as:
#   python scripts/train.py ...
# without requiring `pip install -e .`.
# -----------------------------------------------------------------------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
del _os, _sys, _REPO_ROOT


"""Train the diffusion backmapping model.

Highlights
----------
- Optionally writes random PDB overlays for VMD (GT vs prediction vs CG)

Usage
-----
  python scripts/train.py --config configs/train_example.yaml

You can override common options:
  python scripts/train.py --config configs/train_example.yaml \
      --pickle path/to/amino_acid_baskets.pkl \
      --out runs/exp01 --device cuda --epochs 20 --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from backmap.config import Config
from backmap.data.collate import collate_graph_samples
from backmap.data.oscillator_dataset import DatasetConfig, OscillatorDataset, build_default_vocab_from_pickle
from backmap.data.splits import split_indices
from backmap.model.diffusion import GaussianDiffusion, make_schedule
from backmap.model.gnn import BackmapGNN, EdgeCutoffs
from backmap.model.pipeline import (
    atoms_local_to_global,
    build_atom_node_mask,
    build_node_atom_group,
    build_node_geom_sph,
    build_node_pos,
)
from backmap.train.losses import LossBreakdown, compute_losses
from backmap.train.metrics import BatchMetrics, compute_batch_metrics
from backmap.train.plotting import plot_epoch_metrics
from backmap.utils.checkpoint import load_checkpoint, save_checkpoint
from backmap.utils.logging import JsonlWriter, setup_logger
from backmap.utils.seed import seed_all
from backmap.utils.pdb import (
    aggregate_atomistic_from_oscillators,
    aggregate_cg_from_oscillators,
    aggregate_predicted_from_oscillator_predictions,
    write_multichain_pdb,
)


def _device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_batch_to(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensor values in a collated batch to device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _build_model(cfg: Config, vocab, device: torch.device) -> BackmapGNN:
    model = BackmapGNN(
        num_resnames=vocab.num_resnames,
        num_names=vocab.num_names,
        num_node_types=2,  # bead / atom
        num_atom_groups=vocab.num_atom_groups,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        time_embed_dim=cfg.model.time_embed_dim,
        pos_embed_dim=cfg.model.pos_embed_dim,
        rbf_num_centers=cfg.model.rbf_num_centers,
        rbf_max_dist=cfg.model.rbf_max_dist,
        max_atom_radius=cfg.model.max_atom_radius,
        eps=cfg.loss.eps,
    ).to(device=device)
    return model


def _build_diffusion(cfg: Config, device: torch.device) -> GaussianDiffusion:
    schedule = make_schedule(
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        device=device,
    )
    return GaussianDiffusion(
        schedule=schedule,
        timesteps=cfg.diffusion.timesteps,
        max_radius=cfg.model.max_atom_radius,
        clip_each_step=True,
        eps=cfg.loss.eps,
    )


def _forward_denoise_step(
    *,
    model: BackmapGNN,
    diffusion: GaussianDiffusion,
    batch: Dict[str, Any],
    edge_cutoffs: EdgeCutoffs,
    cfg: Config,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, LossBreakdown, Optional[BatchMetrics]]:
    """Run one denoising training step on a collated batch.

    Returns
    -------
    total_loss:
        Scalar tensor.
    breakdown:
        Loss breakdown dataclass.
    metrics:
        Optional rich metrics for plotting.
    """
    # Ground truth x0 (local) for all atoms in this batch
    x0_local = batch["x0_local"]  # [Na,3]
    Na = int(x0_local.shape[0])
    if Na <= 0:
        raise RuntimeError("Na<=0 atoms in batch: this would produce zero loss")

    # Sample one diffusion timestep per *sample* (oscillator) for stability
    B = int(batch["batch_size"])
    T = int(cfg.diffusion.timesteps)
    if rng is None:
        t_sample = torch.randint(low=0, high=T, size=(B,), device=x0_local.device)
    else:
        t_sample = torch.randint(low=0, high=T, size=(B,), device=x0_local.device, generator=rng)

    t_node = t_sample[batch["node_batch"]]  # [Nn]
    t_atom = t_sample[batch["atom_batch"]]  # [Na]

    # Forward diffusion in local coordinates
    xt_local = diffusion.q_sample(x0_local, t_atom)

    # Convert current xt to global to build node positions
    xt_global = atoms_local_to_global(xt_local, batch["atom_res"], batch["bb_pos"], batch["bb_frames"])

    # Build full node features
    Nn = int(batch["num_nodes"])
    node_pos = build_node_pos(
        num_nodes=Nn,
        bead_node_indices=batch["bead_node_indices"],
        bead_pos=batch["bead_pos"],
        atom_node_indices=batch["atom_node_indices"],
        atom_pos=xt_global,
    )
    node_geom = build_node_geom_sph(Nn, batch["atom_node_indices"], xt_local, eps=cfg.loss.eps)
    node_atom_group = build_node_atom_group(Nn, batch["atom_node_indices"], batch["atom_group"])
    atom_mask = build_atom_node_mask(Nn, batch["atom_node_indices"])

    # Predict x0 (local) for the atom nodes
    _, x0_pred_local = model(
        node_pos=node_pos,
        node_type=batch["node_type"],
        node_name=batch["node_name"],
        node_resname=batch["node_resname"],
        node_res=batch["node_res"],
        node_res_in_frame=batch["node_res_in_frame"],
        node_atom_group=node_atom_group,
        edge_index=batch["edge_index"],
        edge_type=batch["edge_type"],
        bb_frames=batch["bb_frames"],
        t_node=t_node,
        node_geom_sph=node_geom,
        atom_node_mask=atom_mask,
        edge_cutoffs=edge_cutoffs,
    )

    breakdown = compute_losses(pred_local=x0_pred_local, target_local=x0_local, batch=batch, cfg=cfg.loss)

    # Rich metrics for plotting (optionally)
    metrics = compute_batch_metrics(pred_local=x0_pred_local, target_local=x0_local, batch=batch, cfg=cfg.loss)

    return breakdown.total, breakdown, metrics


@torch.no_grad()
def _evaluate(
    *,
    loader: DataLoader,
    model: BackmapGNN,
    diffusion: GaussianDiffusion,
    cfg: Config,
    device: torch.device,
    edge_cutoffs: EdgeCutoffs,
    max_batches_for_metrics: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate a split.

    Notes
    -----
    Unlike the training loop, evaluation must *not* allow malformed batches to
    dominate the reported mean (e.g., clamped-to-cap values). We therefore
    **skip** batches flagged as bad by compute_losses (non-finite intermediate
    values), and we log how many were skipped.

    MODIFIED: Now collects metrics from ALL batches when max_batches_for_metrics
    is None or 0 (for publication-quality plots with complete data).

    Returns
    -------
    loss_means:
        Mean losses for the split (over good batches only).
    metrics_agg:
        Dict of numpy arrays for plotting (aggregated over all batches
        or up to max_batches_for_metrics if specified).
    """
    model.eval()

    sum_total = 0.0
    sum_den_cart = 0.0
    sum_den_sph = 0.0
    sum_bond = 0.0
    sum_angle = 0.0
    sum_dihed = 0.0
    sum_dip = 0.0
    sum_contact = 0.0

    n_good = 0
    n_bad = 0
    n_metric_batches = 0

    # These are the metrics we'd LIKE to collect for plotting if they exist.
    # (Do not include names that are not guaranteed by BatchMetrics.)
    preferred_metric_keys = [
        "radial_true",
        "radial_pred",
        "bond_true",
        "bond_pred",
        "angle_true",
        "angle_pred",
        "dihedral_true",
        "dihedral_pred",
        "dipole_cos",
        "nonbond_min_true",      # NEW: for publication plots
        "nonbond_min_pred",      # NEW: for publication plots
        "repulsion_energy_true", # NEW: for publication plots
        "repulsion_energy_pred", # NEW: for publication plots
        # contact metric name has changed across versions; accept either if present
        "contact",
        "contact_penalty",
        "contact_pen",
    ]

    # We will resolve actual keys after seeing the first non-None metrics object.
    metrics_lists: Dict[str, List[np.ndarray]] = {}
    resolved_metric_keys: List[str] = []

    def _init_metrics_lists(metrics_obj) -> None:
        """Initialize metric collection using fields that actually exist."""
        nonlocal metrics_lists, resolved_metric_keys

        # BatchMetrics is typically a dataclass; dataclasses have __dict__.
        # Fall back to dir() if needed.
        if hasattr(metrics_obj, "__dict__"):
            available = set(metrics_obj.__dict__.keys())
        else:
            available = {k for k in dir(metrics_obj) if not k.startswith("_")}

        # Keep preferred keys that actually exist
        resolved_metric_keys = [k for k in preferred_metric_keys if k in available]

        # If none of the preferred metrics exist, still avoid crashing:
        # collect any tensor-like public fields (useful for debugging)
        if not resolved_metric_keys:
            for k in sorted(available):
                if k.startswith("_"):
                    continue
                v = getattr(metrics_obj, k, None)
                if torch.is_tensor(v):
                    resolved_metric_keys.append(k)

        metrics_lists = {k: [] for k in resolved_metric_keys}

    # CRITICAL FIX: Determine if we should collect ALL batches
    # When max_batches_for_metrics is None or 0, collect everything
    collect_all_batches = (max_batches_for_metrics is None or 
                          max_batches_for_metrics <= 0)

    for batch in loader:
        batch = _move_batch_to(batch, device=device)

        _, breakdown, metrics = _forward_denoise_step(
            model=model,
            diffusion=diffusion,
            batch=batch,
            edge_cutoffs=edge_cutoffs,
            cfg=cfg,
            rng=None,
        )

        if breakdown.bad or (not torch.isfinite(breakdown.total)):
            n_bad += 1
            continue

        sum_total += float(breakdown.total.detach().cpu())
        sum_den_cart += float(breakdown.denoise_cart.detach().cpu())
        sum_den_sph += float(breakdown.denoise_sph.detach().cpu())
        sum_bond += float(breakdown.bond.detach().cpu())
        sum_angle += float(breakdown.angle.detach().cpu())
        sum_dihed += float(breakdown.dihedral.detach().cpu())
        sum_dip += float(breakdown.dipole.detach().cpu())
        sum_contact += float(breakdown.contact.detach().cpu())
        n_good += 1

        # MODIFIED: Collect metrics from ALL batches (or up to specified limit)
        # Handle case where max_batches_for_metrics might be None
        should_collect_metrics = False
        if metrics is not None:
            if collect_all_batches:
                # Collect all batches
                should_collect_metrics = True
            else:
                # Only collect up to limit
                should_collect_metrics = n_metric_batches < int(max_batches_for_metrics)
        
        if should_collect_metrics:
            if not metrics_lists:
                _init_metrics_lists(metrics)

            for k in resolved_metric_keys:
                v = getattr(metrics, k, None)
                if v is None:
                    continue
                if not torch.is_tensor(v):
                    continue
                # We only want finite values for plots; keep shape consistent
                arr = v.detach().cpu().numpy()
                metrics_lists[k].append(arr)

            n_metric_batches += 1


    if n_good == 0:
        loss_means = {
            "total": float("inf"),
            "denoise_cart": float("inf"),
            "denoise_sph": float("inf"),
            "bond": float("inf"),
            "angle": float("inf"),
            "dihedral": float("inf"),
            "dipole": float("inf"),
            "contact": float("inf"),
        }
        # If we never initialized metrics_lists, return empty arrays for preferred keys
        empty_keys = resolved_metric_keys if resolved_metric_keys else [
            "radial_true", "radial_pred", "bond_true", "bond_pred",
            "angle_true", "angle_pred", "dihedral_true", "dihedral_pred",
            "dipole_cos", "contact"
        ]
        metrics_agg = {k: np.zeros((0,), dtype=np.float32) for k in empty_keys}
        return loss_means, metrics_agg

    loss_means = {
        "total": sum_total / n_good,
        "denoise_cart": sum_den_cart / n_good,
        "denoise_sph": sum_den_sph / n_good,
        "bond": sum_bond / n_good,
        "angle": sum_angle / n_good,
        "dihedral": sum_dihed / n_good,
        "dipole": sum_dip / n_good,
        "contact": sum_contact / n_good,
    }

    # Concatenate metric arrays across batches (if any)
    metrics_agg = {}
    for k, vlist in metrics_lists.items():
        if len(vlist):
            try:
                metrics_agg[k] = np.concatenate(vlist, axis=0)
            except Exception:
                # In case shapes differ across batches, flatten each then concat
                metrics_agg[k] = np.concatenate([x.reshape(-1) for x in vlist], axis=0)
        else:
            metrics_agg[k] = np.zeros((0,), dtype=np.float32)

    return loss_means, metrics_agg

@torch.no_grad()
def _write_random_frame_overlays(
    *,
    out_dir: Path,
    epoch: int,
    cfg: Config,
    dataset: OscillatorDataset,
    model: BackmapGNN,
    diffusion: GaussianDiffusion,
    device: torch.device,
    split_name: str,
    indices: list[int],
    max_frames: int,
    edge_cutoffs: EdgeCutoffs,
) -> None:
    """Write a few random frame overlays for VMD.

    This uses *ground truth atoms/beads* from the pickle, and *predicted atoms*
    from a single reverse-diffusion sampling run.

    We keep this very small per epoch (default 2 frames) to avoid slowing training.
    """
    if max_frames <= 0:
        return

    # Build a map of (folder,frame) -> indices
    frame_to_indices: Dict[Tuple[str, int], list[int]] = {}
    for i in indices:
        osc = dataset.oscillators[int(i)]
        key = (str(osc.get("folder", "")), int(osc.get("frame", -1)))
        frame_to_indices.setdefault(key, []).append(int(i))

    keys = list(frame_to_indices.keys())
    if not keys:
        return

    # Deterministic-ish selection based on epoch
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.train.seed) + int(epoch) * 10007)

    # Choose up to max_frames random frame keys
    perm = torch.randperm(len(keys), generator=rng).tolist()
    chosen = [keys[j] for j in perm[: max_frames]]

    from backmap.model.sampling import sample_atoms_full

    for (folder, frame) in chosen:
        osc_indices = frame_to_indices[(folder, frame)]
        if cfg.viz.max_oscillators_per_frame is not None:
            osc_indices = osc_indices[: int(cfg.viz.max_oscillators_per_frame)]

        # Build graph samples for these oscillators (on CPU), collate, and sample atoms.
        samples = [dataset[i] for i in osc_indices]
        batch = collate_graph_samples(samples)
        batch = _move_batch_to(batch, device)

        x_final_local, x_final_global = sample_atoms_full(
            model=model,
            diffusion=diffusion,
            batch=batch,
            timesteps=cfg.diffusion.timesteps,
            max_atom_radius=cfg.model.max_atom_radius,
            edge_cutoffs=edge_cutoffs,
            eps=cfg.loss.eps,
            init=cfg.infer.init,
        )

        # Build ground-truth and CG tables from raw oscillator dicts
        oscillators = [dataset.oscillators[i] for i in osc_indices]
        gt_atoms = aggregate_atomistic_from_oscillators(oscillators)
        cg = aggregate_cg_from_oscillators(oscillators)

        # Build predicted tables from oscillator samples + predicted global positions
        pred_atoms = aggregate_predicted_from_oscillator_predictions(samples, x_final_global.detach().cpu().numpy())

        out_pdb = out_dir / "viz" / f"epoch_{epoch:04d}" / split_name / f"{folder}_frame{frame:05d}.pdb"
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        write_multichain_pdb(
           out_path=str(out_pdb),
           atoms_A=gt_atoms,
           atoms_B=pred_atoms,
           beads_C=cg,
        )

def _format_loss_dict(d: dict, keys: list[str] | None = None) -> str:
    """
    Format a dict of loss scalars into a compact, stable log string.

    We keep ordering stable so logs are easy to diff/grep.
    """
    if d is None:
        return ""
    if keys is None:
        # Default order: total first, then major components.
        keys = ["total", "denoise_cart", "denoise_sph", "bond", "angle", "dihedral", "dipole", "contact"]

    parts = []
    for k in keys:
        if k in d:
            try:
                parts.append(f"{k}={float(d[k]):.4f}")
            except Exception:
                # If a value isn't directly float-able, just skip it
                continue
    # Also include any extra keys (sorted) that weren't in the preferred list
    extra = sorted([k for k in d.keys() if k not in keys])
    for k in extra:
        try:
            parts.append(f"{k}={float(d[k]):.4f}")
        except Exception:
            continue
    return " ".join(parts)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--pickle", type=str, default=None, help="Override data.pickle_path")
    ap.add_argument("--out", type=str, default=None, help="Override train.out_dir")
    ap.add_argument("--device", type=str, default=None, help="Override train.device")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    ap.add_argument("--batch_size", type=int, default=None, help="Override train.batch_size")
    ap.add_argument("--lr", type=float, default=None, help="Override train.lr")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()

    # Apply common overrides
    if args.pickle is not None:
        cfg = cfg.replace(data={"pickle_path": args.pickle})
    if args.out is not None:
        cfg = cfg.replace(train={"out_dir": args.out})
    if args.device is not None:
        cfg = cfg.replace(train={"device": args.device})
    if args.epochs is not None:
        cfg = cfg.replace(train={"epochs": int(args.epochs)})
    if args.batch_size is not None:
        cfg = cfg.replace(train={"batch_size": int(args.batch_size)})
    if args.lr is not None:
        cfg = cfg.replace(train={"lr": float(args.lr)})

    run_dir = Path(cfg.train.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(run_dir), name="train")
    seed_all(cfg.train.seed)

    # Save config snapshot for reproducibility
    cfg.save_json(run_dir / "config.json")
    if args.config:
        # also keep the original YAML (if any)
        try:
            import shutil

            shutil.copy(args.config, run_dir / "config.yaml")
        except Exception:
            pass

    if not cfg.data.pickle_path:
        raise SystemExit("Config is missing data.pickle_path")

    logger.info(f"Loading vocab from: {cfg.data.pickle_path}")
    vocab = build_default_vocab_from_pickle(cfg.data.pickle_path)
    logger.info(
        f"Vocab sizes: resnames={vocab.num_resnames}, names={vocab.num_names}, atom_groups={vocab.num_atom_groups}"
    )

    # Dataset
    ds_cfg = DatasetConfig(
        drop_zero_atoms=cfg.data.drop_zero_atoms,
        max_sidechain_beads=cfg.data.max_sidechain_beads,
        fully_connected_edges=cfg.data.fully_connected_edges,
        max_oscillators=cfg.data.max_oscillators,
        include_sidechains=bool(cfg.data.include_sidechains),
    )
    dataset = OscillatorDataset(
        pickle_path=cfg.data.pickle_path,
        vocab=vocab,
        cfg=ds_cfg,
        device="cpu",
        dtype=torch.float32 if cfg.data.dtype == "float32" else torch.float64,
    )
    logger.info(f"Dataset size: {len(dataset)} oscillators")

    split = split_indices(
        folders_by_index=dataset.folders(),
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.data.split_seed,
        split_by=cfg.data.split_by,
        min_items_per_split=cfg.data.min_items_per_split,
    )

    train_set = Subset(dataset, split.train_indices)
    val_set = Subset(dataset, split.val_indices)
    test_set = Subset(dataset, split.test_indices)

    logger.info(
        f"Split sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)} (split_by={cfg.data.split_by})"
    )

    # Loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_graph_samples,
    )

    device = _device(cfg.train.device)
    logger.info(f"Device: {device}")

    model = _build_model(cfg, vocab, device)
    diffusion = _build_diffusion(cfg, device)

    edge_cutoffs = EdgeCutoffs(bead_bead=cfg.model.bead_edge_cutoff, atom_any=cfg.model.atom_edge_cutoff)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        payload = load_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        start_epoch = int(payload.get("epoch") or 0) + 1
        best_val = float(payload.get("extra", {}).get("best_val", best_val))

    metrics_writer = JsonlWriter(run_dir / "metrics.jsonl")

    # Main loop
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        sum_total = 0.0
        sum_den_cart = 0.0
        sum_den_sph = 0.0
        sum_bond = 0.0
        sum_angle = 0.0
        sum_dihed = 0.0
        sum_dip = 0.0
        sum_contact = 0.0
        n_batches = 0

        n_bad_batches = 0
        n_bad_grads = 0

        for step, batch in enumerate(train_loader):
            batch = _move_batch_to(batch, device)

            optimizer.zero_grad(set_to_none=True)

            total, breakdown, _ = _forward_denoise_step(
                model=model,
                diffusion=diffusion,
                batch=batch,
                edge_cutoffs=edge_cutoffs,
                cfg=cfg,
            )

            # If this batch produced any NaNs/Infs in losses or inputs, DO NOT backprop.
            if getattr(breakdown, "bad", False) or (not torch.isfinite(total).item()):
                n_bad_batches += 1
                if n_bad_batches <= 10 or (n_bad_batches % 100 == 0):
                    # Print a small amount of metadata to help locate the offending data.
                    folder = batch.get("meta_folder", ["?"])
                    frame = batch.get("meta_frame", ["?"])
                    osc_i = batch.get("meta_oscillator_index", ["?"])
                    logger.warning(
                        f"[SKIP] non-finite batch at epoch {epoch:04d} step {step:05d} "
                        f"bad_reason={getattr(breakdown, 'bad_reason', '')} "
                        f"example_meta=(folder={folder[0]}, frame={frame[0]}, osc={osc_i[0]})"
                    )
                continue

            total.backward()

            grad_norm = None
            if cfg.train.grad_clip_norm is not None and float(cfg.train.grad_clip_norm) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))

            # Skip step if gradients are non-finite (prevents permanently poisoning weights)
            if grad_norm is not None and (not torch.isfinite(grad_norm).item()):
                n_bad_grads += 1
                if n_bad_grads <= 10 or (n_bad_grads % 100 == 0):
                    logger.warning(
                        f"[SKIP] non-finite grad_norm at epoch {epoch:04d} step {step:05d}: {grad_norm}"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            # Optional: detect if parameters became non-finite (abort fast)
            if (step + 1) % int(cfg.train.log_every_steps) == 0:
                any_bad_param = False
                for p in model.parameters():
                    if not torch.isfinite(p).all().item():
                        any_bad_param = True
                        break
                if any_bad_param:
                    raise RuntimeError(
                        "Model parameters became non-finite (NaN/Inf). "
                        "This usually indicates a bad batch with NaNs/Infs or an instability. "
                        "Enable debug data validation and/or lower LR."
                    )

            sum_total += float(breakdown.total.detach().cpu())
            sum_den_cart += float(breakdown.denoise_cart.detach().cpu())
            sum_den_sph += float(breakdown.denoise_sph.detach().cpu())
            sum_bond += float(breakdown.bond.detach().cpu())
            sum_angle += float(breakdown.angle.detach().cpu())
            sum_dihed += float(breakdown.dihedral.detach().cpu())
            sum_dip += float(breakdown.dipole.detach().cpu())
            sum_contact += float(breakdown.contact.detach().cpu())
            n_batches += 1

            if (step + 1) % int(cfg.train.log_every_steps) == 0:
                logger.info(
                    f"epoch {epoch:04d} step {step:05d} "
                    f"total={sum_total/n_batches:.4f} den={sum_den_cart/n_batches:.4f} "
                    f"bond={sum_bond/n_batches:.4f} angle={sum_angle/n_batches:.4f} "
                    f"dihed={sum_dihed/n_batches:.4f} dip={sum_dip/n_batches:.4f} contact={sum_contact/n_batches:.4f}"
                )

        train_means = {
            "total": sum_total / max(n_batches, 1),
            "denoise_cart": sum_den_cart / max(n_batches, 1),
            "denoise_sph": sum_den_sph / max(n_batches, 1),
            "bond": sum_bond / max(n_batches, 1),
            "angle": sum_angle / max(n_batches, 1),
            "dihedral": sum_dihed / max(n_batches, 1),
            "dipole": sum_dip / max(n_batches, 1),
            "contact": sum_contact / max(n_batches, 1),
        }

        # Evaluate train/val/test each epoch as requested.
        val_means, val_metrics = _evaluate(
            loader=val_loader,
            model=model,
            diffusion=diffusion,
            cfg=cfg,
            device=device,
            edge_cutoffs=edge_cutoffs,
            max_batches_for_metrics=cfg.plot.max_eval_batches,
        )
 

        for k, v in val_metrics.items():
           if isinstance(v, np.ndarray) and v.size > 0 and not np.any(np.isfinite(v)):
              logger.warning(f"[VAL] metric '{k}' is all non-finite at epoch {epoch}")


        test_means, test_metrics = _evaluate(
            loader=test_loader,
            model=model,
            diffusion=diffusion,
            cfg=cfg,
            device=device,
            edge_cutoffs=edge_cutoffs,
            max_batches_for_metrics=cfg.plot.max_eval_batches,
        )


        for k, v in test_metrics.items():
           if isinstance(v, np.ndarray) and v.size > 0 and not np.any(np.isfinite(v)):
              logger.warning(f"[TEST] metric '{k}' is all non-finite at epoch {epoch}")

        logger.info(f"epoch {epoch:04d} DONE | TRAIN {_format_loss_dict(train_means)}")
        logger.info(f"epoch {epoch:04d} DONE | VAL   {_format_loss_dict(val_means)}")
        logger.info(f"epoch {epoch:04d} DONE | TEST  {_format_loss_dict(test_means)}")

        # Write JSONL record
        record = {
            "epoch": epoch,
            "train": train_means,
            "val": val_means,
            "test": test_means,
        }
        metrics_writer.write(record)

        # Save checkpoint (last)
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=None,
            extra={"best_val": best_val},
        )

        # Save best
        val_total = float(val_means["total"]) if val_means["total"] == val_means["total"] else float("inf")
        if val_total < best_val:
            best_val = val_total
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=None,
                extra={"best_val": best_val},
            )
            logger.info(f"New best val_total={best_val:.6f} -> saved checkpoints/best.pt")

        # Plots
        if cfg.plot.enable and ((epoch + 1) % int(cfg.plot.every_epochs) == 0):
            try:
                plot_epoch_metrics(out_dir=run_dir, epoch=epoch, split="val", losses=val_means, metrics=val_metrics, cfg=cfg.plot)
                plot_epoch_metrics(out_dir=run_dir, epoch=epoch, split="test", losses=test_means, metrics=test_metrics, cfg=cfg.plot)
            except Exception:
                logger.exception(f"Plotting failed at epoch {epoch:04d} (split=val/test); continuing training.")

        # VMD overlays (random frames)
        if cfg.viz.enable and ((epoch + 1) % int(cfg.viz.every_epochs) == 0):
            if cfg.viz.split == "train":
                idx = split.train_indices
            elif cfg.viz.split == "test":
                idx = split.test_indices
            else:
                idx = split.val_indices
            _write_random_frame_overlays(
                out_dir=run_dir,
                epoch=epoch,
                cfg=cfg,
                dataset=dataset,
                model=model,
                diffusion=diffusion,
                device=device,
                split_name=cfg.viz.split,
                indices=idx,
                max_frames=int(cfg.viz.frames_per_epoch),
                edge_cutoffs=edge_cutoffs,
            )

    logger.info("Training finished")


if __name__ == "__main__":
    main()
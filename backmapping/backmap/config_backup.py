from __future__ import annotations

"""Configuration system.

The original `backmap_diffusion.zip` project used frozen dataclasses for config.
This production version keeps that style, and adds:

- YAML config loading (recommended for production runs)
- Nested overrides from CLI for common hyperparameters
- Additional knobs for plotting and visualization outputs

All config values are intentionally simple Python scalars so they serialize cleanly.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    # Path to the amino_acid_baskets pickle
    pickle_path: str = ""

    # Dataset -> graph conversion
    drop_zero_atoms: bool = True
    max_sidechain_beads: int = 4
    fully_connected_edges: bool = True

    # If False, ignore sidechain oscillators and train on backbone only.
    include_sidechains: bool = True

    # Optional truncation (debug). If set, keep only the first N oscillators
    # after deterministic sorting.
    max_oscillators: Optional[int] = None

    # Splits
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 123
    split_by: str = "folder"  # "folder" (recommended) or "random" (debug)
    min_items_per_split: int = 1

    # Float dtype used for coordinates
    dtype: str = "float32"


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 200
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Sampling
    clip_radius_each_step: bool = True
    sample_init: str = "gaussian"  # "gaussian" or "uniform_ball"


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.0

    # Embeddings
    time_embed_dim: int = 128
    pos_embed_dim: int = 64

    # Edge encodings
    rbf_num_centers: int = 16
    rbf_max_dist: float = 20.0

    # Geometric caps / neighborhood cutoffs (Å)
    max_atom_radius: float = 6.0
    bead_edge_cutoff: float = 10.0
    atom_edge_cutoff: float = 10.0


@dataclass(frozen=True)
class LossConfig:
    """Loss weights and stability settings.

    The model predicts atom coordinates in the **local residue frame**.

    - Denoising is computed in local coordinates.
    - Bond/angle/dihedral/dipole/contact are computed in global coordinates.
    """

    # Denoising losses
    w_denoise_cart: float = 1.0
    w_denoise_sph: float = 0.0

    # Geometry losses
    w_bond: float = 0.1
    w_angle: float = 0.05
    w_dihedral: float = 0.05
    w_dipole: float = 0.05

    # Close-contact penalty (keeps training stable; also useful during sampling eval)
    w_contact: float = 0.02
    contact_r0: float = 1.1  # Å; distances below this are penalized

    # Global stability
    max_term_value: float = 1e3
    eps: float = 1e-8
    r_min: float = 0.5  # clamp distances to at least this when computing energies


@dataclass(frozen=True)
class PlotConfig:
    enable: bool = True

    # Generate plots every N epochs
    every_epochs: int = 1

    # For expensive per-epoch distribution plots, you can cap how many eval
    # batches are used.
    max_eval_batches: int = 50

    # Histogram bins
    bins: int = 60


@dataclass(frozen=True)
class VizConfig:
    """Write VMD-friendly PDB overlays periodically."""

    enable: bool = True
    every_epochs: int = 1

    # How many random (folder,frame) structures to write per epoch.
    frames_per_epoch: int = 2

    # To keep runtime under control on large frames, optionally cap the number
    # of oscillators per frame during visualization sampling.
    max_oscillators_per_frame: Optional[int] = None

    # Which split to draw visualization frames from: "train", "val", or "test".
    split: str = "val"


@dataclass(frozen=True)
class TrainConfig:
    out_dir: str = "runs/backmap_diffusion"
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0

    # Performance
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "cuda"  # "cuda" or "cpu"

    # Logging / evaluation
    log_every_steps: int = 50
    val_every_epochs: int = 1
    test_every_epochs: int = 1
    save_every_epochs: int = 1

    # Validation mode:
    #   - "loss": denoising objective at random t (fast)
    #   - "sample": full reverse diffusion (slow but matches inference)
    val_mode: str = "loss"

    # If val_mode=="sample", how many batches to sample.
    val_sample_batches: int = 2

    # Mixed precision (CUDA only). If enabled on CPU, it is ignored.
    amp: bool = False

    # Reproducibility
    seed: int = 1


@dataclass(frozen=True)
class InferConfig:
    """Default knobs for inference/sampling."""

    batch_size: int = 32
    init: str = "gaussian"  # "gaussian" or "uniform_ball"


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    plot: PlotConfig = PlotConfig()
    viz: VizConfig = VizConfig()
    train: TrainConfig = TrainConfig()
    infer: InferConfig = InferConfig()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML is not available; cannot save YAML")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def replace(self, **updates: Dict[str, Any]) -> "Config":
        """Return a new Config with partial overrides.

        This is a convenience used by the CLI scripts. Example:

            cfg = cfg.replace(train={"epochs": 10, "batch_size": 32})

        Unknown keys are ignored (consistent with :meth:`from_dict`).
        """

        base = self.to_dict()

        def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_update(dst[k], v)
                else:
                    dst[k] = v
            return dst

        _deep_update(base, updates)
        return Config.from_dict(base)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        """Create Config from a nested dict.

        Unknown keys are ignored, which makes it resilient to older config files.
        """

        def _sub(cls, key: str):
            sub = d.get(key, {}) if isinstance(d, dict) else {}
            if not isinstance(sub, dict):
                sub = {}
            # Filter unknown keys
            valid = {k: v for k, v in sub.items() if k in cls.__dataclass_fields__}
            return cls(**valid)

        return Config(
            data=_sub(DataConfig, "data"),
            diffusion=_sub(DiffusionConfig, "diffusion"),
            model=_sub(ModelConfig, "model"),
            loss=_sub(LossConfig, "loss"),
            plot=_sub(PlotConfig, "plot"),
            viz=_sub(VizConfig, "viz"),
            train=_sub(TrainConfig, "train"),
            infer=_sub(InferConfig, "infer"),
        )

    @staticmethod
    def load(path: str | Path) -> "Config":
        """Load config from YAML or JSON."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        if p.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is not available; cannot load YAML")
            with p.open("r") as f:
                d = yaml.safe_load(f) or {}
            if not isinstance(d, dict):
                raise ValueError("YAML config must be a mapping")
            return Config.from_dict(d)

        if p.suffix.lower() == ".json":
            with p.open("r") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("JSON config must be an object")
            return Config.from_dict(d)

        raise ValueError(f"Unknown config extension: {p.suffix}")

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        """Backwards-compatible alias.

        Many earlier scripts used `Config.from_yaml(...)`. We keep the name
        so existing command lines keep working.
        """
        return Config.load(path)

def device_from_config(device_str: str) -> "torch.device":
    """Helper that resolves 'cuda' -> torch.device('cuda') only if available."""

    import torch

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# CG-to-Atomistic Backmapping with Diffusion Models

A diffusion-based deep learning framework for reconstructing atomistic protein structures from coarse-grained (CG) representations. The model learns to predict peptide backbone oscillator atoms (C=O bonds) and sidechain amide groups from CG bead positions.

## Overview

This codebase provides:
1. **Data extraction** from MD trajectories (CG + atomistic)
2. **Training** a diffusion-based GNN model
3. **Inference** on new CG trajectories
4. **Validation plotting** for publication-quality figures

## Installation

### Dependencies

```bash
# Core ML
pip install torch torchvision
pip install torch-geometric

# Molecular dynamics
pip install MDAnalysis

# Scientific computing
pip install numpy scipy

# Utilities
pip install pyyaml tqdm matplotlib seaborn
```

### Setup

Clone and navigate to the repository:
```bash
cd backmapping
```

No installation required - scripts add the package to path automatically.

---

## Quick Start

```bash
# 1. Extract training data
python extract_cg_atomistic_data_basket.py --config config-basket.yaml

# 2. Train model
python scripts/train.py --config train_example.yaml

# 3. Run inference
python inference.py --config config_inference_traj.yaml --checkpoint best.pt

# 4. Generate plots
python plot_backmap_validation.py --root denoise_results_new --output-dir plots
```

---

## Data Extraction For Training

### Input Data Structure

Organize your protein folders as follows:

```
base_directory/
├── protein_A/
│   ├── prod.tpr                                    # GROMACS topology
│   ├── prod_centered.xtc                           # Atomistic trajectory
│   ├── protein_A_CG_protein_only_full_trajectory.pdb   # CG trajectory
│   └── diagonal_hamiltonian.txt                    # Vibrational data
├── protein_B/
│   ├── prod.tpr
│   ├── prod_centered.xtc
│   ├── protein_B_CG_protein_only_full_trajectory.pdb
│   └── diagonal_hamiltonian.txt
└── ...
```

**File naming convention:**
- CG PDB files must follow the pattern: `{folder_name}_CG_protein_only_full_trajectory.pdb`
- This pattern is configurable via `cg_pdb_file_pattern` in the config

### Configuration (config-basket.yaml)

```yaml
base_directory: "."                    # Root directory containing protein folders
frames_to_process: 500                 # Number of frames to sample (null = all)
frame_selection_method: "random"       # Options: random, sequential, stride
cpu_cores: 8                           # Parallel processing cores
output_filename: "amino_acid_baskets_complete_500.pkl"

# File patterns
atomistic_trajectory_file: "prod_centered.xtc"
atomistic_topology_file: "prod.tpr"
cg_pdb_file_pattern: "{folder}_CG_protein_only_full_trajectory.pdb"

strict_validation: true                # Enforce consistency checks
```

### Running Extraction

```bash
# Using config file
python extract_cg_atomistic_data_basket.py --config config-basket.yaml

# With command-line overrides
python extract_cg_atomistic_data_basket.py --config config-basket.yaml \
    --frames 1000 --cpu 16 --output my_data.pkl
```

**Command-line options:**
- `--config`: Path to YAML config file
- `--output`: Override output filename
- `--frames`: Override number of frames
- `--cpu`: Override CPU cores
- `--overwrite`: Overwrite existing output
- `--no-strict`: Disable strict validation
- `--debug-sample ALA`: Print sample oscillator data for debugging

### Output Format

The extraction produces a pickle file containing oscillators grouped by residue type:

```python
{
    "ALA": [oscillator1, oscillator2, ...],    # Backbone oscillators
    "GLY": [...],
    "GLN-SC": [...],                            # Sidechain oscillators (GLN amide)
    "ASN-SC": [...],                            # Sidechain oscillators (ASN amide)
    ...
}
```

Each oscillator dictionary contains:
- `atoms`: Atomistic coordinates (C, O, N, CA, H positions)
- `bb_curr`, `bb_next`: CG backbone bead positions
- `sc_beads`: Sidechain CG beads
- `rama_nnfs`: Ramachandran angles (phi/psi)
- `hamiltonian`: Vibrational frequency data
- `folder`, `frame`: Source information

---

## Training

### Configuration (train_example.yaml)

```yaml
data:
  pickle_path: "/path/to/amino_acid_baskets_complete_500.pkl"
  drop_zero_atoms: true
  max_sidechain_beads: 4
  fully_connected_edges: true
  include_sidechains: true

  # Train/val/test split
  train_frac: 0.8
  val_frac: 0.1
  test_frac: 0.1
  split_by: folder          # Prevents data leakage between proteins
  split_seed: 123

diffusion:
  timesteps: 200
  beta_schedule: cosine
  beta_start: 0.0001
  beta_end: 0.02

model:
  hidden_dim: 256
  num_layers: 6
  dropout: 0.0
  time_embed_dim: 128
  pos_embed_dim: 64
  rbf_num_centers: 16
  rbf_max_dist: 20.0
  max_atom_radius: 6.0      # Angstroms

loss:
  w_denoise_cart: 1.0       # Cartesian denoising
  w_denoise_sph: 0.7        # Spherical denoising
  w_bond: 0.7               # Bond length constraint
  w_angle: 0.7              # Bond angle constraint
  w_dihedral: 1.0           # Dihedral angle constraint
  w_dipole: 1.0             # Dipole moment constraint
  w_contact: 0.5            # Steric clash penalty

train:
  out_dir: runs/backmap_diffusion_prod
  epochs: 500
  batch_size: 16
  lr: 0.0001
  weight_decay: 1.0e-6
  grad_clip_norm: 1.0
  device: cuda
```

### Running Training

```bash
# Basic training
python scripts/train.py --config train_example.yaml

# With overrides
python scripts/train.py --config train_example.yaml \
    --out runs/experiment1 --epochs 100 --batch_size 32 --device cuda

# Resume from checkpoint
python scripts/train.py --config train_example.yaml --resume runs/experiment1/epoch_50.pt
```

**Command-line options:**
- `--config`: Path to YAML config
- `--pickle`: Override data path
- `--out`: Override output directory
- `--device`: Override device (cuda/cpu)
- `--epochs`: Override epoch count
- `--batch_size`: Override batch size
- `--lr`: Override learning rate
- `--resume`: Resume from checkpoint

### Training Output

```
runs/backmap_diffusion_prod/
├── epoch_0.pt              # Checkpoints
├── epoch_1.pt
├── ...
├── best.pt                 # Best validation checkpoint
├── log.jsonl               # Training logs
└── plots_epoch_N/          # Periodic validation plots
```

---

## Inference

### Configuration (inference/config_inference_traj.yaml)

```yaml
# Must match training config
model:
  hidden_dim: 256
  num_layers: 6
  # ... (copy from training config)

diffusion:
  timesteps: 200
  beta_schedule: cosine
  # ... (copy from training config)

infer:
  protein_name: "3fp5_A"
  protein_folder: "/path/to/3fp5_A"

  # Input files
  xtc_filename: "prod_centered.xtc"
  tpr_filename: "prod.tpr"
  cg_pdb_pattern: "{folder}_CG_protein_only_full_trajectory.pdb"

  # Processing
  max_frames: 5000
  frame_selection: first     # Options: first, last, range, random
  extract_atomistic: true    # Extract ground truth (slower)

  # Sampling
  ddim_steps: 50             # DDIM steps (20-100, lower=faster)
  use_fp16: true             # Mixed precision
  batch_size: 4126
  init: "uniform_ball"

  # Analysis
  compute_analysis: true              # Predict Ramachandran & dipole
  compute_atomistic_analysis: true    # Ground truth comparison

  # Output
  output: "denoise_results_new/3fp5_A.pkl"

  # Vocabulary (any pickle with same residue types)
  vocab_pickle: "/path/to/training_data.pkl"
```

### Running Inference

```bash
python inference.py \
    --config config_inference_traj.yaml \
    --checkpoint best.pt
```

**Command-line options:**
- `--config`: Path to inference config
- `--checkpoint`: Path to trained model checkpoint
- `--output`: Override output path

### Output Format

The inference produces pickle files with the same structure as training data, plus predicted fields:

```python
{
    "ALA": [
        {
            "predicted_atoms": {...},           # Model predictions
            "predicted_rama_nnfs": {...},       # Predicted phi/psi
            "predicted_dipole": np.array(),     # Predicted C=O dipole
            "atoms": {...},                      # Ground truth (if extract_atomistic=true)
            "rama_nnfs": {...},                  # Ground truth angles
            "atomistic_dipole": np.array(),     # Ground truth dipole
            "folder": "3fp5_A",
            "frame": 0,
            ...
        },
        ...
    ]
}
```

---

## Plotting & Analysis

Generate publication-quality validation plots from inference results.

### Usage

```bash
# Basic usage - analyze all pkl files in split folders
python plot_backmap_validation.py \
    --root denoise_results_new \
    --output-dir plots

# Specify splits and options
python plot_backmap_validation.py \
    --root denoise_results_new \
    --splits val test \
    --output-dir plots \
    --dpi 300 \
    --font-size 10

# Include advanced metrics
python plot_backmap_validation.py \
    --root denoise_results_new \
    --output-dir plots \
    --advanced-metrics
```

**Command-line options:**
- `--root`: Root directory containing pkl files or split folders
- `--splits`: Split folders to analyze (default: val test)
- `--pattern`: Glob pattern for pkl files (default: *.pkl)
- `--output-dir`: Output directory for plots
- `--dpi`: Figure resolution (default: 300)
- `--font-size`: Base font size (default: 10)
- `--advanced-metrics`: Compute TM-score, GDT-TS, lDDT, etc.

### Generated Plots

**Core plots (8):**
1. Ramachandran scatter (4-panel density)
2. Ramachandran angle distributions
3. Ramachandran maps (phi vs psi)
4. Bond length distributions (C-O, C-N, N-CA, CA-C)
5. Dipole orientation analysis
6. Dipole component correlations (x, y, z)
7. Per-frame all-atom RMSD
8. Summary metrics table

**Advanced metrics (with --advanced-metrics):**
- TM-score distribution
- GDT-TS distribution
- lDDT per-residue and distribution
- Contact map comparison
- Secondary structure Q3 accuracy
- Clash analysis

---

## Model Architecture

The model uses a graph neural network (GNN) with diffusion-based denoising:

- **Input**: Small graphs (~5-20 nodes) per oscillator containing CG beads + noised atom positions
- **Output**: Predicted atomistic coordinates in residue-local frame
- **Training**: Forward diffusion adds noise; model learns to denoise
- **Inference**: DDIM sampling for fast generation (50 steps typical)

Key design choices:
- SE(3)-equivariant predictions via local coordinate frames
- Physics-informed losses (bonds, angles, dihedrals, dipoles)
- Folder-based train/test splits prevent data leakage

---

## Citation

If you use this code, please cite the associated publication.

## License

[Add license information]

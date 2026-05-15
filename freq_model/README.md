# Site Frequency Perturbation Model

A physics-informed deep learning framework for predicting infrared (IR) spectra from coarse-grained (CG) molecular dynamics simulations of proteins. This codebase uses transformer-based neural networks to predict site energies for amide I vibrational modes, enabling fast and accurate spectral predictions from CG protein structures.

## Table of Contents

- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Running Inference with Pre-trained Model

```bash
python inference_spectra.py --config config_inference.json
```

Or specify paths directly:

```bash
python inference_spectra.py \
    --model best_model.pt \
    --data path/to/your/data.pkl \
    --output results/
```

### Training a New Model

```bash
python train/main.py \
    --train_dir path/to/train/data/ \
    --test_dir path/to/test/data/ \
    --output_dir checkpoints/ \
    --epochs 100 \
    --batch_size 8
```

## Data Format

### Input Data (PKL Files)

The input data should be pickle files of `dict` objects containing oscillator information organized by amino acid type:

```python
pkl_data = {
    'amino_acid_type': [
        {
            'frame': int,                    # Frame index
            'atoms': {                       # Atomistic (ground truth) coordinates
                'C_prev': [x, y, z],         # Carbonyl carbon
                'O_prev': [x, y, z],         # Carbonyl oxygen
                'N_curr': [x, y, z]          # Amide nitrogen
            },
            'predicted_atoms': {             # CG-predicted coordinates
                'C_prev': [x, y, z],
                'O_prev': [x, y, z],
                'N_curr': [x, y, z]
            },
            'oscillator_type': str,          # 'backbone' or 'sidechain'
            'residue_name': str,             # Three-letter residue code
            'charge': float,                 # Partial charge
            'rama_angles': [phi, psi, omega, theta],  # Dihedral angles
            'H_diag': float                  # Site energy (cm⁻¹) - training target
        },
        ...
    ]
}
```

## Training

### Basic Training

```bash
python train/main.py \
    --train_dir /path/to/training/pkl/files/ \
    --test_dir /path/to/test/pkl/files/ \
    --output_dir ./checkpoints/
```

### Full Training Options

```bash
python train/main.py \
    --train_dir /path/to/train/ \
    --test_dir /path/to/test/ \
    --output_dir ./checkpoints/ \
    --epochs 500 \
    --batch_size 8 \
    --lr 5e-5 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 6 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --cutoff 20.0 \
    --gamma 10.0 \
    --n_clusters 1600 \
    --samples_per_cluster 1
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 500 | Number of training epochs |
| `--batch_size` | 8 | Batch size (frames per batch) |
| `--lr` | 5e-5 | Learning rate |
| `--d_model` | 128 | Transformer hidden dimension |
| `--n_heads` | 8 | Number of attention heads |
| `--n_layers` | 6 | Number of transformer layers |
| `--cutoff` | 20.0 | Neighbor cutoff distance (Angstroms) |
| `--gamma` | 10.0 | Lorentzian broadening width (cm⁻¹) |
| `--n_clusters` | 1600 | Number of clusters for diverse sampling |
| `--dropout` | 0.1 | Dropout rate | 


## Inference

### Configuration File

Create a JSON configuration file (see `config_inference.json`):

```json
{
    "model_path": "best_model.pt",
    "data_path": "path/to/data.pkl",
    "output_dir": "results/",
    "model_config": {
        "own_feature_dim": 16,
        "neighbor_feature_dim": 18,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 6,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "max_neighbors": 80,
        "min_energy": 1450.0,
        "max_energy": 1850.0
    },
    "cutoff": 20.0,
    "max_neighbors": 80,
    "gamma": 10.0,
    "omega_min": 1500.0,
    "omega_max": 1750.0,
    "omega_step": 1.0,
    "use_cuda": true,
    "batch_size": 64
}
```

### Running Inference

```bash
# Using configuration file
python inference_spectra.py --config config_inference.json

# Override specific parameters
python inference_spectra.py \
    --config config_inference.json \
    --data new_data.pkl \
    --output new_results/
```


## Output Files

### Inference Outputs

| File | Description |
|------|-------------|
| `oscillator_info.txt` | Number of oscillators per frame |
| `hamiltonians_predicted.dat` | Predicted Hamiltonians (NISE format) |
| `hamiltonians_groundtruth.dat` | Ground truth Hamiltonians |
| `dipoles_predicted.dat` | Predicted dipole vectors |
| `dipoles_groundtruth.dat` | Ground truth dipole vectors |
| `spectra.dat` | Predicted and true spectra |
| `spectra_comparison.png` | Visual comparison plots |

### Hamiltonian File Format (NISE-compatible)

```
frame_index  H_11  H_12  H_13  ...  H_22  H_23  ...  H_NN
```

Upper triangular elements, row by row.

### Dipole File Format

```
frame_index  μx_1  μx_2  ...  μx_N  μy_1  μy_2  ...  μy_N  μz_1  μz_2  ...  μz_N
```

### Spectra File Format

```
frame_index  frequency  predicted_intensity  true_intensity
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or use `--use_cuda false`
2. **NaN in loss**: Check data quality; ensure site energies are within physical range
3. **Poor convergence**: Try adjusting learning rate or increasing `n_clusters` for more diverse training




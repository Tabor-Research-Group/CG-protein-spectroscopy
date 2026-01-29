# CG Spectroscopy: Neural Network Prediction of Amide I IR Spectra

A physics-informed deep learning framework for predicting infrared (IR) spectra from coarse-grained (CG) molecular dynamics simulations of proteins. This codebase uses transformer-based neural networks to predict site energies for amide I vibrational modes, enabling fast and accurate spectral predictions from CG protein structures.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Physics Background](#physics-background)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Citation](#citation)

## Overview

This project bridges coarse-grained molecular simulations with spectroscopic predictions. The neural network learns to predict amide I site energies (diagonal Hamiltonian elements) from local structural features, which are then combined with physics-based transition dipole coupling calculations to generate full IR spectra.

### Key Features

- **Transformer-based architecture** for processing variable numbers of oscillators per frame
- **Physics-informed loss function** combining spectral correlation, MSE, and gradient matching
- **Constrained output layer** ensuring physically meaningful site energies (1450-1850 cm⁻¹)
- **Batched differentiable physics calculations** for end-to-end training
- **NISE-compatible output format** for integration with spectral simulation packages

## Project Structure

```
spectra_code/
├── inference_spectra.py      # Main inference script
├── best_model.pt             # Pre-trained model weights
├── config_inference.json     # Inference configuration
├── README.md                 # This file
└── train/
    ├── main.py               # Main training entry point
    ├── train_optimized.py    # Optimized training loop with batched physics
    ├── model_fixed.py        # Constrained output model (recommended)
    ├── model.py              # Original model architecture
    ├── dataset.py            # PyTorch dataset and data loading
    ├── physics.py            # Physics calculations (Torii, Tasumi, spectra)
    ├── features.py           # Feature extraction utilities
    ├── data_utils.py         # Data loading and preprocessing
    ├── data_utils_lazy.py    # Memory-efficient data loading
    ├── clustering.py         # Spectrum-based frame clustering
    ├── evaluate.py           # Evaluation metrics and plotting
    ├── feature_importance.py # Feature analysis utilities
    └── plot_frame_comparison.py  # Visualization tools
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- scikit-learn
- Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd spectra_code

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib
```

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

The input data should be pickle files containing oscillator information organized by amino acid type:

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

### Supported Oscillator Types

- **Backbone amides**: Regular peptide bonds (C=O...N-H)
- **Proline backbone**: Special handling for proline residues
- **Sidechain amides**: ASN, GLN sidechain C=O groups

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
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-7 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 6 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --cutoff 20.0 \
    --gamma 10.0 \
    --n_clusters 50 \
    --samples_per_cluster 1
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 8 | Batch size (frames per batch) |
| `--lr` | 1e-7 | Learning rate |
| `--d_model` | 128 | Transformer hidden dimension |
| `--n_heads` | 8 | Number of attention heads |
| `--n_layers` | 6 | Number of transformer layers |
| `--cutoff` | 20.0 | Neighbor cutoff distance (Angstroms) |
| `--gamma` | 10.0 | Lorentzian broadening width (cm⁻¹) |
| `--n_clusters` | 50 | Number of clusters for diverse sampling |

### Loss Function

The training uses a three-component loss function:

```
Loss = λ_corr × (1 - Pearson_corr) + λ_mse × MSE + λ_grad × Gradient_MSE
```

- **Correlation Loss**: Ensures spectral shape similarity
- **MSE Loss**: Ensures intensity and frequency alignment
- **Gradient Loss**: Preserves peak positions and spectral structure

Default weights: `λ_corr = 1.0`, `λ_mse = 1.0`, `λ_grad = 500.0`

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

## Model Architecture

### Overview

The model uses a transformer encoder to predict site energies from local structural features:

```
Input Features (per oscillator)
         ↓
    Embedding Layer
         ↓
  Transformer Encoder (6 layers)
         ↓
  Constrained Output Head
         ↓
   Site Energies (cm⁻¹)
```

### Feature Extraction

**Own Features (16 dimensions per oscillator):**
- Oscillator type one-hot encoding (3)
- Ramachandran angles sin/cos (8)
- Secondary structure one-hot (4)
- Partial charge (1)

**Neighbor Features (18 dimensions per neighbor, up to 80 neighbors):**
- Distance feature (1/r³ scaled)
- Spherical angles sin/cos (4)
- Neighbor charge (1)
- Neighbor Ramachandran angles (8)
- Neighbor secondary structure (4)

### Constrained Output

The output layer uses sigmoid activation scaled to the physical range [1450, 1850] cm⁻¹:

```python
output = min_energy + sigmoid(x) × (max_energy - min_energy)
```

This prevents unphysical predictions that would cause numerical instability in spectrum generation.

## Physics Background

### Amide I Vibrational Mode

The amide I mode (1600-1700 cm⁻¹) is the primary infrared signature of protein secondary structure, arising from C=O stretching vibrations of the peptide backbone.

### Torii Dipole Model

Transition dipole moments are calculated using the Torii model:

```
μ = 0.276 × (s - projection correction)
```

where `s` is a combination of C=O and C-N unit vectors.

### Tasumi Transition Dipole Coupling

Inter-oscillator coupling is calculated using the transition dipole coupling (TDC) model:

```
J_ij = 5034 × (μ_i · μ_j / r³ - 3(μ_i · r)(μ_j · r) / r⁵)
```

### Spectrum Generation

1. Construct Hamiltonian: `H = diag(site_energies) + J`
2. Diagonalize: `eigenvalues, eigenvectors = eigh(H)`
3. Calculate transition strengths from eigenvector-weighted dipoles
4. Apply Lorentzian broadening

## Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `own_feature_dim` | 16 | Own feature vector dimension |
| `neighbor_feature_dim` | 18 | Neighbor feature vector dimension |
| `d_model` | 128 | Transformer hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of transformer encoder layers |
| `dim_feedforward` | 512 | Feed-forward network dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_neighbors` | 80 | Maximum neighbors per oscillator |
| `min_energy` | 1450.0 | Minimum output energy (cm⁻¹) |
| `max_energy` | 1850.0 | Maximum output energy (cm⁻¹) |

### Physics Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cutoff` | 20.0 | Neighbor cutoff distance (Angstroms) |
| `gamma` | 10.0 | Lorentzian broadening HWHM (cm⁻¹) |
| `omega_min` | 1500.0 | Spectrum start frequency (cm⁻¹) |
| `omega_max` | 1750.0 | Spectrum end frequency (cm⁻¹) |
| `omega_step` | 1.0 | Spectrum frequency resolution (cm⁻¹) |

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

## Evaluation Metrics

The following metrics are computed during training and inference:

- **Spectrum MSE**: Mean squared error between predicted and true spectra
- **Spectrum Correlation**: Pearson correlation coefficient
- **Peak Position Error**: Difference in peak maxima positions (cm⁻¹)
- **Site Energy MAE**: Mean absolute error of site energies (cm⁻¹)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or use `--use_cuda false`
2. **NaN in loss**: Check data quality; ensure site energies are within physical range
3. **Poor convergence**: Try adjusting learning rate or increasing `n_clusters` for more diverse training

### Data Quality Checks

The pipeline includes automatic filtering for:
- Invalid bond lengths (C-O, C-N distances)
- Missing coordinate data
- Frames with too few oscillators

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
    title={Neural Network Prediction of Amide I IR Spectra from Coarse-Grained Protein Structures},
    author={Your Name et al.},
    journal={Journal Name},
    year={Year}
}
```

## License

[Add your license information here]

## Contact

[Add contact information here]

#!/usr/bin/env python3
"""
Main training script for site energy prediction.

Usage:
    python main.py --train_pkl 1fd3_A.pkl --test_pkl 3lpe_B.pkl --output_dir results/
"""

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .data_utils import load_pkl_data, print_data_summary, organize_by_frames, filter_frames_by_quality, load_pkl_from_directory, load_individual_files_from_directory, extract_ground_truth_data, filter_frames_by_quality
from .data_utils_lazy import load_pkl_sampled, estimate_memory_usage
from .clustering import generate_and_cluster_spectra, plot_cluster_summary
from .dataset import create_dataloaders
from .model import create_model
from .train_optimized import SpectrumLoss, train_one_epoch, evaluate, save_checkpoint
from .evaluate import (
    plot_training_curves,
    plot_spectra_comparison,
    plot_site_energy_comparison,
    plot_site_energy_distribution,
    plot_average_spectrum,
    save_metrics_summary
)
from .plot_frame_comparison import plot_individual_frames_detailed, plot_spectra_grid_comparison
from .feature_importance import run_all_analyses
from .extended_plots import generate_all_extended_plots


def parse_args():
    parser = argparse.ArgumentParser(description='Train site energy prediction model')

    # Data
    data_group = parser.add_argument_group('Data Input/Output')
    data_group.add_argument('--train_dir', type=str, help='Training data directory (contains *.pkl files)')
    data_group.add_argument('--val_dir', type=str, help='Validation data directory (contains *.pkl files)')
    data_group.add_argument('--test_dir', type=str, help='Test data directory (contains *.pkl files)')
    data_group.add_argument('--train_pkl', type=str, help='Training PKL file (alternative to --train_dir)')
    data_group.add_argument('--test_pkl', type=str, help='Test PKL file (alternative to --val_dir/--test_dir)')
    data_group.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    data_group.add_argument('--per_protein_eval', action='store_true', help='Evaluate each protein individually after training')

    # Clustering
    cluster_group = parser.add_argument_group('Sample Clustering')
    cluster_group.add_argument('--n_clusters', type=int, default=1600, help='Number of clusters')
    cluster_group.add_argument('--samples_per_cluster', type=int, default=1, help='Samples per cluster')
    cluster_group.add_argument('--use_all_test_frames', action='store_true', help='Use all test frames (no clustering)')

    # Model
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--d_model', type=int, default=128, help='Transformer hidden dimension')
    model_group.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    model_group.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    model_group.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    model_group.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    training_group.add_argument('--batch_size', type=int, default=8, help='Batch size')
    training_group.add_argument('--lr', type=float, default=5e-5, help='Learning rate (constant, very low to prevent overshooting)')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Loss (multi-component spectrum loss)
    loss_group = parser.add_argument_group('Loss Function Parameters')
    loss_group.add_argument('--lambda_peak', type=float, default=0.5, help='Peak position loss weight')
    loss_group.add_argument('--lambda_correlation', type=float, default=0.3, help='Correlation loss weight')

    # Physics
    physics_group = parser.add_argument_group('Physics-Based Parameters')
    physics_group.add_argument('--cutoff', type=float, default=20.0, help='Neighbor cutoff (Angstroms)')
    physics_group.add_argument('--max_neighbors', type=int, default=80, help='Max neighbors per oscillator')
    physics_group.add_argument('--gamma', type=float, default=10.0, help='Lorentzian width (cm^-1)')

    # Early stopping
    es_group = parser.add_argument_group('Early Stopping Parameters')
    es_group.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    es_group.add_argument('--patience', type=int, default=100, help='Early stopping patience (epochs)')
    es_group.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')

    # Per-protein tracking
    tracking_group = parser.add_argument_group('Per-Protein Metric Settings')
    tracking_group.add_argument('--track_per_protein', action='store_true', help='Track per-protein metrics during training')
    tracking_group.add_argument('--plot_interval', type=int, default=10, help='Interval for updating tracking plots (epochs)')

    # Memory efficiency
    mem_group = parser.add_argument_group('Memory Efficiency')
    mem_group.add_argument('--frames_per_file', type=int, default=None,
                        help='Sample N frames per file (for low-memory systems). If None, loads all frames.')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*80)
    print("SITE ENERGY PREDICTION FROM CG STRUCTURES")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    # Determine if using directories or individual files
    val_files_data = None
    test_files_data = None

    if args.train_dir:
        # Use memory-efficient loading if frames_per_file is specified
        if args.frames_per_file is not None:
            print(f"\n⚠️  Memory-efficient mode: sampling {args.frames_per_file} frames per file")
            train_data, train_file_mapping = load_pkl_sampled(
                args.train_dir,
                frames_per_file=args.frames_per_file,
                seed=args.seed
            )
        else:
            train_data, train_file_mapping = load_pkl_from_directory(args.train_dir)
    elif args.train_pkl:
        train_data = load_pkl_data(args.train_pkl)
    else:
        raise ValueError("Must specify either --train_dir or --train_pkl")

    if args.val_dir:
        # Use memory-efficient loading if frames_per_file is specified
        if args.frames_per_file is not None:
            test_data, val_file_mapping = load_pkl_sampled(
                args.val_dir,
                frames_per_file=args.frames_per_file,
                seed=args.seed
            )
        else:
            test_data, val_file_mapping = load_pkl_from_directory(args.val_dir)
        # Load individual files for per-protein evaluation
        if args.per_protein_eval:
            val_files_data = load_individual_files_from_directory(args.val_dir)
    elif args.test_dir:
        test_data, test_file_mapping = load_pkl_from_directory(args.test_dir)
        # Load individual files for per-protein evaluation
        if args.per_protein_eval:
            test_files_data = load_individual_files_from_directory(args.test_dir)
    elif args.test_pkl:
        test_data = load_pkl_data(args.test_pkl)
    else:
        raise ValueError("Must specify either --val_dir, --test_dir, or --test_pkl")

    print_data_summary(train_data, name="Training Data")
    print_data_summary(test_data, name="Test/Validation Data")

    # Organize by frames
    train_frames_dict = organize_by_frames(train_data)
    test_frames_dict = organize_by_frames(test_data)

    # Filter frames by quality (v9 addition)
    print("="*80)
    print("FILTERING FRAMES BY QUALITY")
    print("="*80)

    train_frames_dict, train_excluded, train_stats = filter_frames_by_quality(
        train_frames_dict, min_bond=0.8, max_bond=2.0, verbose=True
    )
    test_frames_dict, test_excluded, test_stats = filter_frames_by_quality(
        test_frames_dict, min_bond=0.8, max_bond=2.0, verbose=True
    )

    print(f"\nFiltering complete - Train: {len(train_frames_dict)} valid frames, Test: {len(test_frames_dict)} valid frames")

    # Cluster and sample training frames
    print("="*80)
    print("CLUSTERING TRAINING SPECTRA")
    print("="*80)

    # Cluster and sample (returns labels to avoid re-clustering)
    train_frame_indices, train_all_spectra, omega_grid, labels, sampled_array_indices = generate_and_cluster_spectra(
        train_frames_dict,
        n_clusters=args.n_clusters,
        samples_per_cluster=args.samples_per_cluster,
        gamma=args.gamma,
        random_state=args.seed
    )

    # Plot clusters
    plot_cluster_summary(
        train_all_spectra,
        omega_grid,
        labels=labels,
        sampled_indices=np.array(sampled_array_indices),
        save_path=str(output_dir / 'train_clusters.png')
    )

    # Test frames (use all or cluster)
    if args.use_all_test_frames:
        test_frame_indices = sorted(test_frames_dict.keys())
        print(f"\nUsing all {len(test_frame_indices)} test frames")
    else:
        print("\n" + "="*80)
        print("CLUSTERING TEST SPECTRA")
        print("="*80)
        # Unpack all 5 return values (even if we don't use labels/indices for test)
        test_frame_indices, test_all_spectra, _, _, _ = generate_and_cluster_spectra(
            test_frames_dict,
            n_clusters=args.n_clusters // 2,  # Fewer clusters for test
            samples_per_cluster=1,
            gamma=args.gamma,
            random_state=args.seed
        )

    print(f"\nFinal dataset sizes:")
    print(f"  Training frames: {len(train_frame_indices)}")
    print(f"  Test frames: {len(test_frame_indices)}")

    # Create dataloaders
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)

    train_loader, test_loader = create_dataloaders(
        train_data, test_data,
        train_frame_indices, test_frame_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        gamma=args.gamma
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Analyze ground truth energy range
    print("\n" + "="*80)
    print("GROUND TRUTH SITE ENERGY ANALYSIS")
    print("="*80)

    all_train_energies = []
    all_test_energies = []

    for frame_idx in train_frame_indices[:100]:  # Sample first 100 training frames
        if frame_idx in train_frames_dict:
            gt = extract_ground_truth_data(train_frames_dict[frame_idx])
            all_train_energies.extend(gt['H_diag'].tolist())

    for frame_idx in test_frame_indices[:100]:  # Sample first 100 test frames
        if frame_idx in test_frames_dict:
            gt = extract_ground_truth_data(test_frames_dict[frame_idx])
            all_test_energies.extend(gt['H_diag'].tolist())

    print(f"\nTraining Data (sampled {len(all_train_energies)} oscillators):")
    print(f"  Min energy: {np.min(all_train_energies):.1f} cm⁻¹")
    print(f"  Max energy: {np.max(all_train_energies):.1f} cm⁻¹")
    print(f"  Mean: {np.mean(all_train_energies):.1f} cm⁻¹")
    print(f"  Std: {np.std(all_train_energies):.1f} cm⁻¹")

    print(f"\nValidation Data (sampled {len(all_test_energies)} oscillators):")
    print(f"  Min energy: {np.min(all_test_energies):.1f} cm⁻¹")
    print(f"  Max energy: {np.max(all_test_energies):.1f} cm⁻¹")
    print(f"  Mean: {np.mean(all_test_energies):.1f} cm⁻¹")
    print(f"  Std: {np.std(all_test_energies):.1f} cm⁻¹")

    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'max_neighbors': args.max_neighbors,
        'min_energy': 1450.0,  # Widened from 1500.0 to allow learning of low-energy sites
        'max_energy': 1850.0,  # Widened from 1800.0 to capture high-frequency components
    }

    # Use fixed model with constrained output to prevent eigenvalue decomposition failures
    model = create_model(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Check if model constraints cover ground truth range
    print(f"\nModel Energy Constraints: [{model_config['min_energy']:.1f}, {model_config['max_energy']:.1f}] cm⁻¹")
    if all_train_energies and all_test_energies:
        train_min, train_max = np.min(all_train_energies), np.max(all_train_energies)
        test_min, test_max = np.min(all_test_energies), np.max(all_test_energies)

        below_min_train = sum(e < model_config['min_energy'] for e in all_train_energies)
        above_max_train = sum(e > model_config['max_energy'] for e in all_train_energies)
        below_min_test = sum(e < model_config['min_energy'] for e in all_test_energies)
        above_max_test = sum(e > model_config['max_energy'] for e in all_test_energies)

        if below_min_train > 0 or above_max_train > 0:
            print(f"⚠️  WARNING: {below_min_train} train energies below min, {above_max_train} above max")
        if below_min_test > 0 or above_max_test > 0:
            print(f"⚠️  WARNING: {below_min_test} test energies below min, {above_max_test} above max")

        if below_min_train == 0 and above_max_train == 0 and below_min_test == 0 and above_max_test == 0:
            print(f"✓ Model constraints fully cover ground truth range")

    # Frequency grid (create before loss)
    omega_grid_tensor = torch.from_numpy(omega_grid).float().to(device)

    # Loss and optimizer (with improved multi-component loss and scaling)
    criterion = SpectrumLoss(
        lambda_peak=args.lambda_peak,
        lambda_correlation=args.lambda_correlation,
        omega_grid=omega_grid_tensor,
        peak_scale=100.0  # Scale peak loss: 100 cm^-1 difference -> loss of 1.0
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler: CONSTANT (no warmup!)
    # CRITICAL FIX: Warmup was causing model to regress to mean predictions
    # First epoch works well, then LR increases → model overshoots → predicts mean
    # Solution: Keep LR constant and low throughout training

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    print(f"\nLearning Rate Schedule:")
    print(f"  Mode: CONSTANT (no warmup, no decay)")
    print(f"  Learning rate: {args.lr:.2e} (constant throughout)")
    print(f"  Total steps: {total_steps}")
    print(f"  Reason: Warmup was causing regression to mean predictions")

    # Constant LR (return 1.0 always)
    def lr_lambda(current_step):
        return 1.0  # Keep LR constant

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"  LR at all steps: {args.lr:.2e}")

    # Training history
    history = {
        'train_loss': [],
        'train_spectrum_mse': [],
        'train_spectrum_corr': [],
        'train_peak_error_cm': [],
        'train_site_energy_mae': [],
        'test_loss': [],
        'test_spectrum_mse': [],
        'test_spectrum_corr': [],
        'test_peak_error_cm': [],
        'test_site_energy_mae': [],
    }

    # Per-protein tracking
    per_protein_history = {}
    if args.track_per_protein and val_files_data:
        for protein_id in val_files_data.keys():
            per_protein_history[protein_id] = defaultdict(list)
        print(f"Per-protein tracking enabled for {len(val_files_data)} proteins")
        print(f"  Plot update interval: every {args.plot_interval} epochs")

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    best_test_corr = -1.0
    best_epoch = 0

    # Early stopping
    if args.early_stopping:
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
        epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, omega_grid_tensor, scheduler
        )

        print(f"\nTrain Metrics:")
        print(f"  Loss: {train_metrics['loss']:.6f}")
        print(f"  Spectrum MSE: {train_metrics['spectrum_mse']:.6f}")
        print(f"  Spectrum Corr: {train_metrics['spectrum_corr']:.4f}")
        print(f"  Peak Error: {train_metrics['peak_error_cm']:.2f} cm⁻¹")
        print(f"  Site Energy MAE: {train_metrics['site_energy_mae']:.2f} cm⁻¹")
        # Print new multi-scale loss components
        if 'loss_fine' in train_metrics:
            print(f"  Multi-scale Loss Components:")
            print(f"    Fine (exact):     {train_metrics['loss_fine']:.6f}")
            print(f"    Medium (shapes):  {train_metrics['loss_medium']:.6f}")
            print(f"    Coarse (envelope):{train_metrics['loss_coarse']:.6f}")
            print(f"    Multi-peak:       {train_metrics['loss_peaks']:.6f}")
            print(f"    Gradient:         {train_metrics['loss_gradient']:.6f}")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")

        # Evaluate
        test_metrics, sample_results = evaluate(
            model, test_loader, criterion, device, omega_grid_tensor
        )

        print(f"\nTest Metrics:")
        print(f"  Loss: {test_metrics['loss']:.6f}")
        print(f"  Spectrum MSE: {test_metrics['spectrum_mse']:.6f}")
        print(f"  Spectrum Corr: {test_metrics['spectrum_corr']:.4f}")
        print(f"  Peak Error: {test_metrics['peak_error_cm']:.2f} cm⁻¹")
        print(f"  Site Energy MAE: {test_metrics['site_energy_mae']:.2f} cm⁻¹")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_spectrum_mse'].append(train_metrics['spectrum_mse'])
        history['train_spectrum_corr'].append(train_metrics['spectrum_corr'])
        history['train_peak_error_cm'].append(train_metrics['peak_error_cm'])
        history['train_site_energy_mae'].append(train_metrics['site_energy_mae'])

        history['test_loss'].append(test_metrics['loss'])
        history['test_spectrum_mse'].append(test_metrics['spectrum_mse'])
        history['test_spectrum_corr'].append(test_metrics['spectrum_corr'])
        history['test_peak_error_cm'].append(test_metrics['peak_error_cm'])
        history['test_site_energy_mae'].append(test_metrics['site_energy_mae'])

        # Learning rate schedule
        scheduler.step(test_metrics['loss'])

        # Save best model
        if test_metrics['spectrum_corr'] > best_test_corr + args.min_delta:
            improvement = test_metrics['spectrum_corr'] - best_test_corr
            best_test_corr = test_metrics['spectrum_corr']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, train_metrics, test_metrics,
                output_dir / 'best_model.pt'
            )
            print(f"  → New best model saved (corr: {best_test_corr:.4f}, improvement: {improvement:.4f})")

            if args.early_stopping:
                epochs_without_improvement = 0
        else:
            if args.early_stopping:
                epochs_without_improvement += 1
                print(f"  → No improvement ({epochs_without_improvement}/{args.patience})")

        # Early stopping check
        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING: No improvement for {args.patience} epochs")
            print(f"Best correlation: {best_test_corr:.4f} at epoch {best_epoch}")
            print(f"{'='*80}")
            break

        # Per-protein tracking
        if args.track_per_protein and val_files_data:
            from per_protein_tracking import evaluate_per_protein, plot_per_protein_evolution

            # Create config with all needed parameters
            eval_config = {
                **model_config,  # Include model config
                'cutoff': args.cutoff,  # Add cutoff from args
                'max_neighbors': args.max_neighbors,  # Already in model_config but make sure
                'lambda_peak': args.lambda_peak,
                'lambda_correlation': args.lambda_correlation
            }

            per_protein_metrics = evaluate_per_protein(
                model=model,
                val_files_data=val_files_data,
                omega_grid=omega_grid,
                device=device,
                config=eval_config,
                criterion=criterion
            )

            # Store in history
            for protein_id, metrics in per_protein_metrics.items():
                for metric_key, value in metrics.items():
                    per_protein_history[protein_id][metric_key].append(value)

            # Plot every plot_interval epochs
            if epoch % args.plot_interval == 0:
                tracking_dir = output_dir / 'per_protein_tracking'
                tracking_dir.mkdir(exist_ok=True)
                plot_per_protein_evolution(
                    per_protein_history,
                    tracking_dir / f'evolution_epoch_{epoch}.png'
                )

        # Generate spectrum plots at plot_interval
        if epoch % args.plot_interval == 0:
            print(f"\n{'='*80}")
            print(f"GENERATING PLOTS FOR EPOCH {epoch}")
            print(f"{'='*80}")

            from evaluate import plot_spectra_comparison, plot_average_spectrum, plot_site_energy_comparison

            # Create epoch-specific directory for plots
            epoch_plots_dir = output_dir / f'epoch_{epoch}_plots'
            epoch_plots_dir.mkdir(exist_ok=True)

            # Plot 1: Combined test set (all validation proteins together)
            if sample_results:
                print(f"  Generating COMBINED test set plots...")
                combined_dir = epoch_plots_dir / 'combined_test_set'
                combined_dir.mkdir(exist_ok=True)

                print(f"    - Spectra comparison (5 frames from all proteins)...")
                plot_spectra_comparison(
                    sample_results,
                    omega_grid,
                    combined_dir / 'spectra_comparison_5frames.png',
                    max_samples=5
                )

                print(f"    - Average spectrum (all test frames)...")
                plot_average_spectrum(
                    sample_results,
                    omega_grid,
                    combined_dir / 'average_spectrum_all_frames.png'
                )

                print(f"    - Site energy comparison...")
                plot_site_energy_comparison(
                    sample_results,
                    combined_dir / 'site_energy_comparison.png',
                    max_samples=5
                )
                print(f"    ✓ Combined plots saved to {combined_dir}/")

            # Plot 2: Per-protein plots (each protein separately)
            if args.track_per_protein and val_files_data:
                print(f"\n  Generating PER-PROTEIN plots...")
                from evaluate import evaluate_protein_file

                per_protein_plots_dir = epoch_plots_dir / 'per_protein'
                per_protein_plots_dir.mkdir(exist_ok=True)

                # Create config with loss weights
                eval_config = {
                    **model_config,
                    'cutoff': args.cutoff,
                    'max_neighbors': args.max_neighbors,
                    'lambda_peak': args.lambda_peak,
                    'lambda_correlation': args.lambda_correlation
                }

                for protein_id, protein_data in val_files_data.items():
                    frames = organize_by_frames(protein_data)
                    frames, _, _ = filter_frames_by_quality(frames, verbose=False)

                    if len(frames) == 0:
                        continue

                    protein_output_dir = per_protein_plots_dir / protein_id
                    protein_output_dir.mkdir(exist_ok=True)

                    print(f"    - {protein_id} ({len(frames)} frames)...")

                    try:
                        evaluate_protein_file(
                            model=model,
                            frames_dict=frames,
                            omega_grid=omega_grid,
                            device=device,
                            output_dir=protein_output_dir,
                            protein_id=protein_id,
                            config=eval_config,
                            n_sample_frames=5,
                            protein_data=protein_data
                        )
                    except Exception as e:
                        print(f"      ⚠️  Error plotting {protein_id}: {e}")

                    # Clear GPU memory after each protein to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"    ✓ Per-protein plots saved to {per_protein_plots_dir}/")

            if not sample_results:
                print(f"  ⚠️  No sample results available for plotting")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics, test_metrics,
                output_dir / f'checkpoint_epoch_{epoch}.pt'
            )

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    final_test_metrics, final_sample_results = evaluate(
        model, test_loader, criterion, device, omega_grid_tensor
    )

    print(f"\nFinal Test Metrics:")
    for key, value in final_test_metrics.items():
        print(f"  {key}: {value:.6f}" if 'corr' not in key else f"  {key}: {value:.4f}")

    print(f"\nBest model was at epoch {best_epoch} with correlation {best_test_corr:.4f}")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    plot_training_curves(history, output_dir / 'training_curves.png')
    plot_spectra_comparison(final_sample_results, omega_grid, output_dir / 'spectra_comparison.png')
    plot_site_energy_comparison(final_sample_results, output_dir / 'site_energy_comparison.png')
    plot_site_energy_distribution(final_sample_results, output_dir / 'site_energy_distribution.png')
    plot_average_spectrum(final_sample_results, omega_grid, output_dir / 'average_spectrum.png')

    # Frame-by-frame detailed comparisons
    print("Generating frame-by-frame detailed plots...")
    plot_spectra_grid_comparison(final_sample_results, omega_grid, output_dir / 'spectra_grid_comparison.png', max_frames=16)
    plot_individual_frames_detailed(final_sample_results, omega_grid, output_dir / 'frame_details', max_frames=20)

    # Extended analysis plots
    generate_all_extended_plots(history, final_sample_results, omega_grid, output_dir)

    # Per-protein evolution plots (final)
    if args.track_per_protein and per_protein_history:
        print("\n" + "="*80)
        print("PER-PROTEIN EVOLUTION PLOTS")
        print("="*80)
        from per_protein_tracking import plot_per_protein_evolution, plot_final_per_protein_comparison

        tracking_dir = output_dir / 'per_protein_tracking'
        tracking_dir.mkdir(exist_ok=True)

        plot_per_protein_evolution(
            per_protein_history,
            tracking_dir / 'final_evolution.png'
        )

        plot_final_per_protein_comparison(
            per_protein_history,
            tracking_dir / 'final_comparison.png'
        )

        # Save per-protein history to JSON
        per_protein_json = {}
        for protein_id, metrics in per_protein_history.items():
            per_protein_json[protein_id] = {k: list(v) for k, v in metrics.items()}

        with open(tracking_dir / 'per_protein_history.json', 'w') as f:
            json.dump(per_protein_json, f, indent=2)

        print(f"  Per-protein tracking saved to: {tracking_dir}")

    # Feature importance analysis
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    feature_importance_dir = output_dir / 'feature_importance'
    feature_importance_dir.mkdir(exist_ok=True, parents=True)

    print("Running comprehensive feature importance analyses...")
    print("This may take a few minutes...")
    feature_results = run_all_analyses(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=feature_importance_dir
    )

    # Save metrics summary
    save_metrics_summary(history, final_test_metrics, output_dir / 'metrics_summary.txt')

    # Save history (convert numpy types to Python types)
    history_serializable = {}
    for key, values in history.items():
        history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # Per-protein evaluation
    if args.per_protein_eval and (val_files_data or test_files_data):
        print("\n" + "="*80)
        print("PER-PROTEIN EVALUATION")
        print("="*80)

        files_to_eval = val_files_data if val_files_data else test_files_data
        per_protein_dir = output_dir / 'per_protein_results'
        per_protein_dir.mkdir(exist_ok=True)

        print(f"\nEvaluating {len(files_to_eval)} proteins individually...")

        from evaluate import evaluate_protein_file

        per_protein_results = {}

        for protein_id, protein_data in sorted(files_to_eval.items()):
            print(f"\n  Evaluating {protein_id}...")

            # Organize protein data
            protein_frames = organize_by_frames(protein_data)

            # Filter frames
            from data_utils import filter_frames_by_quality
            protein_frames, excluded, _ = filter_frames_by_quality(protein_frames, verbose=False)

            if len(protein_frames) == 0:
                print(f"    WARNING: No valid frames for {protein_id}, skipping...")
                continue

            # Evaluate and plot
            protein_output_dir = per_protein_dir / protein_id
            protein_output_dir.mkdir(exist_ok=True)

            # Create config with all needed parameters
            eval_config = {
                **model_config,
                'cutoff': args.cutoff,
                'max_neighbors': args.max_neighbors,
                'lambda_peak': args.lambda_peak,
                'lambda_correlation': args.lambda_correlation
            }

            results = evaluate_protein_file(
                model=model,
                frames_dict=protein_frames,
                omega_grid=omega_grid,
                device=device,
                output_dir=protein_output_dir,
                protein_id=protein_id,
                config=eval_config,
                n_sample_frames=5,  # Plot 5 sample frames per protein
                protein_data=protein_data  # Pass raw data for dataloader creation
            )

            per_protein_results[protein_id] = results

            print(f"    Correlation: {results['spectrum_corr']:.4f}, Peak Error: {results['peak_error_cm']:.2f} cm⁻¹")

        # Save summary
        import pandas as pd
        df = pd.DataFrame(per_protein_results).T
        df.to_csv(per_protein_dir / 'per_protein_summary.csv')

        print(f"\n  Per-protein results saved to: {per_protein_dir}")
        print(f"  Summary CSV: {per_protein_dir / 'per_protein_summary.csv'}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {output_dir / 'best_model.pt'}")
    print(f"Best test correlation: {best_test_corr:.4f} (epoch {best_epoch})")


if __name__ == '__main__':
    main()

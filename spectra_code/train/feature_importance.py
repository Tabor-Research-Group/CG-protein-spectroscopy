"""
Feature importance analysis for the transformer model.

Implements multiple methods:
1. Embedding layer weight analysis
2. Gradient-based saliency
3. Integrated gradients
4. Attention weight visualization
5. Ablation study
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


# Feature names for interpretation
OWN_FEATURE_NAMES = [
    'OscType_Regular', 'OscType_PRO', 'OscType_Sidechain',  # 0-2
    'sin(φ)', 'cos(φ)', 'sin(ψ)', 'cos(ψ)',  # 3-6
    'sin(φ_prev)', 'cos(φ_prev)', 'sin(ψ_next)', 'cos(ψ_next)',  # 7-10
    'SS_Coil', 'SS_Helix', 'SS_Sheet', 'SS_Turn',  # 11-14
    'Charge',  # 15
]

NEIGHBOR_FEATURE_NAMES = [
    '1/r³',  # 0
    'sin(θ)', 'cos(θ)', 'sin(φ)', 'cos(φ)',  # 1-4
    'Charge',  # 5
    'sin(φ)', 'cos(φ)', 'sin(ψ)', 'cos(ψ)',  # 6-9
    'sin(φ_prev)', 'cos(φ_prev)', 'sin(ψ_next)', 'cos(ψ_next)',  # 10-13
    'SS_Coil', 'SS_Helix', 'SS_Sheet', 'SS_Turn',  # 14-17
]


def analyze_embedding_weights(model: torch.nn.Module, save_dir: Path):
    """
    Analyze embedding layer weights to determine linear feature importance.

    Method: Compute L2 norm of weight vectors for each input feature.
    High norm → feature strongly influences embeddings.
    """
    print("\n=== Embedding Weight Analysis ===")

    # Get embedding layer weights
    W_own = model.own_embed.weight.data.cpu().numpy()  # [d_model, 16]
    W_neighbor = model.neighbor_embed.weight.data.cpu().numpy()  # [d_model, 18]

    # Compute importance as L2 norm across embedding dimensions
    importance_own = np.linalg.norm(W_own, axis=0)  # [16]
    importance_neighbor = np.linalg.norm(W_neighbor, axis=0)  # [18]

    # Normalize to [0, 1]
    importance_own = importance_own / importance_own.max()
    importance_neighbor = importance_neighbor / importance_neighbor.max()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot own features
    ax = axes[0]
    colors = ['C0']*3 + ['C1']*8 + ['C2']*4 + ['C3']*1
    bars = ax.barh(range(len(importance_own)), importance_own, color=colors, alpha=0.7)
    ax.set_yticks(range(len(importance_own)))
    ax.set_yticklabels(OWN_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Importance (L2 norm)', fontsize=12)
    ax.set_title('Own Feature Importance (Embedding Layer)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    # Add values
    for i, (bar, val) in enumerate(zip(bars, importance_own)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    # Plot neighbor features
    ax = axes[1]
    colors_neighbor = ['C4']*1 + ['C5']*4 + ['C3']*1 + ['C1']*8 + ['C2']*4
    bars = ax.barh(range(len(importance_neighbor)), importance_neighbor, color=colors_neighbor, alpha=0.7)
    ax.set_yticks(range(len(importance_neighbor)))
    ax.set_yticklabels(NEIGHBOR_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Importance (L2 norm)', fontsize=12)
    ax.set_title('Neighbor Feature Importance (Embedding Layer)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    # Add values
    for i, (bar, val) in enumerate(zip(bars, importance_neighbor)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / 'embedding_weight_importance.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_dir / 'embedding_weight_importance.png'}")

    # Print top features
    print("\nTop 5 Own Features:")
    top_own = np.argsort(importance_own)[::-1][:5]
    for idx in top_own:
        print(f"  {OWN_FEATURE_NAMES[idx]}: {importance_own[idx]:.3f}")

    print("\nTop 5 Neighbor Features:")
    top_neighbor = np.argsort(importance_neighbor)[::-1][:5]
    for idx in top_neighbor:
        print(f"  {NEIGHBOR_FEATURE_NAMES[idx]}: {importance_neighbor[idx]:.3f}")

    return {
        'own': importance_own,
        'neighbor': importance_neighbor
    }


def compute_gradient_saliency(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 10
):
    """
    Compute gradient-based saliency maps.

    Method: Compute ∂Loss/∂Features and average over dataset.
    High gradient → feature strongly affects predictions.
    """
    print("\n=== Gradient-Based Saliency ===")

    model.eval()

    saliency_own = []
    saliency_neighbor = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Computing saliency', total=num_batches)):
        if batch_idx >= num_batches:
            break

        # Move to device
        own_features = batch['own_features'].to(device)
        neighbor_features = batch['neighbor_features'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        H_true = batch['H_diag_true'].to(device)

        # Enable gradients
        own_features.requires_grad = True
        neighbor_features.requires_grad = True

        # Forward pass
        H_pred = model(own_features, neighbor_features, neighbor_mask)

        # Simple MSE loss on site energies
        mask = H_true > 0
        loss = torch.mean((H_pred[mask] - H_true[mask])**2)

        # Backward
        loss.backward()

        # Collect gradients
        if own_features.grad is not None:
            saliency_own.append(own_features.grad.abs().mean(dim=(0, 1)).cpu().numpy())
        if neighbor_features.grad is not None:
            saliency_neighbor.append(neighbor_features.grad.abs().mean(dim=(0, 1, 2)).cpu().numpy())

    # Average over batches
    saliency_own = np.mean(saliency_own, axis=0)  # [16]
    saliency_neighbor = np.mean(saliency_neighbor, axis=0)  # [18]

    # Normalize
    saliency_own = saliency_own / saliency_own.max()
    saliency_neighbor = saliency_neighbor / saliency_neighbor.max()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Own features
    ax = axes[0]
    colors = ['C0']*3 + ['C1']*8 + ['C2']*4 + ['C3']*1
    bars = ax.barh(range(len(saliency_own)), saliency_own, color=colors, alpha=0.7)
    ax.set_yticks(range(len(saliency_own)))
    ax.set_yticklabels(OWN_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Saliency (|∂Loss/∂Feature|)', fontsize=12)
    ax.set_title('Own Feature Saliency (Gradient-Based)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    for i, (bar, val) in enumerate(zip(bars, saliency_own)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    # Neighbor features
    ax = axes[1]
    colors_neighbor = ['C4']*1 + ['C5']*4 + ['C3']*1 + ['C1']*8 + ['C2']*4
    bars = ax.barh(range(len(saliency_neighbor)), saliency_neighbor, color=colors_neighbor, alpha=0.7)
    ax.set_yticks(range(len(saliency_neighbor)))
    ax.set_yticklabels(NEIGHBOR_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Saliency (|∂Loss/∂Feature|)', fontsize=12)
    ax.set_title('Neighbor Feature Saliency (Gradient-Based)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    for i, (bar, val) in enumerate(zip(bars, saliency_neighbor)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / 'gradient_saliency.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_dir / 'gradient_saliency.png'}")

    print("\nTop 5 Own Features (Saliency):")
    top_own = np.argsort(saliency_own)[::-1][:5]
    for idx in top_own:
        print(f"  {OWN_FEATURE_NAMES[idx]}: {saliency_own[idx]:.3f}")

    print("\nTop 5 Neighbor Features (Saliency):")
    top_neighbor = np.argsort(saliency_neighbor)[::-1][:5]
    for idx in top_neighbor:
        print(f"  {NEIGHBOR_FEATURE_NAMES[idx]}: {saliency_neighbor[idx]:.3f}")

    return {
        'own': saliency_own,
        'neighbor': saliency_neighbor
    }


def compute_integrated_gradients(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 5,
    num_steps: int = 20
):
    """
    Compute integrated gradients for robust feature attribution.

    Method: Integrate gradients along path from baseline (zeros) to input.
    More robust than single gradient computation.
    """
    print("\n=== Integrated Gradients ===")

    model.eval()

    integrated_own = []
    integrated_neighbor = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Computing integrated gradients', total=num_batches)):
        if batch_idx >= num_batches:
            break

        own_features = batch['own_features'].to(device)
        neighbor_features = batch['neighbor_features'].to(device)
        neighbor_mask = batch['neighbor_mask'].to(device)
        H_true = batch['H_diag_true'].to(device)

        # Baseline: zeros (or mean values)
        baseline_own = torch.zeros_like(own_features)
        baseline_neighbor = torch.zeros_like(neighbor_features)

        # Accumulate gradients
        accum_grad_own = torch.zeros_like(own_features)
        accum_grad_neighbor = torch.zeros_like(neighbor_features)

        for alpha in np.linspace(0, 1, num_steps):
            # Interpolate between baseline and input
            interp_own = baseline_own + alpha * (own_features - baseline_own)
            interp_neighbor = baseline_neighbor + alpha * (neighbor_features - baseline_neighbor)

            interp_own.requires_grad = True
            interp_neighbor.requires_grad = True

            # Forward
            H_pred = model(interp_own, interp_neighbor, neighbor_mask)

            # Loss
            mask = H_true > 0
            loss = torch.mean((H_pred[mask] - H_true[mask])**2)

            # Backward
            loss.backward()

            # Accumulate
            if interp_own.grad is not None:
                accum_grad_own += interp_own.grad
            if interp_neighbor.grad is not None:
                accum_grad_neighbor += interp_neighbor.grad

            # Zero gradients
            model.zero_grad()

        # Integrated gradient = (input - baseline) * average_gradient
        ig_own = (own_features - baseline_own) * (accum_grad_own / num_steps)
        ig_neighbor = (neighbor_features - baseline_neighbor) * (accum_grad_neighbor / num_steps)

        # Average over batch and oscillators
        integrated_own.append(ig_own.abs().mean(dim=(0, 1)).cpu().numpy())
        integrated_neighbor.append(ig_neighbor.abs().mean(dim=(0, 1, 2)).cpu().numpy())

    # Average over batches
    integrated_own = np.mean(integrated_own, axis=0)
    integrated_neighbor = np.mean(integrated_neighbor, axis=0)

    # Normalize
    integrated_own = integrated_own / integrated_own.max()
    integrated_neighbor = integrated_neighbor / integrated_neighbor.max()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Own features
    ax = axes[0]
    colors = ['C0']*3 + ['C1']*8 + ['C2']*4 + ['C3']*1
    bars = ax.barh(range(len(integrated_own)), integrated_own, color=colors, alpha=0.7)
    ax.set_yticks(range(len(integrated_own)))
    ax.set_yticklabels(OWN_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Attribution (Integrated Gradients)', fontsize=12)
    ax.set_title('Own Feature Attribution (Integrated Gradients)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    for i, (bar, val) in enumerate(zip(bars, integrated_own)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    # Neighbor features
    ax = axes[1]
    colors_neighbor = ['C4']*1 + ['C5']*4 + ['C3']*1 + ['C1']*8 + ['C2']*4
    bars = ax.barh(range(len(integrated_neighbor)), integrated_neighbor, color=colors_neighbor, alpha=0.7)
    ax.set_yticks(range(len(integrated_neighbor)))
    ax.set_yticklabels(NEIGHBOR_FEATURE_NAMES, fontsize=10)
    ax.set_xlabel('Normalized Attribution (Integrated Gradients)', fontsize=12)
    ax.set_title('Neighbor Feature Attribution (Integrated Gradients)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

    for i, (bar, val) in enumerate(zip(bars, integrated_neighbor)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / 'integrated_gradients.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_dir / 'integrated_gradients.png'}")

    print("\nTop 5 Own Features (Integrated Gradients):")
    top_own = np.argsort(integrated_own)[::-1][:5]
    for idx in top_own:
        print(f"  {OWN_FEATURE_NAMES[idx]}: {integrated_own[idx]:.3f}")

    print("\nTop 5 Neighbor Features (Integrated Gradients):")
    top_neighbor = np.argsort(integrated_neighbor)[::-1][:5]
    for idx in top_neighbor:
        print(f"  {NEIGHBOR_FEATURE_NAMES[idx]}: {integrated_neighbor[idx]:.3f}")

    return {
        'own': integrated_own,
        'neighbor': integrated_neighbor
    }


def compare_all_methods(
    results: Dict[str, Dict],
    save_dir: Path
):
    """
    Compare all feature importance methods side-by-side.
    """
    print("\n=== Comparing All Methods ===")

    methods = list(results.keys())
    n_methods = len(methods)

    # Create comparison plots
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))

    for i, method in enumerate(methods):
        # Own features
        ax = axes[0, i]
        importance = results[method]['own']
        colors = ['C0']*3 + ['C1']*8 + ['C2']*4 + ['C3']*1
        ax.barh(range(len(importance)), importance, color=colors, alpha=0.7)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(OWN_FEATURE_NAMES, fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'{method.upper()}\n(Own Features)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.1)

        # Neighbor features
        ax = axes[1, i]
        importance = results[method]['neighbor']
        colors_neighbor = ['C4']*1 + ['C5']*4 + ['C3']*1 + ['C1']*8 + ['C2']*4
        ax.barh(range(len(importance)), importance, color=colors_neighbor, alpha=0.7)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(NEIGHBOR_FEATURE_NAMES, fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'{method.upper()}\n(Neighbor Features)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_dir / 'feature_importance_comparison.png'}")

    # Compute consensus ranking (average rank across methods)
    print("\n=== Consensus Feature Ranking ===")

    own_ranks = []
    neighbor_ranks = []

    for method in methods:
        own_ranks.append(np.argsort(np.argsort(results[method]['own'])[::-1]))
        neighbor_ranks.append(np.argsort(np.argsort(results[method]['neighbor'])[::-1]))

    avg_own_ranks = np.mean(own_ranks, axis=0)
    avg_neighbor_ranks = np.mean(neighbor_ranks, axis=0)

    print("\nTop 5 Own Features (Consensus):")
    top_own = np.argsort(avg_own_ranks)[:5]
    for rank, idx in enumerate(top_own, 1):
        print(f"  {rank}. {OWN_FEATURE_NAMES[idx]} (avg rank: {avg_own_ranks[idx]:.1f})")

    print("\nTop 5 Neighbor Features (Consensus):")
    top_neighbor = np.argsort(avg_neighbor_ranks)[:5]
    for rank, idx in enumerate(top_neighbor, 1):
        print(f"  {rank}. {NEIGHBOR_FEATURE_NAMES[idx]} (avg rank: {avg_neighbor_ranks[idx]:.1f})")


def analyze_attention_patterns(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 5
):
    """
    Visualize attention patterns from transformer layers.

    Shows which neighbors the model attends to.
    """
    print("\n=== Attention Pattern Analysis ===")

    model.eval()

    # We need to modify the model to return attention weights
    # For now, let's analyze the learned behavior by examining outputs

    print("Note: Full attention visualization requires modifying model forward pass")
    print("      to return attention weights. Skipping for now.")

    # TODO: Implement attention extraction if needed
    pass


def run_all_analyses(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_dir: Path
):
    """
    Run all feature importance analyses.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    # 1. Embedding weights
    results['embedding'] = analyze_embedding_weights(model, save_dir)

    # 2. Gradient saliency
    results['saliency'] = compute_gradient_saliency(model, dataloader, device, save_dir)

    # 3. Integrated gradients
    results['integrated'] = compute_integrated_gradients(model, dataloader, device, save_dir)

    # 4. Compare all methods
    compare_all_methods(results, save_dir)

    print("\n" + "="*60)
    print("Feature importance analysis complete!")
    print(f"Results saved to: {save_dir}")
    print("="*60)

    return results

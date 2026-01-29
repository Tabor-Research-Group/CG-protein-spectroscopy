"""
Transformer-based model for site energy prediction - FIXED VERSION.

KEY FIX: Added output activation to constrain site energies to physical range (1500-1800 cm⁻¹).
This prevents ill-conditioned Hamiltonians that cause eigenvalue decomposition failures.

Architecture:
1. Feature embedding for own features and neighbor features
2. Transformer encoder to aggregate information
3. Output head with CONSTRAINED output to predict site energies H_ii
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer (optional, if needed)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            x: [B, L, d_model] with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class ConstrainedOutputHead(nn.Module):
    """
    Output head with constrained range for site energies.

    Physical constraints:
    - Site energies should be in range ~1500-1800 cm⁻¹
    - Typical values: 1600-1700 cm⁻¹ with std ~25 cm⁻¹
    - Extreme values outside this range cause ill-conditioned Hamiltonians

    Strategy:
    - Use sigmoid activation to map to [0, 1]
    - Scale to [min_energy, max_energy] range
    - Allow some flexibility beyond typical range for model learning
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        min_energy: float = 1500.0,
        max_energy: float = 1800.0,
    ):
        super().__init__()

        self.min_energy = min_energy
        self.max_energy = max_energy
        self.energy_range = max_energy - min_energy

        # MLP layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)

        # Initialize fc2 bias to map to center of range (~1640)
        # sigmoid(x) = 0.5 when x = 0
        # So bias = 0 gives output = 1640 initially
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, d_model] input features

        Returns:
            energy: [*, 1] predicted site energies in range [min_energy, max_energy]
        """
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [..., 1] raw logits

        # Apply sigmoid to get [0, 1]
        x = torch.sigmoid(x)

        # Scale to [min_energy, max_energy]
        energy = self.min_energy + x * self.energy_range

        return energy


class SiteEnergyPredictorConstrained(nn.Module):
    """
    More efficient version with CONSTRAINED output range.

    Uses batched transformer attention.
    """

    def __init__(
        self,
        own_feature_dim: int = 16,
        neighbor_feature_dim: int = 18,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_neighbors: int = 80,
        min_energy: float = 1500.0,
        max_energy: float = 1800.0,
    ):
        super().__init__()

        self.own_feature_dim = own_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim
        self.d_model = d_model
        self.max_neighbors = max_neighbors
        self.min_energy = min_energy
        self.max_energy = max_energy

        # Feature embeddings
        self.own_embed = nn.Linear(own_feature_dim, d_model)
        self.neighbor_embed = nn.Linear(neighbor_feature_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head with constrained range
        self.output_head = ConstrainedOutputHead(
            d_model=d_model,
            dropout=dropout,
            min_energy=min_energy,
            max_energy=max_energy
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        own_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass (efficient batched version).

        Args:
            own_features: [B, N, F_own]
            neighbor_features: [B, N, K, F_neighbor]
            neighbor_mask: [B, N, K]

        Returns:
            H_diag: [B, N] predicted site energies in range [min_energy, max_energy]
        """
        B, N, K, _ = neighbor_features.shape

        # Embed features
        own_embed = self.own_embed(own_features)  # [B, N, d_model]
        neighbor_embed = self.neighbor_embed(neighbor_features)  # [B, N, K, d_model]

        # Reshape to treat all oscillators as separate sequences
        # Concatenate own + neighbors for each oscillator
        own_expand = own_embed.unsqueeze(2)  # [B, N, 1, d_model]
        tokens = torch.cat([own_expand, neighbor_embed], dim=2)  # [B, N, 1+K, d_model]

        # Reshape to [B*N, 1+K, d_model]
        tokens_flat = tokens.reshape(B * N, 1 + K, self.d_model)

        # Create padding mask
        own_mask = torch.ones(B, N, 1, device=neighbor_mask.device)
        full_mask = torch.cat([own_mask, neighbor_mask], dim=2)  # [B, N, 1+K]
        mask_flat = full_mask.reshape(B * N, 1 + K)
        attn_mask = (mask_flat == 0)  # True = ignore

        # Transformer
        encoded_flat = self.transformer(
            tokens_flat,
            src_key_padding_mask=attn_mask
        )  # [B*N, 1+K, d_model]

        # Extract own token (first token)
        own_encoded_flat = encoded_flat[:, 0, :]  # [B*N, d_model]

        # Predict site energy with constrained output
        H_flat = self.output_head(own_encoded_flat).squeeze(-1)  # [B*N]

        # Reshape back
        H_diag = H_flat.reshape(B, N)  # [B, N]

        return H_diag


def create_model_constrained(config: dict) -> nn.Module:
    """
    Create model with constrained output from configuration.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        model: SiteEnergyPredictorConstrained instance
    """
    model = SiteEnergyPredictorConstrained(
        own_feature_dim=config.get('own_feature_dim', 16),
        neighbor_feature_dim=config.get('neighbor_feature_dim', 18),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout', 0.1),
        max_neighbors=config.get('max_neighbors', 80),
        min_energy=config.get('min_energy', 1500.0),
        max_energy=config.get('max_energy', 1800.0),
    )

    return model


if __name__ == '__main__':
    # Test model
    B, N, K = 2, 10, 80
    F_own, F_neighbor = 16, 18

    own_features = torch.randn(B, N, F_own)
    neighbor_features = torch.randn(B, N, K, F_neighbor)
    neighbor_mask = torch.ones(B, N, K)
    neighbor_mask[:, :, 50:] = 0  # Pad last 30 neighbors

    config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_neighbors': K,
        'min_energy': 1500.0,
        'max_energy': 1800.0,
    }

    model = create_model_constrained(config)
    H_diag = model(own_features, neighbor_features, neighbor_mask)

    print(f"Input shapes:")
    print(f"  own_features: {own_features.shape}")
    print(f"  neighbor_features: {neighbor_features.shape}")
    print(f"  neighbor_mask: {neighbor_mask.shape}")
    print(f"\nOutput shape: {H_diag.shape}")
    print(f"Output (H_diag): {H_diag}")
    print(f"  Min: {H_diag.min().item():.2f} cm⁻¹")
    print(f"  Max: {H_diag.max().item():.2f} cm⁻¹")
    print(f"  Mean: {H_diag.mean().item():.2f} cm⁻¹")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n✓ All outputs are constrained to [{config['min_energy']}, {config['max_energy']}] cm⁻¹")

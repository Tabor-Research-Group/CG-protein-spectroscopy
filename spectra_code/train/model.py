"""
Transformer-based model for site energy prediction.

Architecture:
1. Feature embedding for own features and neighbor features
2. Transformer encoder to aggregate information
3. Output head to predict site energies H_ii
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


class SiteEnergyPredictor(nn.Module):
    """
    Transformer-based model to predict site energies H_ii.

    Architecture:
        1. Embed own features
        2. Embed neighbor features
        3. Concatenate own + neighbors
        4. Transformer encoder
        5. Aggregate (mean pool over neighbors)
        6. Output head
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
    ):
        super().__init__()

        self.own_feature_dim = own_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim
        self.d_model = d_model
        self.max_neighbors = max_neighbors

        # Feature embeddings
        self.own_embed = nn.Linear(own_feature_dim, d_model)
        self.neighbor_embed = nn.Linear(neighbor_feature_dim, d_model)

        # Positional encoding (optional)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_neighbors + 1)

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

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize final layer bias to expected site energy mean (~1640 cm⁻¹)
        # This helps the model start near the correct range
        if hasattr(self.output_head[-1], 'bias') and self.output_head[-1].bias is not None:
            nn.init.constant_(self.output_head[-1].bias, 1640.0)

    def forward(
        self,
        own_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            own_features: [B, N, F_own] own oscillator features
            neighbor_features: [B, N, max_neighbors, F_neighbor] neighbor features
            neighbor_mask: [B, N, max_neighbors] mask (1=valid, 0=padded)

        Returns:
            H_diag: [B, N] predicted site energies
        """
        B, N, _ = own_features.shape

        # Embed own features
        own_embed = self.own_embed(own_features)  # [B, N, d_model]

        # Embed neighbor features
        neighbor_embed = self.neighbor_embed(neighbor_features)  # [B, N, max_neighbors, d_model]

        # Process each oscillator separately
        H_diag_list = []

        for i in range(N):
            # Get features for oscillator i across batch
            own_i = own_embed[:, i, :]  # [B, d_model]
            neighbors_i = neighbor_embed[:, i, :, :]  # [B, max_neighbors, d_model]
            mask_i = neighbor_mask[:, i, :]  # [B, max_neighbors]

            # Concatenate own + neighbors
            # Add own as first token
            tokens = torch.cat([own_i.unsqueeze(1), neighbors_i], dim=1)  # [B, 1 + max_neighbors, d_model]

            # Create attention mask (1=attend, 0=ignore)
            # Own token always attends
            own_mask = torch.ones(B, 1, device=mask_i.device)
            full_mask = torch.cat([own_mask, mask_i], dim=1)  # [B, 1 + max_neighbors]

            # Transformer expects mask where True=ignore, False=attend
            attn_mask = (full_mask == 0)  # [B, 1 + max_neighbors]

            # Add positional encoding
            tokens = self.pos_encoding(tokens)

            # Transformer (with padding mask)
            # Expand attention mask for multi-head: [B, 1, 1+K] -> [B*n_heads, 1+K, 1+K]
            # For simplicity, use src_key_padding_mask
            encoded = self.transformer(
                tokens,
                src_key_padding_mask=attn_mask
            )  # [B, 1 + max_neighbors, d_model]

            # Extract own token (first token) after encoding
            own_encoded = encoded[:, 0, :]  # [B, d_model]

            # Predict site energy
            H_i = self.output_head(own_encoded).squeeze(-1)  # [B]

            H_diag_list.append(H_i)

        # Stack predictions
        H_diag = torch.stack(H_diag_list, dim=1)  # [B, N]

        return H_diag


class SiteEnergyPredictorEfficient(nn.Module):
    """
    More efficient version that processes all oscillators together.

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
    ):
        super().__init__()

        self.own_feature_dim = own_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim
        self.d_model = d_model
        self.max_neighbors = max_neighbors

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

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize final layer bias to expected site energy mean
        if hasattr(self.output_head[-1], 'bias') and self.output_head[-1].bias is not None:
            nn.init.constant_(self.output_head[-1].bias, 1640.0)

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
            H_diag: [B, N]
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

        # Predict site energy
        H_flat = self.output_head(own_encoded_flat).squeeze(-1)  # [B*N]

        # Reshape back
        H_diag = H_flat.reshape(B, N)  # [B, N]

        return H_diag


def create_model(config: dict) -> nn.Module:
    """
    Create model from configuration.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        model: SiteEnergyPredictor instance
    """
    model = SiteEnergyPredictorEfficient(
        own_feature_dim=config.get('own_feature_dim', 16),
        neighbor_feature_dim=config.get('neighbor_feature_dim', 18),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout', 0.1),
        max_neighbors=config.get('max_neighbors', 80),
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
    }

    model = create_model(config)
    H_diag = model(own_features, neighbor_features, neighbor_mask)

    print(f"Input shapes:")
    print(f"  own_features: {own_features.shape}")
    print(f"  neighbor_features: {neighbor_features.shape}")
    print(f"  neighbor_mask: {neighbor_mask.shape}")
    print(f"\nOutput shape: {H_diag.shape}")
    print(f"Output (H_diag): {H_diag}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

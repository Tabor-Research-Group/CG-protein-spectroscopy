from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal embedding for scalar integer/continuous inputs."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalEmbedding dim must be even")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...], returns [..., dim]"""
        half = self.dim // 2
        # frequencies
        device = x.device
        dtype = x.dtype
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / (half - 1))
        )
        # [..., half]
        args = x.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Diffusion timestep embedding -> MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.sin = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is integer timestep in [0,T-1]
        t = t.to(dtype=torch.float32)
        return self.mlp(self.sin(t))

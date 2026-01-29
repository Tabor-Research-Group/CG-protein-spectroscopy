from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

import torch

from backmap.geometry.frames import clamp_norm


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


def _linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def make_schedule(
    timesteps: int,
    beta_schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str = "cpu",
) -> DiffusionSchedule:
    if beta_schedule == "cosine":
        betas = _cosine_beta_schedule(timesteps)
    elif beta_schedule == "linear":
        betas = _linear_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

    betas = betas.to(device=device, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # posterior q(x_{t-1} | x_t, x_0)
    alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device, dtype=torch.float64), alpha_bars[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
    posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
    )


class GaussianDiffusion:
    def __init__(
        self,
        schedule: DiffusionSchedule,
        timesteps: int,
        max_radius: float = 6.0,
        clip_each_step: bool = True,
        eps: float = 1e-8,
    ):
        self.schedule = schedule
        self.timesteps = int(timesteps)
        self.max_radius = float(max_radius)
        self.clip_each_step = bool(clip_each_step)
        self.eps = eps

    def q_sample(
        self,
        x0: torch.Tensor,  # [Na,3]
        t: torch.Tensor,   # [Na] int64 in [0,T-1]
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t given x_0 and t (vectorized per atom)."""
        if noise is None:
            noise = torch.randn_like(x0)

        # Gather sqrt(alpha_bar_t) and sqrt(1-alpha_bar_t)
        sqrt_ab = self.schedule.sqrt_alpha_bars.to(x0.device, x0.dtype)[t]  # [Na]
        sqrt_om = self.schedule.sqrt_one_minus_alpha_bars.to(x0.device, x0.dtype)[t]  # [Na]

        xt = sqrt_ab.unsqueeze(-1) * x0 + sqrt_om.unsqueeze(-1) * noise

        if self.clip_each_step:
            xt = clamp_norm(xt, self.max_radius, eps=self.eps)
        return xt

    def predict_eps_from_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0_pred: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars.to(xt.device, xt.dtype)[t]
        sqrt_om = self.schedule.sqrt_one_minus_alpha_bars.to(xt.device, xt.dtype)[t]
        return (xt - sqrt_ab.unsqueeze(-1) * x0_pred) / torch.clamp(sqrt_om.unsqueeze(-1), min=self.eps)

    def p_mean_variance(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior mean/var for p(x_{t-1} | x_t)."""
        t0 = t
        coef1 = self.schedule.posterior_mean_coef1.to(xt.device, xt.dtype)[t0]
        coef2 = self.schedule.posterior_mean_coef2.to(xt.device, xt.dtype)[t0]
        mean = coef1.unsqueeze(-1) * x0_pred + coef2.unsqueeze(-1) * xt
        var = self.schedule.posterior_variance.to(xt.device, xt.dtype)[t0]
        log_var = self.schedule.posterior_log_variance_clipped.to(xt.device, xt.dtype)[t0]
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xt: torch.Tensor,
        t_scalar: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single reverse step for all atoms (same timestep for all)."""
        device = xt.device
        Na = xt.shape[0]
        t = torch.full((Na,), int(t_scalar), device=device, dtype=torch.long)

        x0_pred = model_fn(xt, t)  # [Na,3]
        mean, var, _ = self.p_mean_variance(xt, t, x0_pred)
        if t_scalar == 0:
            out = mean
        else:
            if noise is None:
                noise = torch.randn_like(xt)
            out = mean + torch.sqrt(torch.clamp(var, min=self.eps)).unsqueeze(-1) * noise

        if self.clip_each_step:
            out = clamp_norm(out, self.max_radius, eps=self.eps)
        return out

    @torch.no_grad()
    def sample_loop(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, int],  # (Na,3)
        init: str = "gaussian",
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        Na, dim = shape
        assert dim == 3

        if init == "gaussian":
            xt = torch.randn((Na, 3), device=device)
        elif init == "uniform_ball":
            # Sample uniform inside ball radius max_radius
            v = torch.randn((Na, 3), device=device)
            v = v / torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=self.eps))
            u = torch.rand((Na, 1), device=device)
            r = self.max_radius * (u ** (1.0 / 3.0))
            xt = v * r
        else:
            raise ValueError(f"Unknown init: {init}")
        if self.clip_each_step:
            xt = clamp_norm(xt, self.max_radius, eps=self.eps)

        for t in reversed(range(self.timesteps)):
            xt = self.p_sample(model_fn, xt, t_scalar=t)
        return xt

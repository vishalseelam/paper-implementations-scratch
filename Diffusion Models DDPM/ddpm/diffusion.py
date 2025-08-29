from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .schedules import get_beta_schedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "linear"


class Diffusion(nn.Module):
    def __init__(self, model: nn.Module, image_size: int, channels: int, timesteps: int = 1000, schedule: str = "linear"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        betas = get_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recipm1_alphas", torch.sqrt(1.0 / alphas - 1))

        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev)) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)) / (1.0 - alphas_cumprod))

    @torch.no_grad()
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        return torch.mean((noise - predicted_noise) ** 2)

    @torch.no_grad()
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor):
        eps_pred = self.model(x_t, t)
        x0_pred = extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t - \
                  extract(self.sqrt_recipm1_alphas, t, x_t.shape) * eps_pred
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + \
                     extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return model_mean, model_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(x_t, t)
        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(x_t)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img


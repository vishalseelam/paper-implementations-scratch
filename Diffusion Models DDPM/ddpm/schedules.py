import torch


def beta_linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def beta_cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    # Nichol & Dhariwal cosine schedule
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)


def get_beta_schedule(name: str, timesteps: int) -> torch.Tensor:
    name = name.lower()
    if name == "linear":
        return beta_linear_schedule(timesteps)
    if name == "cosine":
        return beta_cosine_schedule(timesteps)
    raise ValueError(f"Unknown schedule: {name}")


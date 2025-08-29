import os
from datetime import datetime

import torch
from torchvision.utils import save_image


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


def denormalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1, 1) + 1) * 0.5
    return x


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_grid(x: torch.Tensor, path: str, nrow: int = 8):
    x = denormalize_to_uint8(x)
    save_image(x, path, nrow=nrow)


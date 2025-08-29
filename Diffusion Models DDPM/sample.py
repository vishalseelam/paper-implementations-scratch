import argparse
import os

import torch

from ddpm.model import UNetModel
from ddpm.diffusion import Diffusion
from ddpm.ema import ExponentialMovingAverage
from ddpm.utils import ensure_dir, save_grid


def parse_args():
    p = argparse.ArgumentParser(description="Sample from trained DDPM")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--outdir", type=str, default="outputs/samples")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get("args", {})
    in_channels = 3 if train_args.get("dataset", "cifar10") == "cifar10" else 1
    image_size = train_args.get("image_size", 32)
    timesteps = train_args.get("timesteps", 1000)
    schedule = train_args.get("schedule", "linear")

    model = UNetModel(
        in_channels=in_channels,
        out_channels=in_channels,
        image_size=image_size,
        model_channels=128 if image_size <= 64 else 192,
        channel_mults=(1, 2, 2, 2) if image_size <= 32 else (1, 2, 2, 4),
        attention_resolutions=(16,) if image_size <= 32 else (16, 8),
        num_res_blocks=2,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ema = ExponentialMovingAverage(model)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)

    diffusion = Diffusion(model=model, image_size=image_size, channels=in_channels, timesteps=timesteps, schedule=schedule)
    model.eval()
    outdir = ensure_dir(args.outdir)

    all_samples = []
    remaining = args.num_samples
    while remaining > 0:
        b = min(args.batch_size, remaining)
        with torch.no_grad():
            samples = diffusion.sample(batch_size=b, device=device)
        all_samples.append(samples.cpu())
        remaining -= b

    samples = torch.cat(all_samples, dim=0)
    save_grid(samples, os.path.join(outdir, "samples.png"), nrow=int(args.num_samples ** 0.5))


if __name__ == "__main__":
    main()


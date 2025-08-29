import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from ddpm.data import get_dataset, get_dataloader
from ddpm.model import UNetModel
from ddpm.diffusion import Diffusion
from ddpm.ema import ExponentialMovingAverage
from ddpm.utils import ensure_dir, save_grid, timestamp


def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--outdir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, in_channels = get_dataset(args.dataset, args.data_dir, args.image_size)
    loader = get_dataloader(dataset, args.batch_size, num_workers=args.num_workers)

    model = UNetModel(
        in_channels=in_channels,
        out_channels=in_channels,
        image_size=args.image_size,
        model_channels=128 if args.image_size <= 64 else 192,
        channel_mults=(1, 2, 2, 2) if args.image_size <= 32 else (1, 2, 2, 4),
        attention_resolutions=(16,) if args.image_size <= 32 else (16, 8),
        num_res_blocks=2,
        dropout=0.1,
    ).to(device)

    diffusion = Diffusion(model=model, image_size=args.image_size, channels=in_channels, timesteps=args.timesteps, schedule=args.schedule)
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = GradScaler(enabled=args.mixed_precision)

    out_root = ensure_dir(args.outdir)
    run_dir = ensure_dir(os.path.join(out_root, f"ddpm-{args.dataset}-{timestamp()}"))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    sample_dir = ensure_dir(os.path.join(run_dir, "samples"))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.mixed_precision):
                loss = diffusion.p_losses(x, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ema.update(model)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

        if epoch % args.sample_every == 0:
            model.eval()
            ema.apply_to(model)
            with torch.no_grad():
                samples = diffusion.sample(batch_size=args.sample_count, device=device)
            save_grid(samples, os.path.join(sample_dir, f"epoch_{epoch}.png"), nrow=int(args.sample_count ** 0.5))


if __name__ == "__main__":
    main()


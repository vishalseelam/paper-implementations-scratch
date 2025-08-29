## DDPM (Denoising Diffusion Probabilistic Models)

End-to-end PyTorch implementation of DDPM (Ho et al., 2020) with a compact U-Net, linear/cosine beta schedules, EMA, mixed-precision training, and sampling scripts.

### Features
- Linear and cosine noise schedules
- Epsilon-parameterization loss (MSE)
- U-Net with time embeddings and optional attention
- Exponential Moving Average (EMA) of weights
- AMP (float16) training on GPU
- CIFAR-10/MNIST loaders with simple augmentations
- DDPM ancestral sampler (optionally deterministic DDIM)

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python train.py \
  --dataset cifar10 \
  --data-dir ./data \
  --image-size 32 \
  --batch-size 128 \
  --timesteps 1000 \
  --schedule linear \
  --lr 2e-4 \
  --epochs 200
```

Checkpoints and samples will be written under `outputs/`.

### Sample

```bash
python sample.py --ckpt outputs/checkpoints/last.pt --num-samples 64 --image-size 32
```

Generated grids are saved to `outputs/samples/`.

### References
- Ho et al., 2020: Denoising Diffusion Probabilistic Models
- Nichol & Dhariwal, 2021: Improved Denoising Diffusion Probabilistic Models


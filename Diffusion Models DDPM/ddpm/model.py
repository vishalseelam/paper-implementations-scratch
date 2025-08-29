from __future__ import annotations

import math
from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        attn = torch.einsum('bncd,bnce->bnde', q, k) * (1.0 / math.sqrt(c // self.num_heads))
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bnde,bnce->bncd', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        channel_mults: Iterable[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.image_size = image_size

        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down path
        down = []
        in_ch = model_channels
        resolutions = [image_size]
        channels_list = [in_ch]
        for i, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                down.append(ResidualBlock(in_ch, out_ch, time_dim, dropout))
                if resolutions[-1] in attention_resolutions:
                    down.append(AttentionBlock(out_ch))
                in_ch = out_ch
                channels_list.append(in_ch)
            if i != len(channel_mults) - 1:
                down.append(Downsample(in_ch))
                resolutions.append(resolutions[-1] // 2)
        self.down = nn.ModuleList(down)

        # Middle
        mid_ch = in_ch
        self.mid1 = ResidualBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, time_dim, dropout)

        # Up path
        up = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                up.append(ResidualBlock(in_ch + out_ch, out_ch, time_dim, dropout))
                if resolutions[-1] in attention_resolutions:
                    up.append(AttentionBlock(out_ch))
                in_ch = out_ch
            if i != 0:
                up.append(Upsample(in_ch))
                resolutions.pop()
        self.up = nn.ModuleList(up)

        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(timesteps, self.model_channels)
        t_emb = self.time_mlp(t_emb)

        x = self.init_conv(x)

        hs = [x]
        for module in self.down:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
                hs.append(x)
            elif isinstance(module, (AttentionBlock, Downsample)):
                x = module(x)
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for module in self.up:
            if isinstance(module, ResidualBlock):
                x = torch.cat([x, hs.pop()], dim=1)
                x = module(x, t_emb)
            elif isinstance(module, (AttentionBlock, Upsample)):
                x = module(x)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


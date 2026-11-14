"""Small time-conditioned U-Net trained to predict diffusion noise."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim >= 2 and dim % 2 == 0
        self.dim = dim

    def forward(self, t):
        t = t.float()
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        angles = t[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()
        assert in_channels % 8 == 0 and out_channels % 8 == 0

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_dim=64):
        super().__init__()
        c = base_channels

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.enc1 = TimeResBlock(c, c, time_dim)
        self.down1 = Downsample(c)
        self.enc2 = TimeResBlock(c, c * 2, time_dim)
        self.down2 = Downsample(c * 2)

        self.mid1 = TimeResBlock(c * 2, c * 4, time_dim)
        self.mid2 = TimeResBlock(c * 4, c * 4, time_dim)

        self.up2 = Upsample(c * 4)
        self.dec2 = TimeResBlock(c * 6, c * 2, time_dim)
        self.up1 = Upsample(c * 2)
        self.dec1 = TimeResBlock(c * 3, c, time_dim)

        self.output_norm = nn.GroupNorm(8, c)
        self.output_conv = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))
        h = self.input_conv(x)
        e1 = self.enc1(h, t_emb)
        h = self.down1(e1)
        e2 = self.enc2(h, t_emb)
        h = self.down2(e2)

        h = self.mid2(self.mid1(h, t_emb), t_emb)
        h = self.up2(h)
        h = self.dec2(torch.cat([h, e2], dim=1), t_emb)
        h = self.up1(h)
        h = self.dec1(torch.cat([h, e1], dim=1), t_emb)
        return self.output_conv(F.silu(self.output_norm(h)))

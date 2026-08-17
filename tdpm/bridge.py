"""Truncation bridge: maps a degraded observation to the latent at t*.

Three parameterisations are provided; they correspond one-to-one with the
ablation reported in the paper:

    param_type = "linear"     ->  A y + b
    param_type = "nonlinear"  ->  f(y)
    param_type = "hybrid"     ->  A y + b + f(y)      (default)

``A`` is stored in the factored form ``A = U V^T`` and trained by gradient
descent, which keeps the parameter count tractable at high resolution: a dense
map between two 256x256x3 images would need 1.5e10 entries, whereas a rank-256
factorisation evaluated at 64x64 needs about 7e6.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAM_TYPES = ("linear", "nonlinear", "hybrid")


class LowRankLinear(nn.Module):
    """y -> A y + b with A = U V^T, optionally evaluated at a lower resolution."""

    def __init__(self, in_channels, out_channels=3, image_size=256, rank=256,
                 work_size=None):
        super().__init__()
        self.image_size = image_size
        self.work_size = work_size or image_size
        self.out_channels = out_channels

        in_dim = in_channels * self.work_size * self.work_size
        out_dim = out_channels * self.work_size * self.work_size
        self.U = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(in_dim))
        self.V = nn.Parameter(torch.randn(out_dim, rank) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, cond):
        batch = cond.shape[0]
        x = cond
        if self.work_size != cond.shape[-1]:
            x = F.interpolate(x, size=(self.work_size, self.work_size), mode="area")
        h = x.reshape(batch, -1) @ self.U
        out = (h @ self.V.T + self.bias).view(batch, self.out_channels,
                                              self.work_size, self.work_size)
        if self.work_size != self.image_size:
            out = F.interpolate(out, size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)
        return out


class GroupNormCompat(nn.GroupNorm):
    def __init__(self, channels, num_groups=8):
        super().__init__(math.gcd(num_groups, channels), channels)


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        GroupNormCompat(out_channels), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        GroupNormCompat(out_channels), nn.SiLU())


class CorrectionUNet(nn.Module):
    """Residual correction f(y).

    Three downsampling stages give a receptive field large enough to cover the
    largest inpainting hole used in the experiments (128 px).
    """

    def __init__(self, in_channels, base_channels=48, out_channels=3, zero_init=True):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 4
        self.enc1 = _conv_block(in_channels, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.middle = _conv_block(c3, c4)
        self.dec3 = _conv_block(c4 + c3, c3)
        self.dec2 = _conv_block(c3 + c2, c2)
        self.dec1 = _conv_block(c2 + c1, c1)
        self.out = nn.Conv2d(c1, out_channels, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(self.pool(h1))
        h3 = self.enc3(self.pool(h2))
        m = self.middle(self.pool(h3))
        d = self.dec3(torch.cat([self._up(m, h3), h3], dim=1))
        d = self.dec2(torch.cat([self._up(d, h2), h2], dim=1))
        d = self.dec1(torch.cat([self._up(d, h1), h1], dim=1))
        return self.out(d)


class BridgeNet(nn.Module):
    """Observation -> latent at the truncation step t*."""

    def __init__(self, param_type="hybrid", cond_channels=3, image_size=256,
                 out_channels=3, rank=256, work_size=None, base_channels=48):
        super().__init__()
        if param_type not in PARAM_TYPES:
            raise ValueError(f"param_type must be one of {PARAM_TYPES}")
        self.param_type = param_type
        self.linear_frozen = False
        self.use_correction = True

        if param_type in ("linear", "hybrid"):
            self.linear = LowRankLinear(cond_channels, out_channels, image_size,
                                        rank, work_size)
        if param_type in ("nonlinear", "hybrid"):
            extra = out_channels if param_type == "hybrid" else 0
            self.correction = CorrectionUNet(cond_channels + extra, base_channels,
                                             out_channels,
                                             zero_init=(param_type == "hybrid"))

    def freeze_linear(self):
        if hasattr(self, "linear"):
            for p in self.linear.parameters():
                p.requires_grad_(False)
            self.linear_frozen = True

    def linear_parameters(self):
        return self.linear.parameters()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, cond):
        if self.param_type == "linear":
            return self.linear(cond)
        if self.param_type == "nonlinear":
            return self.correction(cond)

        if not self.use_correction:
            return self.linear(cond)
        if self.linear_frozen:
            with torch.no_grad():
                base = self.linear(cond)
        else:
            base = self.linear(cond)
        return base + self.correction(torch.cat([cond, base], dim=1))

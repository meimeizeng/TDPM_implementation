"""Network building blocks shared by the three restoration tasks.

All tasks use the same conditional UNet; they differ only in how many
conditioning channels are concatenated to the noisy input and, for
deblurring, in an extra embedding produced from the blur kernel.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, gain=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(gain))
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, gain=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight, gain=math.sqrt(gain))
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)


class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(num_groups, channels), channels)

    def forward(self, x):
        return self.norm(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.qkv = Linear(channels, 3 * channels)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = Linear(channels, channels, gain=1e-10)

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.norm(x).permute(0, 2, 3, 1)
        qkv = self.qkv(z).view(b, h * w, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (c ** 0.5))
        z = torch.matmul(attn, v)
        z = self.proj(z).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + z


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 3, stride=2)

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.conv(self.upsample(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, attention, dropout):
        super().__init__()
        self.use_attention = attention
        self.channels_match = in_channels == out_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.act_emb = nn.SiLU()
        self.emb_proj = Linear(emb_dim, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, 3, gain=1e-10)
        if not self.channels_match:
            self.skip = Linear(in_channels, out_channels)
        if attention:
            self.attention = SelfAttentionBlock(out_channels)

    def forward(self, x, emb):
        z = self.conv1(self.act1(self.norm1(x)))
        z = z + self.emb_proj(self.act_emb(emb)).reshape(x.shape[0], self.out_channels, 1, 1)
        z = self.conv2(self.dropout(self.act2(self.norm2(z))))
        if not self.channels_match:
            x = self.skip(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = x + z
        return self.attention(out) if self.use_attention else out


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dim = dim
        self.linear1 = Linear(dim, expansion * dim)
        self.act = nn.SiLU()
        self.linear2 = Linear(expansion * dim, expansion * dim)
        scale = torch.log(torch.tensor(10000.0)) / (dim // 2 - 1)
        freq = torch.exp(torch.arange(0, dim // 2) * -scale)
        self.register_buffer("freq", freq.reshape(1, -1))

    def forward(self, t):
        x = t.reshape(-1, 1).float() * self.freq
        emb = torch.cat((torch.sin(x), torch.cos(x)), dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.linear2(self.act(self.linear1(emb)))


class KernelEncoder(nn.Module):
    """Encodes a blur kernel into a vector added to the timestep embedding."""

    def __init__(self, emb_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(Linear(64, emb_dim), nn.SiLU(), Linear(emb_dim, emb_dim))

    def forward(self, kernel):
        if kernel.dim() == 3:
            kernel = kernel.unsqueeze(1)
        return self.mlp(self.features(kernel))


class ConditionalUNet(nn.Module):
    """Noise-prediction UNet conditioned by channel concatenation.

    Args:
        image_size: spatial resolution the network is trained at, used only to
            decide at which levels self-attention is applied.
        in_channels: 3 (noisy image) + number of conditioning channels.
    """

    def __init__(self, image_size, in_channels, out_channels=3, base_channels=64,
                 channel_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16,),
                 dropout=0.0):
        super().__init__()
        self.emb_dim = base_channels * 4
        self.time_embedding = TimestepEmbedding(base_channels, 4)
        self.input_conv = Conv2d(in_channels, base_channels, 3)

        self.down = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.up = nn.ModuleList()

        attn_resolutions = tuple(attn_resolutions)
        channels = base_channels
        skip_channels = [channels]
        resolution = image_size

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResidualBlock(channels, out_ch, self.emb_dim,
                                               resolution in attn_resolutions, dropout))
                channels = out_ch
                skip_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.down.append(DownBlock(channels, channels))
                skip_channels.append(channels)
                resolution //= 2

        self.middle.append(ResidualBlock(channels, channels, self.emb_dim, True, dropout))
        self.middle.append(ResidualBlock(channels, channels, self.emb_dim, False, dropout))

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResidualBlock(channels + skip_channels.pop(), out_ch,
                                             self.emb_dim,
                                             resolution in attn_resolutions, dropout))
                channels = out_ch
            if level != 0:
                self.up.append(UpBlock(channels, channels))
                resolution *= 2

        self.out_norm = GroupNorm(channels)
        self.out_act = nn.SiLU()
        self.out_conv = Conv2d(channels, out_channels, 3)

    def forward(self, x, cond=None, t=None, extra_emb=None):
        emb = self.time_embedding(t)
        if extra_emb is not None:
            emb = emb + extra_emb

        h = self.input_conv(x if cond is None else torch.cat([x, cond], dim=1))
        skips = [h]
        for module in self.down:
            h = module(h, emb) if isinstance(module, ResidualBlock) else module(h)
            skips.append(h)
        for module in self.middle:
            h = module(h, emb)
        for module in self.up:
            if isinstance(module, ResidualBlock):
                h = module(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = module(h)
        return self.out_conv(self.out_act(self.out_norm(h)))


class KernelConditionedUNet(ConditionalUNet):
    """Conditional UNet with an additional non-blind blur kernel embedding."""

    def __init__(self, *args, use_kernel_embedding=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_encoder = KernelEncoder(self.emb_dim) if use_kernel_embedding else None

    def forward(self, x, cond=None, t=None, kernel=None):
        extra = None
        if self.kernel_encoder is not None and kernel is not None:
            extra = self.kernel_encoder(kernel)
        return super().forward(x, cond, t, extra_emb=extra)

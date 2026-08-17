"""Gaussian blur degradation used by the DIV2K deblurring task.

Protocol (must match the benchmark generator exactly):

  1. the kernel is obtained by running ``scipy.ndimage.gaussian_filter`` on a
     delta image, not from an analytic formula;
  2. kernel support is 61x61 and the kernel is NOT renormalised - scipy
     truncates at 4 sigma without compensating, so ``k.sum()`` is slightly
     below 1 and that gain is part of the operator;
  3. sigma is drawn uniformly from [2.5, 10] during training;
  4. the convolution uses circular ("wrap") boundaries;
  5. blur first, then add noise;
  6. the noise standard deviation 0.05 is defined on the [0,1] scale;
  7. the observation is not clipped.

Two mathematically equivalent implementations of the circular convolution are
available: an FFT one and one built from circular padding plus ``conv2d``. The
second avoids cuFFT, which fails on some torch/CUDA combinations. Selection is
automatic; it can be forced with ``DEBLUR_CONV_IMPL=fft|conv|auto``.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

KERNEL_SIZE = 61
NOISE_LEVEL = 0.05

_IMPL = os.environ.get("DEBLUR_CONV_IMPL", "auto").lower()
_fft_disabled = False


def gaussian_kernel(sigma, kernel_size=KERNEL_SIZE):
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required to build blur kernels: pip install scipy")
    delta = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    delta[kernel_size // 2, kernel_size // 2] = 1.0
    return ndimage.gaussian_filter(delta, sigma=float(sigma)).astype(np.float32)


def sample_sigma(rng=None, low=2.5, high=10.0):
    rng = rng if rng is not None else np.random
    return float(rng.uniform(low, high))


def _kernel_to_otf(kernel, height, width):
    batch, kh, kw = kernel.shape
    otf = torch.zeros(batch, 1, height, width, device=kernel.device, dtype=kernel.dtype)
    otf[:, 0, :kh, :kw] = kernel
    otf = torch.roll(otf, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    return torch.fft.fftn(otf, dim=(-2, -1))


def _blur_fft(x, kernel, adjoint=False):
    _, _, height, width = x.shape
    otf = _kernel_to_otf(kernel.to(x.device, x.dtype), height, width)
    if adjoint:
        otf = torch.conj(otf)
    return torch.real(torch.fft.ifftn(otf * torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1)))


def _blur_conv(x, kernel, adjoint=False):
    batch, channels, height, width = x.shape
    kernel = kernel.to(x.device, x.dtype)
    if kernel.dim() == 2:
        kernel = kernel[None].expand(batch, -1, -1)
    kh, kw = kernel.shape[-2:]
    if not adjoint:
        kernel = torch.flip(kernel, dims=(-2, -1))

    ph, pw = kh // 2, kw // 2
    padded = F.pad(x, (pw, kw - 1 - pw, ph, kh - 1 - ph), mode="circular")
    weight = kernel.reshape(batch, 1, 1, kh, kw).expand(batch, channels, 1, kh, kw)
    weight = weight.reshape(batch * channels, 1, kh, kw).contiguous()
    out = F.conv2d(padded.reshape(1, batch * channels, padded.shape[-2], padded.shape[-1]),
                   weight, groups=batch * channels)
    return out.reshape(batch, channels, height, width)


def _dispatch(x, kernel, adjoint):
    global _fft_disabled
    if kernel.dim() == 2:
        kernel = kernel[None].expand(x.shape[0], -1, -1)
    if _IMPL == "conv":
        return _blur_conv(x, kernel, adjoint)
    if _IMPL == "fft":
        return _blur_fft(x, kernel, adjoint)
    if _fft_disabled:
        return _blur_conv(x, kernel, adjoint)
    try:
        return _blur_fft(x, kernel, adjoint)
    except RuntimeError as exc:
        if "cufft" in str(exc).lower():
            _fft_disabled = True
            print("[deblur] cuFFT failed, switching to the circular-convolution "
                  f"implementation (mathematically identical). Original error: {exc}")
            return _blur_conv(x, kernel, adjoint)
        raise


def blur(x01, kernel):
    """Circular-boundary blur. x01: (B,3,H,W); kernel: (B,kh,kw) or (kh,kw)."""
    return _dispatch(x01, kernel, adjoint=False)


def adjoint(y01, kernel):
    """A^T y, used as an extra conditioning channel."""
    return _dispatch(y01, kernel, adjoint=True)


def degrade(x, kernel, noise_std=NOISE_LEVEL, generator=None):
    """[-1,1] clean image -> [-1,1] observation, without clipping."""
    x01 = (x + 1.0) * 0.5
    y01 = blur(x01, kernel)
    if noise_std > 0:
        if generator is not None:
            noise = torch.randn(y01.shape, generator=generator).to(y01.device, y01.dtype)
        else:
            noise = torch.randn_like(y01)
        y01 = y01 + noise * noise_std
    return y01 * 2.0 - 1.0


def build_condition(y, kernel, use_adjoint=True):
    """Image-domain conditioning: y, optionally concatenated with A^T y."""
    if not use_adjoint:
        return y
    y01 = (y + 1.0) * 0.5
    return torch.cat([y, adjoint(y01, kernel) * 2.0 - 1.0], dim=1)


def normalize_kernel(kernel, enabled):
    """Rescale the kernel to sum to one for the network-facing representation."""
    if not enabled:
        return kernel
    total = kernel.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return kernel / total


def dc_projection(pred, y, kernel):
    """Enforce mean(x) = mean(y) / k.sum() per channel.

    The DC gain of a blur kernel is exactly ``k.sum()``, so this is a hard,
    verifiable data-consistency constraint. It corrects global colour or
    brightness drift; it does not restore detail. Pass the raw (unnormalised)
    kernel, since the gain lives in its sum.
    """
    p01 = (pred + 1) * 0.5
    y01 = (y + 1) * 0.5
    gain = kernel.sum(dim=(-2, -1)).clamp(min=1e-8).reshape(-1, 1, 1, 1)
    target = y01.mean(dim=(2, 3), keepdim=True) / gain
    current = p01.mean(dim=(2, 3), keepdim=True)
    return (p01 + (target - current)) * 2 - 1


def self_test(device="cpu"):
    """Check both implementations against scipy's wrap-mode convolution."""
    print(f"[self-test] device={device} DEBLUR_CONV_IMPL={_IMPL}")
    rng = np.random.RandomState(0)
    image = rng.rand(64, 64, 3).astype(np.float32)
    kernel = gaussian_kernel(4.0)
    reference = ndimage.convolve(image, kernel[..., None], mode="wrap")

    x = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
    k = torch.from_numpy(kernel)[None].to(device)

    out_conv = _blur_conv(x, k)[0].permute(1, 2, 0).cpu().numpy()
    err_conv = np.abs(out_conv - reference).max()
    print(f"  circular-conv vs scipy: max error {err_conv:.3e}")
    assert err_conv < 1e-4, "circular-convolution implementation disagrees with scipy"

    try:
        out_fft = _blur_fft(x, k)[0].permute(1, 2, 0).cpu().numpy()
        err_fft = np.abs(out_fft - reference).max()
        print(f"  fft           vs scipy: max error {err_fft:.3e}")
        assert err_fft < 1e-4, "FFT implementation disagrees with scipy"
    except RuntimeError as exc:
        print(f"  fft unavailable on this machine ({exc}); the conv path takes over")

    for sigma in (3.0, 6.0, 9.0):
        k_sum = gaussian_kernel(sigma).sum()
        print(f"  sigma={sigma:4.1f}  k.sum()={k_sum:.6f}")
    print("[self-test] passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    self_test(parser.parse_args().device)

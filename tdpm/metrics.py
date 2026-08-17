"""Image quality metrics.

PSNR and SSIM are computed after quantisation to uint8 RGB over the full image
(no border cropping, no conversion to the Y channel), so that numbers reported
here are directly comparable with the benchmark scoring script in
``tools/score_folder.py``.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def to_uint8(x):
    """(B,3,H,W) in [-1,1] -> (B,H,W,3) uint8."""
    x = ((x + 1.0) * 0.5).clamp(0, 1).detach().cpu().numpy()
    return (np.transpose(x, (0, 2, 3, 1)) * 255).round().astype(np.uint8)


def ssim_uint8(a, b):
    try:
        return sk_ssim(a, b, channel_axis=2, data_range=255)
    except TypeError:
        return sk_ssim(a, b, multichannel=True, data_range=255)


class LPIPSMetric:
    def __init__(self, net="alex", device="cpu"):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.net = net

    @torch.no_grad()
    def __call__(self, target, pred):
        return float(self.model(target.clamp(-1, 1), pred.clamp(-1, 1)).mean())


def build_lpips(net="alex", device="cpu"):
    try:
        return LPIPSMetric(net, device)
    except Exception as exc:  # noqa: BLE001 - metric is optional
        print(f"[metrics] LPIPS unavailable ({exc}); perceptual scores skipped")
        return None


def batch_metrics(target, pred, lpips_metric=None):
    """Returns (psnr, ssim, lpips) averaged over the batch."""
    t = to_uint8(target)
    p = to_uint8(pred)
    psnr = [sk_psnr(t[i], p[i], data_range=255) for i in range(t.shape[0])]
    ssim = [ssim_uint8(t[i], p[i]) for i in range(t.shape[0])]
    lpips_value = float(lpips_metric(target, pred)) if lpips_metric is not None else float("nan")
    return float(np.mean(psnr)), float(np.mean(ssim)), lpips_value


def masked_psnr(target, pred, mask):
    """PSNR restricted to the masked region, in [0,1] scale."""
    t = (target + 1) * 0.5
    p = (pred + 1) * 0.5
    m = mask.expand_as(t)
    mse = (((t - p) ** 2) * m).sum() / m.sum().clamp(min=1.0)
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))

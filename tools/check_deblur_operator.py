#!/usr/bin/env python
"""Check that the training-time degradation matches a benchmark directory.

Run this before training or before filling in a results table:

    python tools/check_deblur_operator.py --bench_dir benchmarks/div2k_gauss

Three checks are performed:

  1. both circular-convolution implementations agree with scipy;
  2. the kernel scale used for training matches the benchmark kernels;
  3. re-applying the operator to the ground truth reproduces the benchmark
     observation up to the noise, i.e. mean |A(x) - y| should be close to
     0.0798 on the [-1,1] scale, the expected absolute value of N(0, 0.1^2).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdpm.degradations.deblur import (adjoint, degrade, gaussian_kernel,
                                      self_test)

EXPECTED_RESIDUAL = 0.0798


def _find(directory, stem, prefer_npy=False):
    extensions = [".npy", ".png", ".jpg"] if prefer_npy else [".png", ".jpg", ".npy"]
    for ext in extensions:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _load_image(path):
    from PIL import Image
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)) * 2 - 1


def check_adjoint(device):
    from scipy import ndimage

    rng = np.random.RandomState(0)
    image = rng.rand(64, 64, 3).astype(np.float32)
    kernel = gaussian_kernel(4.0)
    reference = ndimage.correlate(image, kernel[..., None], mode="wrap")
    out = adjoint(torch.from_numpy(image).permute(2, 0, 1)[None].to(device),
                  torch.from_numpy(kernel)[None].to(device))[0]
    error = np.abs(out.permute(1, 2, 0).cpu().numpy() - reference).max()
    status = "ok" if error < 1e-4 else "FAILED - A^T y is wrong"
    print(f"[adjoint] max error vs scipy: {error:.3e}  {status}")


def check_benchmark(bench_dir, device, max_images=8):
    gt_dir = os.path.join(bench_dir, "gt")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir))[:max_images]

    train_sums = [gaussian_kernel(s).sum() for s in (3.0, 6.0, 9.0)]
    print(f"[kernels] training kernel sums: {[f'{v:.6f}' for v in train_sums]}")

    bench_sums, residuals = [], []
    for stem in stems:
        gt_path = _find(gt_dir, stem)
        lq_path = _find(os.path.join(bench_dir, "lq"), stem, prefer_npy=True)
        kernel_path = _find(os.path.join(bench_dir, "kernel"), stem, prefer_npy=True)
        if not (gt_path and lq_path and kernel_path):
            print(f"  skipping {stem}: incomplete triple")
            continue

        kernel_np = np.load(kernel_path).squeeze().astype(np.float32)
        bench_sums.append(float(kernel_np.sum()))

        y = np.load(lq_path).astype(np.float32) if lq_path.endswith(".npy") else _load_image(lq_path)
        if y.ndim == 3 and y.shape[0] not in (1, 3):
            y = np.transpose(y, (2, 0, 1))
        if y.max() <= 1.01 and y.min() >= -0.01:
            y = y * 2 - 1

        x = torch.from_numpy(_load_image(gt_path))[None].to(device)
        kernel = torch.from_numpy(kernel_np)[None].to(device)
        noiseless = degrade(x, kernel, noise_std=0.0)
        residuals.append(float(np.abs(noiseless[0].cpu().numpy() - y).mean()))
        print(f"  {stem}: mean |A(x) - y| = {residuals[-1]:.4f}")

    if bench_sums:
        gap = abs(np.mean(bench_sums) - np.mean(train_sums))
        if gap < 1e-3:
            print("[kernels] scales agree; --set data.normalize_kernel can stay false")
        else:
            print(f"[kernels] scales differ by {gap:.4f}; "
                  f"evaluate with data.normalize_kernel=true")
    if residuals:
        mean = float(np.mean(residuals))
        verdict = "operator aligned" if abs(mean - EXPECTED_RESIDUAL) < 0.01 else \
            "MISMATCH - check kernel placement, boundary mode and value range"
        print(f"[operator] mean residual {mean:.4f} (expected ~{EXPECTED_RESIDUAL}) -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    self_test(args.device)
    check_adjoint(args.device)
    if args.bench_dir:
        check_benchmark(args.bench_dir, args.device)


if __name__ == "__main__":
    main()

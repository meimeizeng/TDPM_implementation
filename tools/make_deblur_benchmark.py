#!/usr/bin/env python
"""Build the fixed DIV2K deblurring benchmark.

Produces a directory that every method is evaluated on:

    <out>/gt/<stem>.png        ground truth crop, uint8
    <out>/lq/<stem>.npy        observation, float32, [0,1], not clipped
    <out>/lq/<stem>.png        the same observation, clipped, for viewing only
    <out>/kernel/<stem>.npy    the blur kernel, unnormalised

The .npy observation is the ground truth input: the PNG version is clipped and
quantised and would change the metrics slightly.

    python tools/make_deblur_benchmark.py --source data/DIV2K/DIV2K_valid_HR \
        --out benchmarks/div2k_gauss --num 100
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdpm.data.div2k_deblur import list_images
from tdpm.degradations.deblur import (KERNEL_SIZE, NOISE_LEVEL, degrade,
                                      gaussian_kernel)


def center_crop(image, size):
    width, height = image.size
    if width < size or height < size:
        scale = size / min(width, height)
        image = image.resize((max(size, int(round(width * scale))),
                              max(size, int(round(height * scale)))), Image.BICUBIC)
        width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def main():
    parser = argparse.ArgumentParser("build the deblurring benchmark")
    parser.add_argument("--source", required=True, help="folder of clean images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--sigma_low", type=float, default=2.5)
    parser.add_argument("--sigma_high", type=float, default=10.0)
    parser.add_argument("--noise_std", type=float, default=NOISE_LEVEL)
    parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    gt_dir = os.path.join(args.out, "gt")
    lq_dir = os.path.join(args.out, "lq")
    kernel_dir = os.path.join(args.out, "kernel")
    for directory in (gt_dir, lq_dir, kernel_dir):
        os.makedirs(directory, exist_ok=True)

    paths = list_images(args.source)[:args.num]
    rng = np.random.RandomState(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    print(f"[bench] {len(paths)} images -> {args.out}")

    for index, path in enumerate(paths):
        stem = f"{index:05d}"
        image = center_crop(Image.open(path).convert("RGB"), args.patch_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        clean = torch.from_numpy(np.transpose(array, (2, 0, 1)))[None] * 2 - 1

        sigma = float(rng.uniform(args.sigma_low, args.sigma_high))
        kernel_np = gaussian_kernel(sigma, args.kernel_size)
        kernel = torch.from_numpy(kernel_np)[None]

        observation = degrade(clean, kernel, args.noise_std, generator=generator)
        observation01 = ((observation + 1) * 0.5)[0].numpy()

        image.save(os.path.join(gt_dir, stem + ".png"))
        np.save(os.path.join(lq_dir, stem + ".npy"), observation01.astype(np.float32))
        preview = (np.clip(observation01, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(np.transpose(preview, (1, 2, 0))).save(
            os.path.join(lq_dir, stem + ".png"))
        np.save(os.path.join(kernel_dir, stem + ".npy"), kernel_np)

        if (index + 1) % 20 == 0 or index + 1 == len(paths):
            print(f"  {index + 1}/{len(paths)} (last sigma={sigma:.2f})", flush=True)

    print(f"[bench] done. Kernel sums stay slightly below 1 by construction.")


if __name__ == "__main__":
    main()

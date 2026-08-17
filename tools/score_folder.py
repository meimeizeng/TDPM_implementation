#!/usr/bin/env python
"""Unified scorer for a folder of restored images.

Every method must be scored with this script so that all reported numbers share
one protocol:

  PSNR   uint8 RGB, full image, no border cropping, no Y-channel conversion
  SSIM   skimage.structural_similarity(channel_axis=2, data_range=255)
  LPIPS  one backbone for all methods (default: alex)

    python tools/score_folder.py --gt bench/gt --pred results/tdpm --name tdpm

Results are printed, written per image to ``results/<name>.csv`` and appended
to ``leaderboard.csv``. The ``--note`` field is free text; use it to record
inference-time settings so that rows stay comparable weeks later.
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def find_image(directory, stem):
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


def compute_ssim(a, b):
    try:
        return sk_ssim(a, b, channel_axis=2, data_range=255)
    except TypeError:
        return sk_ssim(a, b, multichannel=True, data_range=255)


def build_lpips(net, device):
    try:
        import lpips as lpips_lib
        import torch
    except ImportError:
        print("[warn] lpips or torch missing; perceptual metric skipped")
        return None

    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    model = lpips_lib.LPIPS(net=net).to(resolved).eval()

    def score(a_u8, b_u8):
        def prepare(x):
            t = torch.from_numpy(x.astype(np.float32) / 255.0)
            return (t.permute(2, 0, 1)[None] * 2 - 1).to(resolved)

        with torch.no_grad():
            return float(model(prepare(a_u8), prepare(b_u8)).item())

    return score


def main():
    parser = argparse.ArgumentParser("score restored images")
    parser.add_argument("--gt", required=True, help="ground truth directory")
    parser.add_argument("--pred", required=True, help="restored image directory")
    parser.add_argument("--name", required=True, help="method name for the leaderboard")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--note", default="")
    parser.add_argument("--no_lpips", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.gt):
        sys.exit(f"Ground truth directory not found: {args.gt}")
    if not os.path.isdir(args.pred):
        sys.exit(f"Prediction directory not found: {args.pred}")

    os.makedirs(args.out_dir, exist_ok=True)
    leaderboard = args.leaderboard or os.path.join(args.out_dir, "leaderboard.csv")

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.gt)
                   if f.lower().endswith(IMAGE_EXTENSIONS))
    if args.limit > 0:
        stems = stems[:args.limit]
    if not stems:
        sys.exit(f"No images under {args.gt}")

    lpips_fn = None if args.no_lpips else build_lpips(args.lpips_net, args.device)
    print(f"[score] method={args.name}  images={len(stems)}  "
          f"lpips={args.lpips_net if lpips_fn else 'skipped'}")

    rows, missing = [], []
    for stem in stems:
        gt_path = find_image(args.gt, stem)
        pred_path = find_image(args.pred, stem)
        if pred_path is None:
            missing.append(stem)
            continue

        gt = load_uint8(gt_path)
        pred = load_uint8(pred_path)
        if gt.shape != pred.shape:
            sys.exit(f"Shape mismatch for {stem}: gt={gt.shape} pred={pred.shape}. "
                     f"Restored images must keep the ground truth resolution.")

        psnr = sk_psnr(gt, pred, data_range=255)
        ssim = compute_ssim(gt, pred)
        lpips_value = lpips_fn(gt, pred) if lpips_fn else float("nan")
        rows.append((stem, psnr, ssim, lpips_value))
        print(f"  {stem}: PSNR {psnr:6.2f}  SSIM {ssim:.4f}  LPIPS {lpips_value:.4f}")

    if missing:
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        print(f"[warn] {len(missing)} images missing from the prediction folder: {preview}")
    if not rows:
        sys.exit("Nothing matched; check that predictions share the ground truth filenames.")

    psnr_mean = float(np.mean([r[1] for r in rows]))
    ssim_mean = float(np.mean([r[2] for r in rows]))
    lpips_mean = float(np.nanmean([r[3] for r in rows]))

    print("\n" + "=" * 52)
    print(f"  {args.name}  ({len(rows)}/{len(stems)} images)")
    print(f"  PSNR  {psnr_mean:.3f}")
    print(f"  SSIM  {ssim_mean:.4f}")
    print(f"  LPIPS {lpips_mean:.4f}  (backbone={args.lpips_net})")
    print("=" * 52 + "\n")

    detail = os.path.join(args.out_dir, f"{args.name}.csv")
    with open(detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips"])
        for stem, psnr, ssim, lpips_value in rows:
            writer.writerow([stem, f"{psnr:.4f}", f"{ssim:.6f}", f"{lpips_value:.6f}"])
    print(f"[score] per-image results -> {detail}")

    header = ["name", "psnr", "ssim", "lpips", "lpips_net", "n_images",
              "pred_dir", "note", "timestamp"]
    is_new = not os.path.exists(leaderboard)
    with open(leaderboard, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow([args.name, f"{psnr_mean:.3f}", f"{ssim_mean:.4f}",
                         f"{lpips_mean:.4f}", args.lpips_net, len(rows), args.pred,
                         args.note, datetime.now().strftime("%Y-%m-%d %H:%M")])
    print(f"[score] appended to {leaderboard}")


if __name__ == "__main__":
    main()

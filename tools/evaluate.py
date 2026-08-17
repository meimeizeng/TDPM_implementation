#!/usr/bin/env python
"""Evaluation and image export.

Compare samplers on the validation split:

    python tools/evaluate.py --config configs/deblur_div2k.yaml \
        --methods ddim50 dpm20 tdpm

Export restored images so that final metrics can be computed with the shared
scorer (this is how the numbers in the paper are produced):

    python tools/evaluate.py --config configs/inpaint_ffhq256.yaml \
        --methods tdpm --export_dir results/inpaint_tdpm --num 200
    python tools/score_folder.py --gt results/inpaint_tdpm/gt \
        --pred results/inpaint_tdpm/pred --name tdpm

For deblurring the fixed benchmark built by ``tools/make_deblur_benchmark.py``
can be used instead of the validation loader with ``--bench_dir``.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdpm.config import load_config
from tdpm.diffusion import GaussianDiffusion
from tdpm.engine.evaluate import evaluate
from tdpm.engine.sampling import run_method
from tdpm.engine.train_backbone import load_backbone
from tdpm.engine.train_bridge import load_bridge
from tdpm.metrics import to_uint8
from tdpm.tasks import build_task
from tdpm.utils import Workspace, resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser("TDPM evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--backbone_ckpt", default=None)
    parser.add_argument("--bridge_ckpt", default=None)
    parser.add_argument("--use_ema", type=int, default=1)
    parser.add_argument("--export_dir", default=None,
                        help="write gt/ and pred/ PNGs instead of printing a table")
    parser.add_argument("--bench_dir", default=None,
                        help="deblurring only: fixed benchmark with gt/, lq/, kernel/")
    parser.add_argument("--dc_fix", action="store_true",
                        help="deblurring only: apply the DC projection mean(x)=mean(y)/k.sum()")
    return parser.parse_args()


def build(args):
    cfg = load_config(args.config, args.overrides)
    cfg.setdefault("eval", {})
    if args.methods:
        cfg["eval"]["methods"] = args.methods
    if args.num:
        cfg["eval"]["num_images"] = args.num
    cfg["eval"].setdefault("methods", ["ddim50", "tdpm"])

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    workspace = Workspace(cfg, "eval")
    task = build_task(cfg, device)
    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.get("schedule", "linear"),
        pred_type=cfg.diffusion.get("pred_type", "eps"),
        zero_terminal_snr=bool(cfg.diffusion.get("zero_terminal_snr", False)),
        beta_scale_by_steps=bool(cfg.diffusion.get("beta_scale_by_steps", True)),
        device=device,
    )
    backbone = load_backbone(cfg, task, workspace, device, args.backbone_ckpt,
                             use_ema=bool(args.use_ema))
    bridge = load_bridge(cfg, task, workspace, device, args.bridge_ckpt,
                         use_ema=bool(args.use_ema))
    return cfg, task, workspace, diffusion, backbone, bridge, device


def save_uint8(tensor, path):
    Image.fromarray(to_uint8(tensor[None])[0]).save(path)


@torch.no_grad()
def export_from_loader(args, cfg, task, diffusion, backbone, bridge, val_loader):
    method = cfg["eval"]["methods"][0]
    gt_dir = os.path.join(args.export_dir, "gt")
    pred_dir = os.path.join(args.export_dir, "pred")
    obs_dir = os.path.join(args.export_dir, "observation")
    for directory in (gt_dir, pred_dir, obs_dir):
        os.makedirs(directory, exist_ok=True)

    limit = cfg["eval"].get("num_images", 100)
    index = 0
    for raw in val_loader:
        batch = task.prepare_batch(raw, train=False)
        prediction, nfe, label = run_method(method, cfg, task, diffusion, backbone,
                                            bridge, batch)
        for i in range(prediction.shape[0]):
            if index >= limit:
                break
            stem = f"{index:05d}"
            save_uint8(batch.x0[i], os.path.join(gt_dir, stem + ".png"))
            save_uint8(batch.observation[i], os.path.join(obs_dir, stem + ".png"))
            save_uint8(prediction[i], os.path.join(pred_dir, stem + ".png"))
            index += 1
        print(f"  exported {index}/{limit}", flush=True)
        if index >= limit:
            break
    print(f"[export] {label} ({nfe} NFE) -> {args.export_dir}")
    print(f"[export] score with:\n  python tools/score_folder.py --gt {gt_dir} "
          f"--pred {pred_dir} --name {method}")


def _find(directory, stem, prefer_npy=False):
    extensions = [".npy", ".png", ".jpg"] if prefer_npy else [".png", ".jpg", ".npy"]
    for ext in extensions:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _load_observation(path, device):
    if path.endswith(".npy"):
        array = np.load(path).astype(np.float32)
        if array.ndim == 3 and array.shape[0] not in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        if array.max() <= 1.01 and array.min() >= -0.01:
            array = array * 2 - 1
    else:
        print(f"  [warning] {path} is a clipped PNG observation; metrics will drift")
        image = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        array = np.transpose(image, (2, 0, 1)) * 2 - 1
    return torch.from_numpy(array)[None].to(device)


@torch.no_grad()
def export_from_benchmark(args, cfg, task, diffusion, backbone, bridge, device):
    from tdpm.degradations.deblur import dc_projection

    method = cfg["eval"]["methods"][0]
    gt_dir = os.path.join(args.bench_dir, "gt")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir))
    limit = cfg["eval"].get("num_images", 0)
    if limit:
        stems = stems[:limit]

    pred_dir = os.path.join(args.export_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    print(f"[bench] {len(stems)} images | method={method} | out={pred_dir}")

    for count, stem in enumerate(stems):
        lq_path = _find(os.path.join(args.bench_dir, "lq"), stem, prefer_npy=True)
        kernel_path = _find(os.path.join(args.bench_dir, "kernel"), stem, prefer_npy=True)
        if lq_path is None or kernel_path is None:
            print(f"  skipping {stem}: missing observation or kernel")
            continue

        y = _load_observation(lq_path, device)
        kernel = torch.from_numpy(
            np.load(kernel_path).squeeze().astype(np.float32))[None].to(device)
        batch = task.batch_from_observation(y, kernel)
        prediction, _, _ = run_method(method, cfg, task, diffusion, backbone, bridge, batch)
        if args.dc_fix:
            prediction = dc_projection(prediction, y, kernel)
        save_uint8(prediction[0], os.path.join(pred_dir, stem + ".png"))
        if (count + 1) % 10 == 0 or count + 1 == len(stems):
            print(f"  {count + 1}/{len(stems)}", flush=True)

    print(f"[bench] score with:\n  python tools/score_folder.py --gt {gt_dir} "
          f"--pred {pred_dir} --name {method}")


def main():
    args = parse_args()
    cfg, task, workspace, diffusion, backbone, bridge, device = build(args)

    if args.bench_dir:
        if cfg.task != "deblur":
            raise SystemExit("--bench_dir is only supported for the deblurring task")
        if not args.export_dir:
            raise SystemExit("--bench_dir requires --export_dir")
        if len(cfg["eval"]["methods"]) != 1:
            raise SystemExit("benchmark export needs exactly one --methods entry")
        export_from_benchmark(args, cfg, task, diffusion, backbone, bridge, device)
        return

    _, val_loader = task.build_dataloaders()
    if args.export_dir:
        if len(cfg["eval"]["methods"]) != 1:
            raise SystemExit("export needs exactly one --methods entry")
        export_from_loader(args, cfg, task, diffusion, backbone, bridge, val_loader)
        return

    evaluate(cfg, task, workspace, diffusion, backbone, bridge, val_loader, device)


if __name__ == "__main__":
    main()

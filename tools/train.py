#!/usr/bin/env python
"""Entry point for both training stages.

    python tools/train.py --config configs/deblur_div2k.yaml --stage backbone
    python tools/train.py --config configs/deblur_div2k.yaml --stage bridge
    python tools/train.py --config configs/deblur_div2k.yaml --stage all

Any configuration field can be overridden from the command line:

    python tools/train.py --config configs/deblur_div2k.yaml --stage bridge \
        --set truncation.t_star=300 bridge.param_type=linear
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdpm.config import load_config
from tdpm.diffusion import GaussianDiffusion
from tdpm.engine.evaluate import evaluate
from tdpm.engine.train_backbone import load_backbone, train_backbone
from tdpm.engine.train_bridge import load_bridge, train_bridge
from tdpm.tasks import build_task
from tdpm.utils import Workspace, resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser("TDPM training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all",
                        choices=["backbone", "bridge", "eval", "all"])
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="config overrides, e.g. bridge.omega=0.5")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.stage = args.stage

    seed_everything(cfg.runtime.seed)
    torch.backends.cudnn.benchmark = True
    device = resolve_device(cfg.runtime.device)

    workspace = Workspace(cfg, args.stage)
    task = build_task(cfg, device)
    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.get("schedule", "linear"),
        pred_type=cfg.diffusion.get("pred_type", "eps"),
        zero_terminal_snr=bool(cfg.diffusion.get("zero_terminal_snr", False)),
        beta_scale_by_steps=bool(cfg.diffusion.get("beta_scale_by_steps", True)),
        device=device,
    )

    print(f"--- task={cfg.task} exp={cfg.exp_name} stage={args.stage} "
          f"T={diffusion.num_timesteps} pred={diffusion.pred_type} device={device} ---")
    print(f"[workspace] {workspace.root}")

    train_loader, val_loader = task.build_dataloaders()

    backbone, bridge = None, None
    if args.stage in ("backbone", "all"):
        backbone = train_backbone(cfg, task, workspace, diffusion, train_loader,
                                  val_loader, device)

    if args.stage in ("bridge", "all"):
        if backbone is None:
            backbone = load_backbone(cfg, task, workspace, device,
                                     cfg.bridge.get("backbone_ckpt"))
        bridge = train_bridge(cfg, task, workspace, diffusion, backbone, train_loader,
                              val_loader, device)

    if args.stage in ("eval", "all"):
        if backbone is None:
            backbone = load_backbone(cfg, task, workspace, device,
                                     cfg.bridge.get("backbone_ckpt"))
        if bridge is None:
            bridge = load_bridge(cfg, task, workspace, device)
        evaluate(cfg, task, workspace, diffusion, backbone, bridge, val_loader, device)


if __name__ == "__main__":
    main()

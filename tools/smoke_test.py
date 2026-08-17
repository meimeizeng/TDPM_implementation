#!/usr/bin/env python
"""Shape and wiring check for all three tasks, without any data.

Builds a tiny model per task on synthetic tensors and runs one backbone step,
one bridge step and one truncated sampling pass. Useful after changing the
architecture or adding a task; runs on CPU in about a minute.

    python tools/smoke_test.py
    python tools/smoke_test.py --task deblur --device cuda
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdpm.config import load_config
from tdpm.diffusion import GaussianDiffusion
from tdpm.engine.sampling import run_method
from tdpm.losses import masked_mse, weighted_loss
from tdpm.tasks import build_task
from tdpm.utils import count_parameters

CONFIGS = {
    "sr": "configs/sr_imagenet.yaml",
    "inpaint": "configs/inpaint_ffhq256.yaml",
    "deblur": "configs/deblur_div2k.yaml",
}

TINY = [
    "model.base_channels=16",
    "model.channel_mult=[1,2]",
    "model.num_res_blocks=1",
    "model.attn_resolutions=[]",
    "diffusion.num_timesteps=100",
    "bridge.rank=8",
    "bridge.work_size=16",
    "bridge.base_channels=8",
    "truncation.t_star=40",
    "truncation.num_steps=3",
]


def fake_batch(task, cfg, device, size=2):
    resolution = 32
    clean = torch.randn(size, 3, resolution, resolution, device=device).clamp(-1, 1)
    if cfg.task == "sr":
        return (clean, clean.flip(-1))
    if cfg.task == "inpaint":
        mask = torch.zeros(size, 1, resolution, resolution, device=device)
        mask[:, :, 8:20, 8:20] = 1.0
        return (clean * (1 - mask), mask, clean)
    kernel = torch.zeros(size, 9, 9, device=device)
    kernel[:, 4, 4] = 1.0
    return (clean, kernel, torch.ones(size, device=device))


def run(name, device):
    print("=" * 60)
    print(f"task: {name}")
    overrides = list(TINY) + ["data.image_size=32", "data.patch_size=32"]
    cfg = load_config(CONFIGS[name], overrides)
    task = build_task(cfg, device)

    diffusion = GaussianDiffusion(cfg.diffusion.num_timesteps,
                                  cfg.diffusion.get("schedule", "linear"),
                                  cfg.diffusion.get("pred_type", "eps"),
                                  bool(cfg.diffusion.get("zero_terminal_snr", False)),
                                  bool(cfg.diffusion.get("beta_scale_by_steps", True)),
                                  device)

    backbone = task.build_backbone().to(device)
    bridge = task.build_bridge().to(device)
    print(f"  backbone {count_parameters(backbone) / 1e6:.2f}M | "
          f"bridge {count_parameters(bridge) / 1e6:.2f}M ({cfg.bridge.param_type})")

    batch = task.prepare_batch(fake_batch(task, cfg, device), train=True)
    print(f"  x0 {tuple(batch.x0.shape)} | cond {tuple(batch.cond.shape)} "
          f"(expected {3 + task.cond_channels} input channels)")

    # stage 1
    t = torch.randint(0, diffusion.num_timesteps, (batch.x0.shape[0],), device=device)
    noise = torch.randn_like(batch.x0)
    x_t = diffusion.q_sample(batch.x0, t, noise)
    target, weight = diffusion.target_and_weight(batch.x0, noise, t)
    prediction = task.model_fn(backbone, batch)(x_t, t)
    loss = masked_mse(prediction, target, weight)
    loss.backward()
    print(f"  stage 1 loss {float(loss):.4f}  gradients ok")

    # stage 2
    for p in backbone.parameters():
        p.requires_grad_(False)
    t_star = torch.full((batch.x0.shape[0],), cfg.truncation.t_star, device=device,
                        dtype=torch.long)
    latent = bridge(task.bridge_input(batch))
    rec = weighted_loss(latent, diffusion.q_mean(batch.x0, t_star), "smooth_l1")
    noisy = latent + diffusion.sqrt_one_minus_alphas_cumprod[cfg.truncation.t_star] \
        * torch.randn_like(latent)
    x0_hat, _ = diffusion.to_x0_eps(task.model_fn(backbone, batch)(noisy, t_star),
                                    noisy, t_star, clip=False)
    total = rec + cfg.bridge.omega * weighted_loss(x0_hat, batch.x0, "l2")
    total.backward()
    print(f"  stage 2 loss {float(total):.4f}  gradients ok")

    # sampling
    for spec in ("ddim4", "dpm4", "tdpm"):
        with torch.no_grad():
            out, nfe, label = run_method(spec, cfg, task, diffusion, backbone, bridge, batch)
        assert out.shape == batch.x0.shape, f"{spec} returned {out.shape}"
        print(f"  {label:<28} -> {tuple(out.shape)}  ({nfe} NFE)")
    print(f"task {name}: ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", "sr", "inpaint", "deblur"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    names = list(CONFIGS) if args.task == "all" else [args.task]
    for name in names:
        run(name, device)
    print("\nall smoke tests passed")


if __name__ == "__main__":
    main()

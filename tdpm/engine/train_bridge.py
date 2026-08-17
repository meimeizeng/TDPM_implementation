"""Stage 2: train the truncation bridge on top of a frozen backbone.

Objective (Sec. V of the paper):

    L = L_rec + omega * L_con

    L_rec = || bridge(y) - E[x_{t*} | x_0] ||_smooth-l1
    L_con = || D(bridge(y) + sigma_{t*} z) - x_0 ||_2

where D is the one-step estimate of the clean image produced by the frozen
backbone. The bridge regresses the *mean* of q(x_{t*} | x_0); the stochastic
component is added back at sampling time, so the model never has to fit
unpredictable noise and sample diversity is preserved.

For the ``hybrid`` parameterisation training runs in two phases: the low-rank
linear map A y + b is fitted first, then frozen while the residual correction
f(y) is trained. This mirrors the decomposition analysed in the paper and makes
the ablation between the three parameterisations a single configuration change.
"""

import os
import time

import torch
import torchvision

from ..ema import EMA
from ..losses import weighted_loss
from ..metrics import batch_metrics, masked_psnr
from ..utils import (append_csv, count_parameters, cosine_lr, infinite_loader,
                     load_checkpoint, prune_checkpoints, set_lr, to01)

CHECKPOINT_PREFIX = "bridge"


def save_bridge(workspace, cfg, bridge, ema, optimizer, step, phase):
    payload = {
        "step": step,
        "phase": phase,
        "model": bridge.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "config": {"bridge": dict(cfg.bridge), "truncation": dict(cfg.truncation),
                   "task": cfg.task},
    }
    torch.save(payload, workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth"))
    torch.save(payload, workspace.ckpt(f"{CHECKPOINT_PREFIX}_step_{step:07d}.pth"))
    prune_checkpoints(workspace.checkpoints, CHECKPOINT_PREFIX, cfg.bridge.get("keep_last", 3))
    print(f"[ckpt] bridge step {step} saved (backbone untouched)", flush=True)


@torch.no_grad()
def visualise(cfg, task, workspace, diffusion, backbone, bridge, batch, step, csv_path):
    from .sampling import bridge_latent, run_method

    bridge.eval()
    started = time.time()
    prediction, nfe, label = run_method("tdpm", cfg, task, diffusion, backbone, bridge, batch)
    elapsed = time.time() - started

    t_star = cfg.truncation.t_star
    latent_mean = bridge_latent(cfg, task, diffusion, bridge, batch, add_noise=False)
    implied_x0 = (latent_mean / diffusion.sqrt_alphas_cumprod[t_star]).clamp(-1, 1)

    psnr, ssim, _ = batch_metrics(batch.x0, prediction)
    row = [step, f"{psnr:.3f}", f"{ssim:.4f}", f"{elapsed:.2f}"]
    header = ["step", "psnr", "ssim", "seconds"]
    if batch.mask is not None:
        row.insert(3, f"{masked_psnr(batch.x0, prediction, batch.mask):.3f}")
        header.insert(3, "psnr_masked")
    print(f"[vis] step {step} {label}: PSNR {psnr:.2f} SSIM {ssim:.3f} "
          f"({nfe} NFE, {elapsed:.2f}s)", flush=True)
    append_csv(csv_path, row, header)

    grid = torch.cat([batch.x0, batch.observation, implied_x0, prediction], dim=0)
    torchvision.utils.save_image(to01(grid),
                                 os.path.join(workspace.vis, f"bridge_{step:07d}.png"),
                                 nrow=batch.x0.shape[0], padding=2)
    bridge.train()


def train_bridge(cfg, task, workspace, diffusion, backbone, train_loader, val_loader, device):
    bridge_cfg = cfg.bridge
    t_star = cfg.truncation.t_star

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    bridge = task.build_bridge().to(device)
    print(f"[model] bridge ({bridge_cfg.param_type}): "
          f"{count_parameters(bridge) / 1e6:.2f}M parameters, t* = {t_star}, "
          f"alpha_bar = {float(diffusion.alphas_cumprod[t_star]):.4f}")

    ema = EMA(bridge, bridge_cfg.get("ema_decay", 0.999))
    optimizer = torch.optim.Adam(bridge.parameters(), lr=bridge_cfg.lr)

    start_step = 0
    resume = bridge_cfg.get("resume", "auto")
    resume_path = (workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth")
                   if resume == "auto" else (None if resume in ("none", None, "") else resume))
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path, device)
        bridge.load_state_dict(ckpt["model"])
        if ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"[resume] bridge resumed from {resume_path} at step {start_step}")
        if start_step >= bridge_cfg.train_steps:
            print("[resume] bridge already finished")
            return ema.ema_model
    else:
        print("[resume] training bridge from scratch")

    from .train_backbone import _slice_batch
    vis_batch = task.prepare_batch(next(iter(val_loader)), train=False)
    vis_batch = _slice_batch(vis_batch, min(bridge_cfg.get("vis_num", 4),
                                            vis_batch.x0.shape[0]))

    loss_csv = workspace.log("bridge_loss.csv")
    vis_csv = workspace.log("bridge_val.csv")
    batches = infinite_loader(train_loader)

    linear_steps = (int(bridge_cfg.train_steps * bridge_cfg.get("linear_ratio", 0.4))
                    if bridge_cfg.param_type == "hybrid" else 0)
    region_weight = bridge_cfg.get("region_weight", 1.0)
    omega = bridge_cfg.get("omega", 0.5)
    rec_norm = bridge_cfg.get("rec_loss", "smooth_l1")
    con_norm = bridge_cfg.get("con_loss", "l2")
    huber_beta = bridge_cfg.get("huber_beta", 1.0)
    predict_mean = bridge_cfg.get("target", "mean") == "mean"

    sqrt_ab = diffusion.sqrt_alphas_cumprod[t_star]
    sqrt_1mab = diffusion.sqrt_one_minus_alphas_cumprod[t_star]

    print(f"--- stage 2: bridge {start_step} -> {bridge_cfg.train_steps} "
          f"(phase A: {linear_steps} steps on A y + b) ---", flush=True)
    started = time.time()
    running = {"total": 0.0, "rec": 0.0, "con": 0.0}
    bridge.train()

    for step in range(start_step, bridge_cfg.train_steps):
        phase_linear = step < linear_steps
        if bridge_cfg.param_type == "hybrid":
            bridge.use_correction = not phase_linear
            if phase_linear:
                _set_requires_grad(bridge.linear, True)
                _set_requires_grad(bridge.correction, False)
                bridge.linear_frozen = False
            elif not bridge.linear_frozen:
                bridge.freeze_linear()
                _set_requires_grad(bridge.correction, True)
                print(f"[bridge] step {step}: A y + b frozen, training f(y)", flush=True)

        set_lr(optimizer, cosine_lr(step, bridge_cfg.lr, bridge_cfg.get("min_lr", 1e-6),
                                    bridge_cfg.train_steps,
                                    bridge_cfg.get("warmup_steps", 0)))

        batch = task.prepare_batch(next(batches), train=True)
        size = batch.x0.shape[0]
        t = torch.full((size,), t_star, device=device, dtype=torch.long)
        weight = task.loss_weight_map(batch, region_weight)

        prediction = bridge(task.bridge_input(batch))
        target = diffusion.q_mean(batch.x0, t) if predict_mean else diffusion.q_sample(batch.x0, t)
        loss_rec = weighted_loss(prediction, target, rec_norm, weight, huber_beta)

        if phase_linear or omega == 0:
            loss_con = torch.zeros((), device=device)
        else:
            if predict_mean:
                latent = prediction + sqrt_1mab * torch.randn_like(prediction)
            else:
                latent = prediction
            output = task.model_fn(backbone, batch)(latent, t)
            x0_hat, _ = diffusion.to_x0_eps(output, latent, t, clip=False)
            loss_con = weighted_loss(x0_hat, batch.x0, con_norm, weight, huber_beta)

        loss = loss_rec + omega * loss_con

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if bridge_cfg.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(bridge.trainable_parameters(), bridge_cfg.grad_clip)
        optimizer.step()
        ema.update()

        running["total"] += float(loss.detach())
        running["rec"] += float(loss_rec.detach())
        running["con"] += float(loss_con.detach())

        if step % bridge_cfg.get("log_every", 100) == 0:
            denom = max(1, bridge_cfg.get("log_every", 100) if step > start_step else 1)
            speed = (step - start_step + 1) / max(1e-6, time.time() - started)
            phase = "linear" if phase_linear else "full"
            print(f"[bridge/{phase}] {step}/{bridge_cfg.train_steps} "
                  f"loss {running['total'] / denom:.4f} "
                  f"(rec {running['rec'] / denom:.4f}, con {running['con'] / denom:.4f}) "
                  f"{speed:.2f} it/s", flush=True)
            append_csv(loss_csv,
                       [step, phase, f"{running['total'] / denom:.6f}",
                        f"{running['rec'] / denom:.6f}", f"{running['con'] / denom:.6f}",
                        f"{optimizer.param_groups[0]['lr']:.3e}"],
                       ["step", "phase", "loss", "rec", "con", "lr"])
            running = {k: 0.0 for k in running}

        if step > start_step and step % bridge_cfg.get("vis_every", 2000) == 0:
            visualise(cfg, task, workspace, diffusion, backbone, ema.ema_model,
                      vis_batch, step, vis_csv)

        if step > start_step and step % bridge_cfg.get("save_every", 2000) == 0:
            save_bridge(workspace, cfg, bridge, ema, optimizer, step,
                        "linear" if phase_linear else "full")

    save_bridge(workspace, cfg, bridge, ema, optimizer, bridge_cfg.train_steps - 1, "full")
    visualise(cfg, task, workspace, diffusion, backbone, ema.ema_model, vis_batch,
              bridge_cfg.train_steps - 1, vis_csv)
    return ema.ema_model


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def load_bridge(cfg, task, workspace, device, path=None, use_ema=True):
    path = path or workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth")
    if not os.path.exists(path):
        print(f"[load] no bridge checkpoint at {path}; truncated methods unavailable")
        return None
    ckpt = load_checkpoint(path, device)
    bridge = task.build_bridge().to(device)
    state = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    bridge.load_state_dict(state)
    bridge.eval()
    print(f"[load] bridge {path} (step={ckpt.get('step')}, "
          f"{'EMA' if use_ema and ckpt.get('ema') else 'raw'} weights)")
    return bridge

"""Stage 1: train the conditional diffusion backbone.

The backbone is a standard conditional diffusion model; the truncation bridge
of stage 2 is trained on top of it with these weights frozen.
"""

import os
import time

import torch
import torchvision

from ..ema import EMA
from ..losses import masked_mse
from ..metrics import batch_metrics
from ..utils import (append_csv, amp_dtype, count_parameters, cosine_lr,
                     infinite_loader, load_checkpoint, prune_checkpoints, set_lr,
                     to01)

CHECKPOINT_PREFIX = "backbone"


def save_backbone(workspace, cfg, model, ema, optimizer, scaler, step):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "config": {"model": dict(cfg.model), "diffusion": dict(cfg.diffusion),
                   "data": dict(cfg.data), "task": cfg.task},
    }
    torch.save(payload, workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth"))
    torch.save(payload, workspace.ckpt(f"{CHECKPOINT_PREFIX}_step_{step:07d}.pth"))
    prune_checkpoints(workspace.checkpoints, CHECKPOINT_PREFIX, cfg.backbone.get("keep_last", 3))
    print(f"[ckpt] backbone step {step} saved", flush=True)


@torch.no_grad()
def visualise(cfg, task, workspace, diffusion, model, batch, step, csv_path):
    from .sampling import run_method

    model.eval()
    started = time.time()
    spec = f"ddim{cfg.backbone.get('vis_steps', 50)}"
    prediction, _, label = run_method(spec, cfg, task, diffusion, model, None, batch)
    elapsed = time.time() - started

    psnr_in, ssim_in, _ = batch_metrics(batch.x0, batch.observation)
    psnr_out, ssim_out, _ = batch_metrics(batch.x0, prediction)
    print(f"[vis] step {step} {label}: PSNR {psnr_in:.2f} -> {psnr_out:.2f}, "
          f"SSIM {ssim_in:.3f} -> {ssim_out:.3f} ({elapsed:.1f}s)", flush=True)
    append_csv(csv_path,
               [step, f"{psnr_in:.3f}", f"{psnr_out:.3f}", f"{ssim_in:.4f}",
                f"{ssim_out:.4f}", f"{elapsed:.1f}"],
               ["step", "psnr_input", "psnr_pred", "ssim_input", "ssim_pred", "seconds"])

    grid = torch.cat(task.visual_rows(batch, prediction), dim=0)
    torchvision.utils.save_image(to01(grid),
                                 os.path.join(workspace.vis, f"backbone_{step:07d}.png"),
                                 nrow=batch.x0.shape[0], padding=2)
    model.train()


def train_backbone(cfg, task, workspace, diffusion, train_loader, val_loader, device):
    train_cfg = cfg.backbone
    model = task.build_backbone().to(device)
    print(f"[model] backbone: {count_parameters(model) / 1e6:.1f}M parameters")

    ema = EMA(model, train_cfg.get("ema_decay", 0.9999))
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    dtype = amp_dtype(cfg.runtime.get("amp", "none"))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.runtime.get("amp", "none") == "fp16"))

    start_step = 0
    resume = train_cfg.get("resume", "auto")
    resume_path = (workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth")
                   if resume == "auto" else (None if resume in ("none", None, "") else resume))
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"[resume] backbone resumed from {resume_path} at step {start_step}")
        if start_step >= train_cfg.train_steps:
            print("[resume] backbone already finished")
            return ema.ema_model
    elif train_cfg.get("init_from"):
        ckpt = load_checkpoint(train_cfg.init_from, device)
        missing, unexpected = model.load_state_dict(ckpt.get("ema", ckpt.get("model")),
                                                    strict=False)
        ema.load_state_dict(model.state_dict())
        print(f"[init] warm start from {train_cfg.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[resume] training backbone from scratch")

    vis_batch = task.prepare_batch(next(iter(val_loader)), train=False)
    vis_size = min(train_cfg.get("vis_num", 4), vis_batch.x0.shape[0])
    vis_batch = _slice_batch(vis_batch, vis_size)

    loss_csv = workspace.log("backbone_loss.csv")
    vis_csv = workspace.log("backbone_val.csv")
    batches = infinite_loader(train_loader)
    accum = max(1, train_cfg.get("accum_steps", 1))
    region_weight = train_cfg.get("region_weight", 1.0)

    model.train()
    started, running = time.time(), 0.0
    print(f"--- stage 1: backbone {start_step} -> {train_cfg.train_steps} ---", flush=True)

    for step in range(start_step, train_cfg.train_steps):
        set_lr(optimizer, cosine_lr(step, train_cfg.lr, train_cfg.get("min_lr", 1e-6),
                                    train_cfg.train_steps, train_cfg.get("warmup_steps", 0)))
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum):
            batch = task.prepare_batch(next(batches), train=True)
            t = torch.randint(0, diffusion.num_timesteps, (batch.x0.shape[0],), device=device)
            noise = torch.randn_like(batch.x0)
            x_t = diffusion.q_sample(batch.x0, t, noise)
            target, weight = diffusion.target_and_weight(
                batch.x0, noise, t, cfg.diffusion.get("snr_gamma", 0.0))
            region = task.loss_weight_map(batch, region_weight)
            if region is not None:
                weight = weight * region

            context = (torch.autocast("cuda", dtype=dtype) if dtype is not None
                       else torch.enable_grad())
            with context:
                prediction = task.model_fn(model, batch)(x_t, t)
                loss = masked_mse(prediction.float(), target, weight) / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach())

        if train_cfg.get("grad_clip", 0) > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        ema.update()

        if step % train_cfg.get("log_every", 100) == 0:
            denom = max(1, train_cfg.get("log_every", 100) if step > start_step else 1)
            speed = (step - start_step + 1) / max(1e-6, time.time() - started)
            lr = optimizer.param_groups[0]["lr"]
            print(f"[backbone] {step}/{train_cfg.train_steps} loss {running / denom:.4f} "
                  f"lr {lr:.2e} {speed:.2f} it/s", flush=True)
            append_csv(loss_csv, [step, f"{running / denom:.6f}", f"{lr:.3e}",
                                  f"{time.time() - started:.0f}"],
                       ["step", "loss", "lr", "elapsed_s"])
            running = 0.0

        if step > start_step and step % train_cfg.get("vis_every", 5000) == 0:
            visualise(cfg, task, workspace, diffusion, ema.ema_model, vis_batch, step, vis_csv)

        if step > start_step and step % train_cfg.get("save_every", 5000) == 0:
            save_backbone(workspace, cfg, model, ema, optimizer, scaler, step)

    save_backbone(workspace, cfg, model, ema, optimizer, scaler, train_cfg.train_steps - 1)
    return ema.ema_model


def _slice_batch(batch, size):
    batch.x0 = batch.x0[:size]
    batch.cond = batch.cond[:size]
    batch.observation = batch.observation[:size]
    if batch.mask is not None:
        batch.mask = batch.mask[:size]
    batch.model_kwargs = {k: (v[:size] if torch.is_tensor(v) else v)
                          for k, v in batch.model_kwargs.items()}
    batch.aux = {k: (v[:size] if torch.is_tensor(v) else v) for k, v in batch.aux.items()}
    return batch


def load_backbone(cfg, task, workspace, device, path=None, use_ema=True):
    path = path or workspace.ckpt(f"{CHECKPOINT_PREFIX}_latest.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}. Run stage 1 first.")
    ckpt = load_checkpoint(path, device)
    model = task.build_backbone().to(device)
    state = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] backbone {path} (step={ckpt.get('step')}, "
          f"{'EMA' if use_ema else 'raw'} weights, frozen)")
    return model

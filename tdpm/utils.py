"""Filesystem, logging and training-loop helpers shared by all tasks."""

import csv
import glob
import math
import os
import random

import numpy as np
import torch

from .config import save_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def amp_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(name)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class Workspace:
    """runs/<exp_name>/{checkpoints,vis,logs,samples}"""

    def __init__(self, cfg, stage="run"):
        self.root = os.path.join(cfg.output_dir, cfg.exp_name)
        self.checkpoints = os.path.join(self.root, "checkpoints")
        self.vis = os.path.join(self.root, "vis")
        self.logs = os.path.join(self.root, "logs")
        self.samples = os.path.join(self.root, "samples")
        for directory in (self.checkpoints, self.vis, self.logs, self.samples):
            os.makedirs(directory, exist_ok=True)
        save_config(cfg, os.path.join(self.root, f"config_{stage}.json"))

    def ckpt(self, name):
        return os.path.join(self.checkpoints, name)

    def log(self, name):
        return os.path.join(self.logs, name)


def append_csv(path, row, header):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def prune_checkpoints(directory, prefix, keep):
    if keep <= 0:
        return
    files = glob.glob(os.path.join(directory, f"{prefix}_step_*.pth"))

    def step_of(path):
        try:
            return int(os.path.basename(path).split("_step_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    for path in sorted(files, key=step_of)[:-keep]:
        try:
            os.remove(path)
        except OSError:
            pass


def cosine_lr(step, base_lr, min_lr, total_steps, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def to01(x):
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    return obj

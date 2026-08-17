"""Evaluation over the held-out validation split.

Every method listed in ``eval.methods`` is run on exactly the same batches with
the same conditioning, so the accelerations reported in the tables are directly
comparable. Numbers produced here are a self-check; the numbers reported in the
paper come from ``tools/score_folder.py`` run on the fixed benchmark sets.
"""

import json
import os
import time

import numpy as np
import torch
import torchvision

from ..metrics import batch_metrics, build_lpips
from ..utils import to01
from .sampling import run_method


@torch.no_grad()
def evaluate(cfg, task, workspace, diffusion, backbone, bridge, val_loader, device):
    eval_cfg = cfg.get("eval", {})
    methods = list(eval_cfg.get("methods", ["ddim50", "tdpm"]))
    num_images = eval_cfg.get("num_images", 8)
    lpips_metric = build_lpips(eval_cfg.get("lpips_net", "alex"), device)

    batches = []
    collected = 0
    for raw in val_loader:
        batch = task.prepare_batch(raw, train=False)
        batches.append(batch)
        collected += batch.x0.shape[0]
        if collected >= num_images:
            break

    results = {}
    psnr, ssim, lpips_value = _aggregate(
        [batch_metrics(b.x0, b.observation, lpips_metric) for b in batches])
    results["observation"] = {"psnr": psnr, "ssim": ssim, "lpips": lpips_value,
                              "seconds": 0.0, "nfe": 0}

    grids = [torch.cat([b.x0 for b in batches])[:num_images],
             torch.cat([b.observation for b in batches])[:num_images]]
    names = ["ground truth", "observation"]

    for spec in methods:
        started = time.time()
        scores, predictions = [], []
        nfe, label = 0, spec
        for batch in batches:
            prediction, nfe, label = run_method(spec, cfg, task, diffusion, backbone,
                                                bridge, batch)
            scores.append(batch_metrics(batch.x0, prediction, lpips_metric))
            predictions.append(prediction)
        elapsed = time.time() - started
        psnr, ssim, lpips_value = _aggregate(scores)
        results[label] = {"psnr": psnr, "ssim": ssim, "lpips": lpips_value,
                          "seconds": elapsed, "nfe": nfe}
        grids.append(torch.cat(predictions)[:num_images])
        names.append(spec)

    _print_table(results)

    with open(workspace.log("eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    out_path = os.path.join(workspace.vis, "eval_grid.png")
    torchvision.utils.save_image(to01(torch.cat(grids, dim=0)), out_path,
                                 nrow=grids[0].shape[0], padding=2)
    print(f"[eval] rows: {' | '.join(names)} -> {out_path}")
    return results


def _aggregate(scores):
    array = np.array(scores, dtype=np.float64)
    return (float(np.nanmean(array[:, 0])), float(np.nanmean(array[:, 1])),
            float(np.nanmean(array[:, 2])))


def _print_table(results):
    print("\n" + "=" * 72)
    print(f"{'method':<32}{'PSNR':>8}{'SSIM':>9}{'LPIPS':>9}{'time(s)':>9}{'NFE':>6}")
    for name, value in results.items():
        print(f"{name:<32}{value['psnr']:>8.2f}{value['ssim']:>9.4f}"
              f"{value['lpips']:>9.4f}{value['seconds']:>9.2f}{value['nfe']:>6d}")
    print("=" * 72)
    print("Validation self-check. Reported numbers come from tools/score_folder.py.\n")

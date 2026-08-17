"""Sampler dispatch.

A method is described by a short string:

    ancestral      full-length ancestral DDPM sampling
    ddim<N>        DDIM with N steps from pure noise
    dpm<N>         DPM-Solver++(2M) with N steps from pure noise
    tdpm           truncated sampling: bridge to t*, then DDIM for the
                   configured number of steps
    tdpm_dpm       truncated sampling with DPM-Solver++ on the remaining steps
"""

import re

import torch

TRUNCATED_METHODS = ("tdpm", "tdpm_dpm")


def parse_method(spec):
    spec = spec.strip().lower()
    if spec in TRUNCATED_METHODS or spec == "ancestral":
        return spec, None
    match = re.fullmatch(r"(ddim|dpm)(\d+)", spec)
    if not match:
        raise ValueError(f"Unrecognised sampling method: {spec}")
    return match.group(1), int(match.group(2))


@torch.no_grad()
def bridge_latent(cfg, task, diffusion, bridge, batch, add_noise=True):
    """Latent at t* produced by the truncation bridge."""
    t_star = cfg.truncation.t_star
    prediction = bridge(task.bridge_input(batch))
    if cfg.bridge.get("target", "mean") == "sample" or not add_noise:
        return prediction
    sigma = diffusion.sqrt_one_minus_alphas_cumprod[t_star]
    return prediction + sigma * torch.randn_like(prediction)


@torch.no_grad()
def run_method(spec, cfg, task, diffusion, backbone, bridge, batch):
    """Returns (prediction, number of network evaluations, display label)."""
    kind, steps = parse_method(spec)
    model_fn = task.model_fn(backbone, batch)
    constraint = task.constraint_fn(batch, diffusion)
    shape = task.sample_shape(batch.x0.shape[0])

    if kind == "ancestral":
        pred = diffusion.ancestral_sample(model_fn, shape, constraint_fn=constraint)
        nfe = diffusion.num_timesteps
        label = f"ancestral({nfe})"
    elif kind == "ddim":
        pred = diffusion.ddim_sample(model_fn, shape, steps=steps,
                                     eta=cfg.get("eval", {}).get("ddim_eta", 0.0),
                                     constraint_fn=constraint)
        nfe, label = steps, f"ddim({steps})"
    elif kind == "dpm":
        pred = diffusion.dpm_solver_sample(model_fn, shape, steps=steps,
                                           constraint_fn=constraint)
        nfe, label = steps, f"dpm-solver++({steps})"
    else:
        if bridge is None:
            raise ValueError(f"Method '{spec}' requires a trained bridge checkpoint")
        t_star = cfg.truncation.t_star
        trunc_steps = cfg.truncation.num_steps
        start = bridge_latent(cfg, task, diffusion, bridge, batch)
        sampler = (diffusion.ddim_sample if kind == "tdpm" else diffusion.dpm_solver_sample)
        pred = sampler(model_fn, shape, steps=trunc_steps, start_img=start,
                       start_t=t_star, constraint_fn=constraint)
        nfe = trunc_steps
        suffix = "" if kind == "tdpm" else "+dpm"
        label = f"tdpm{suffix}(t*={t_star}, {trunc_steps})"

    return task.postprocess(pred, batch), nfe, label

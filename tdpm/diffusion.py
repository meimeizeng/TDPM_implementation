"""Gaussian diffusion process and samplers.

Every sampler accepts ``start_img`` / ``start_t`` so that reverse sampling can
be launched from an intermediate latent produced by the truncation bridge
(Sec. IV of the paper) instead of from pure noise, and an optional
``constraint_fn`` used by inpainting to re-inject the observed pixels at each
reverse step.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SCHEDULES = ("linear", "cosine")
PRED_TYPES = ("eps", "x0", "v")


def make_betas(schedule, num_timesteps, scale_by_steps=True):
    if schedule == "linear":
        scale = (1000.0 / num_timesteps) if scale_by_steps else 1.0
        return torch.linspace(1e-4 * scale, 0.02 * scale, num_timesteps, dtype=torch.float64)
    if schedule == "cosine":
        offset = 0.008
        t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64) / num_timesteps
        f = torch.cos((t + offset) / (1 + offset) * math.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        return torch.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


def rescale_zero_terminal_snr(betas):
    """Rescale betas so that alpha_bar[-1] == 0.

    Without this the training marginal at t = T still carries a small amount of
    signal while sampling starts from pure noise; the mismatch biases the first
    reverse steps.
    """
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ac = alphas_cumprod.sqrt()
    first, last = sqrt_ac[0].clone(), sqrt_ac[-1].clone()
    sqrt_ac = (sqrt_ac - last) * first / (first - last)
    alphas_cumprod = sqrt_ac ** 2
    alphas = torch.cat([alphas_cumprod[:1], alphas_cumprod[1:] / alphas_cumprod[:-1]])
    return 1.0 - alphas


class GaussianDiffusion:
    def __init__(self, num_timesteps, schedule="linear", pred_type="eps",
                 zero_terminal_snr=False, beta_scale_by_steps=True, device="cpu"):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if pred_type not in PRED_TYPES:
            raise ValueError(f"pred_type must be one of {PRED_TYPES}")

        self.num_timesteps = num_timesteps
        self.pred_type = pred_type
        self.device = device

        betas = make_betas(schedule, num_timesteps, beta_scale_by_steps)
        if zero_terminal_snr:
            betas = rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-8)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def register(name, value):
            setattr(self, name, value.float().to(device))

        register("betas", betas)
        register("alphas", alphas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("posterior_variance",
                 betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register("posterior_mean_coef1",
                 betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register("posterior_mean_coef2",
                 (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------ forward
    @staticmethod
    def extract(values, t, shape):
        return values.gather(-1, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return (self.extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
                + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def q_mean(self, x0, t):
        """Mean of q(x_t | x_0), i.e. the deterministic part of the latent."""
        return self.extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0

    def target_and_weight(self, x0, noise, t, snr_gamma=0.0):
        """Regression target for the configured parameterisation and its weight."""
        a = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        snr = (a / s.clamp(min=1e-8)) ** 2

        if self.pred_type == "eps":
            target = noise
            weight = (torch.ones_like(snr) if snr_gamma <= 0
                      else torch.clamp(snr, max=snr_gamma) / snr.clamp(min=1e-8))
        elif self.pred_type == "x0":
            target = x0
            weight = snr if snr_gamma <= 0 else torch.clamp(snr, max=snr_gamma)
        else:
            target = a * noise - s * x0
            weight = (torch.ones_like(snr) if snr_gamma <= 0
                      else torch.clamp(snr, max=snr_gamma) / (snr + 1.0))
        return target, weight

    def to_x0_eps(self, output, x_t, t, clip=True):
        """Convert any network parameterisation into the (x0, eps) pair."""
        a = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        if self.pred_type == "eps":
            eps = output
            x0 = (x_t - s * eps) / a.clamp(min=1e-4)
        elif self.pred_type == "x0":
            x0 = output
            eps = (x_t - a * x0) / s.clamp(min=1e-4)
        else:
            x0 = a * x_t - s * output
            eps = s * x_t + a * output
        if clip:
            x0 = x0.clamp(-1, 1)
            eps = (x_t - a * x0) / s.clamp(min=1e-4)
        return x0, eps

    # ----------------------------------------------------------------- samplers
    def _init_image(self, shape, start_img, start_t, constraint_fn):
        batch = shape[0]
        t_start = (self.num_timesteps - 1) if start_t is None else min(start_t, self.num_timesteps - 1)
        img = torch.randn(shape, device=self.device) if start_img is None else start_img.clone()
        if constraint_fn is not None:
            t = torch.full((batch,), t_start, device=self.device, dtype=torch.long)
            img = constraint_fn(img, t)
        return img, t_start

    @torch.no_grad()
    def ancestral_sample(self, model_fn, shape, start_img=None, start_t=None,
                         constraint_fn=None):
        batch = shape[0]
        img, t_start = self._init_image(shape, start_img, start_t, constraint_fn)
        for i in reversed(range(t_start + 1)):
            t = torch.full((batch,), i, device=self.device, dtype=torch.long)
            x0, _ = self.to_x0_eps(model_fn(img, t), img, t)
            mean = (self.extract(self.posterior_mean_coef1, t, img.shape) * x0
                    + self.extract(self.posterior_mean_coef2, t, img.shape) * img)
            if i > 0:
                log_var = torch.log(
                    self.extract(self.posterior_variance, t, img.shape).clamp(min=1e-20))
                img = mean + torch.exp(0.5 * log_var) * torch.randn_like(img)
                if constraint_fn is not None:
                    t_prev = torch.full((batch,), i - 1, device=self.device, dtype=torch.long)
                    img = constraint_fn(img, t_prev)
            else:
                img = mean
        return img

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, steps=50, eta=0.0, start_img=None,
                    start_t=None, constraint_fn=None):
        batch = shape[0]
        img, t_start = self._init_image(shape, start_img, start_t, constraint_fn)
        steps = max(1, min(steps, t_start + 1))
        sequence = np.linspace(0, t_start, steps).round().astype(int)[::-1].tolist()

        for k, i in enumerate(sequence):
            t = torch.full((batch,), int(i), device=self.device, dtype=torch.long)
            x0, eps = self.to_x0_eps(model_fn(img, t), img, t)
            if k == len(sequence) - 1:
                return x0
            alpha_bar_t = self.extract(self.alphas_cumprod, t, img.shape)
            t_prev = torch.full((batch,), int(sequence[k + 1]), device=self.device,
                                dtype=torch.long)
            alpha_bar_prev = self.extract(self.alphas_cumprod, t_prev, img.shape)
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt() \
                * (1 - alpha_bar_t / alpha_bar_prev).sqrt()
            img = alpha_bar_prev.sqrt() * x0 \
                + (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0:
                img = img + sigma * torch.randn_like(img)
            if constraint_fn is not None:
                img = constraint_fn(img, t_prev)
        return img

    @torch.no_grad()
    def dpm_solver_sample(self, model_fn, shape, steps=20, start_img=None,
                          start_t=None, constraint_fn=None):
        """Multistep DPM-Solver++(2M) in data-prediction form."""
        batch = shape[0]
        img, t_start = self._init_image(shape, start_img, start_t, constraint_fn)
        steps = max(2, min(steps, t_start + 1))
        grid = np.linspace(t_start, 0, steps + 1).round().astype(int).tolist()

        alpha = self.sqrt_alphas_cumprod
        sigma = self.sqrt_one_minus_alphas_cumprod
        lam = torch.log(alpha / sigma.clamp(min=1e-8))

        x0_prev, h_prev, x0_last = None, None, None
        for i in range(steps):
            s, t_next = grid[i], grid[i + 1]
            t = torch.full((batch,), s, device=self.device, dtype=torch.long)
            x0, _ = self.to_x0_eps(model_fn(img, t), img, t)
            x0_last = x0

            h = lam[t_next] - lam[s]
            if x0_prev is None or h_prev is None:
                d = x0
            else:
                r = h_prev / h
                d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0_prev
            img = (sigma[t_next] / sigma[s]) * img - alpha[t_next] * (torch.exp(-h) - 1) * d
            x0_prev, h_prev = x0, h
            if constraint_fn is not None and t_next > 0:
                t_prev = torch.full((batch,), t_next, device=self.device, dtype=torch.long)
                img = constraint_fn(img, t_prev)
        return x0_last if grid[-1] == 0 else img

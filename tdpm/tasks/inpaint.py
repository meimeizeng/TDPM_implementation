"""Box inpainting on FFHQ-256.

The conditioning is the masked image concatenated with the binary mask. During
sampling the observed pixels are re-injected at every reverse step, so the
model only has to synthesise the missing box while staying consistent with the
surrounding context.
"""

import torch

from ..data.ffhq_inpaint import build_inpaint_dataloaders
from .base import BaseTask, Batch


class InpaintTask(BaseTask):
    name = "inpaint"

    @property
    def cond_channels(self):
        return 4

    def build_dataloaders(self):
        return build_inpaint_dataloaders(self.cfg)

    def prepare_batch(self, raw, train=True):
        masked, mask, original = raw
        masked = masked.to(self.device)
        mask = mask.to(self.device)
        original = original.to(self.device)
        cond = torch.cat([masked, mask], dim=1)
        return Batch(x0=original, cond=cond, observation=masked, mask=mask)

    def constraint_fn(self, batch, diffusion):
        if not self.cfg.get("inpaint", {}).get("repaint", True):
            return None
        mask, x0 = batch.mask, batch.x0

        def fn(img, t):
            return mask * img + (1.0 - mask) * diffusion.q_sample(x0, t)

        return fn

    def postprocess(self, prediction, batch):
        return batch.mask * prediction + (1.0 - batch.mask) * batch.x0

"""Non-blind Gaussian deblurring on DIV2K.

The conditioning is the observation concatenated with the adjoint A^T y; the
blur kernel is additionally encoded into the timestep embedding.
"""

import torch

from ..data.div2k_deblur import build_deblur_dataloaders
from ..degradations.deblur import build_condition, degrade, normalize_kernel
from ..modules import KernelConditionedUNet
from .base import BaseTask, Batch


class DeblurTask(BaseTask):
    name = "deblur"

    @property
    def image_size(self):
        return self.cfg.data.patch_size

    @property
    def use_adjoint(self):
        return bool(self.cfg.data.get("condition_on_adjoint", True))

    @property
    def use_kernel_embedding(self):
        return bool(self.cfg.data.get("condition_on_kernel", True))

    @property
    def cond_channels(self):
        return 6 if self.use_adjoint else 3

    def build_dataloaders(self):
        return build_deblur_dataloaders(self.cfg)

    def build_backbone(self):
        model = self.cfg.model
        return KernelConditionedUNet(
            image_size=self.image_size,
            in_channels=3 + self.cond_channels,
            out_channels=3,
            base_channels=model.base_channels,
            channel_mult=tuple(model.channel_mult),
            num_res_blocks=model.num_res_blocks,
            attn_resolutions=tuple(model.attn_resolutions),
            dropout=model.get("dropout", 0.0),
            use_kernel_embedding=self.use_kernel_embedding,
        )

    @torch.no_grad()
    def _observe(self, clean, kernel):
        return degrade(clean, kernel, self.cfg.data.noise_std)

    def prepare_batch(self, raw, train=True):
        clean, kernel = raw[0].to(self.device), raw[1].to(self.device)
        y = self._observe(clean, kernel)
        kernel_net = normalize_kernel(kernel, self.cfg.data.get("normalize_kernel", False))
        cond = build_condition(y, kernel_net, self.use_adjoint)
        return Batch(x0=clean, cond=cond, observation=y,
                     model_kwargs={"kernel": kernel_net},
                     aux={"kernel": kernel})

    def batch_from_observation(self, y, kernel, clean=None):
        """Build a batch from a pre-computed observation (benchmark evaluation)."""
        kernel_net = normalize_kernel(kernel, self.cfg.data.get("normalize_kernel", False))
        cond = build_condition(y, kernel_net, self.use_adjoint)
        return Batch(x0=clean if clean is not None else y, cond=cond, observation=y,
                     model_kwargs={"kernel": kernel_net}, aux={"kernel": kernel})

"""4x super-resolution conditioned on the bicubic-upsampled observation."""

import torch

from ..data.sr_dataset import build_sr_dataloaders
from .base import BaseTask, Batch


class SuperResolutionTask(BaseTask):
    name = "sr"

    @property
    def cond_channels(self):
        return 3

    def build_dataloaders(self):
        return build_sr_dataloaders(self.cfg)

    def prepare_batch(self, raw, train=True):
        hr, lr_up = raw
        hr = hr.to(self.device)
        lr_up = lr_up.to(self.device)
        return Batch(x0=hr, cond=lr_up, observation=lr_up)

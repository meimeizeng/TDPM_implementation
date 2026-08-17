"""Task interface.

A task owns everything that differs between restoration problems: the data, the
conditioning tensor, the backbone signature and any data-consistency step used
during sampling. The training and evaluation engines are written against this
interface and are otherwise task agnostic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from ..bridge import BridgeNet
from ..modules import ConditionalUNet


@dataclass
class Batch:
    """One prepared minibatch."""

    x0: torch.Tensor                                   # clean target, [-1,1]
    cond: torch.Tensor                                 # conditioning channels
    observation: torch.Tensor                          # degraded image, for display
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    mask: Optional[torch.Tensor] = None                # 1 = unknown region
    aux: Dict[str, Any] = field(default_factory=dict)


class BaseTask:
    name = "base"

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------ shapes
    @property
    def image_size(self):
        return self.cfg.data.image_size

    @property
    def cond_channels(self):
        raise NotImplementedError

    def sample_shape(self, batch_size):
        return (batch_size, 3, self.image_size, self.image_size)

    # -------------------------------------------------------------------- data
    def build_dataloaders(self):
        raise NotImplementedError

    def prepare_batch(self, raw, train=True):
        raise NotImplementedError

    # ------------------------------------------------------------------ models
    def build_backbone(self):
        model = self.cfg.model
        return ConditionalUNet(
            image_size=self.image_size,
            in_channels=3 + self.cond_channels,
            out_channels=3,
            base_channels=model.base_channels,
            channel_mult=tuple(model.channel_mult),
            num_res_blocks=model.num_res_blocks,
            attn_resolutions=tuple(model.attn_resolutions),
            dropout=model.get("dropout", 0.0),
        )

    def build_bridge(self):
        bridge = self.cfg.bridge
        return BridgeNet(
            param_type=bridge.param_type,
            cond_channels=self.cond_channels,
            image_size=self.image_size,
            out_channels=3,
            rank=bridge.get("rank", 256),
            work_size=bridge.get("work_size", None),
            base_channels=bridge.get("base_channels", 48),
        )

    # ------------------------------------------------------------------ wiring
    def model_fn(self, model, batch):
        """Closure with the signature expected by the samplers."""
        def fn(x, t):
            return model(x, batch.cond, t.float(), **batch.model_kwargs)
        return fn

    def bridge_input(self, batch):
        return batch.cond

    def constraint_fn(self, batch, diffusion):
        """Optional per-step data consistency (used by inpainting)."""
        return None

    def postprocess(self, prediction, batch):
        return prediction

    def loss_weight_map(self, batch, weight):
        from ..losses import region_weight_map
        return region_weight_map(batch.mask, weight)

    def visual_rows(self, batch, prediction):
        return [batch.x0, batch.observation, prediction]

"""Exponential moving average of model weights."""

import copy

import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            if param.dtype.is_floating_point:
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
            else:
                ema_param.data.copy_(param.data)
        for buf, ema_buf in zip(self.model.buffers(), self.ema_model.buffers()):
            ema_buf.copy_(buf)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

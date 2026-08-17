"""Loss terms used by the two training stages.

The bridge objective is

    L = L_rec + omega * L_con

with L_rec a smooth-l1 (Huber) term on the predicted latent and L_con an l2
term on the image the frozen backbone recovers from that latent. Both norms
are configurable; the defaults reproduce the setting used in the paper.
"""

import torch
import torch.nn.functional as F

NORMS = ("l1", "l2", "smooth_l1")


def elementwise_loss(pred, target, norm="l2", beta=1.0):
    if norm == "l1":
        return (pred - target).abs()
    if norm == "l2":
        return (pred - target) ** 2
    if norm == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    raise ValueError(f"norm must be one of {NORMS}")


def weighted_loss(pred, target, norm="l2", weight=None, beta=1.0):
    """Mean loss, optionally weighted by a per-pixel map."""
    values = elementwise_loss(pred, target, norm, beta)
    if weight is None:
        return values.mean()
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def region_weight_map(mask, weight):
    """Weight ``weight`` inside the mask and 1 outside.

    Normalising by the total weight rather than by the number of pixels keeps
    the term from being diluted when the masked area is small.
    """
    if mask is None or weight == 1.0:
        return None
    return 1.0 + (weight - 1.0) * mask


def combine_weights(*maps):
    out = None
    for m in maps:
        if m is None:
            continue
        out = m if out is None else out * m
    return out


def masked_mse(pred, target, weight=None):
    if weight is None:
        return F.mse_loss(pred, target)
    weight = weight.expand_as(pred)
    return (((pred - target) ** 2) * weight).sum() / weight.sum().clamp(min=1.0)


@torch.no_grad()
def detach_scalar(value):
    return float(value.detach()) if torch.is_tensor(value) else float(value)

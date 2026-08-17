#!/usr/bin/env python
"""Convert checkpoints from the original single-file training scripts.

The refactor renamed modules; the tensors themselves are unchanged, so old
weights can be reused after a key remapping:

    python tools/convert_legacy_checkpoint.py --kind backbone \
        --input checkpoints_inpaint/inpaint_unet_256_step_475000.pth \
        --output runs/inpaint_ffhq256/checkpoints/backbone_latest.pth

Use ``--config`` to verify that the converted state dict actually loads into
the model built from a config file.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP_LEVEL = {
    "emb": "time_embedding",
    "conv1": "input_conv",
    "mid": "middle",
    "norm": "out_norm",
    "final": "out_conv",
    "kernel_enc": "kernel_encoder",
}

INNER = [
    (".gn1.", ".norm1."),
    (".gn2.", ".norm2."),
    (".linear1.", ".emb_proj."),
    (".linear2.", ".skip."),
    (".att.", ".attention."),
    (".gn.", ".norm."),
    (".group_norm.", ".norm."),
]

BRIDGE = [
    ("linear.b", "linear.bias"),
    ("correction.e1.", "correction.enc1."),
    ("correction.e2.", "correction.enc2."),
    ("correction.e3.", "correction.enc3."),
    ("correction.mid.", "correction.middle."),
    ("correction.d3.", "correction.dec3."),
    ("correction.d2.", "correction.dec2."),
    ("correction.d1.", "correction.dec1."),
    ("correction.main.", "correction.enc1."),
]


def convert_backbone_key(key):
    head, _, tail = key.partition(".")
    if head == "emb" and tail.startswith("x"):
        return "time_embedding.freq"
    head = TOP_LEVEL.get(head, head)
    new_key = f"{head}.{tail}" if tail else head
    for old, new in INNER:
        new_key = new_key.replace(old, new)
    return new_key


def convert_bridge_key(key):
    for old, new in BRIDGE:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def convert_state_dict(state_dict, kind):
    convert = convert_backbone_key if kind == "backbone" else convert_bridge_key
    return {convert(k.replace("module.", "")): v for k, v in state_dict.items()}


def pick(checkpoint, *names):
    for name in names:
        if name in checkpoint and checkpoint[name] is not None:
            return checkpoint[name]
    return None


def main():
    parser = argparse.ArgumentParser("convert legacy checkpoints")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", default="backbone", choices=["backbone", "bridge"])
    parser.add_argument("--config", default=None,
                        help="optional: verify the result loads into this model")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    raw_model = pick(checkpoint, "model", "model_state_dict")
    raw_ema = pick(checkpoint, "ema", "ema_state_dict")
    if raw_model is None and raw_ema is None:
        raw_model = checkpoint

    payload = {
        "step": checkpoint.get("step", 0),
        "model": convert_state_dict(raw_model, args.kind) if raw_model else None,
        "ema": convert_state_dict(raw_ema, args.kind) if raw_ema else None,
        "optimizer": {},
        "converted_from": os.path.abspath(args.input),
    }
    if payload["model"] is None:
        payload["model"] = payload["ema"]

    print(f"[convert] {len(payload['model'])} tensors remapped "
          f"({'with' if payload['ema'] else 'without'} EMA weights)")

    if args.config:
        from tdpm.config import load_config
        from tdpm.tasks import build_task

        cfg = load_config(args.config)
        task = build_task(cfg, torch.device("cpu"))
        model = task.build_backbone() if args.kind == "backbone" else task.build_bridge()
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"[verify] missing={len(missing)} unexpected={len(unexpected)}")
        for name in list(missing)[:10]:
            print(f"  missing: {name}")
        for name in list(unexpected)[:10]:
            print(f"  unexpected: {name}")
        if missing or unexpected:
            print("[verify] the architecture in the config does not match the "
                  "checkpoint; check base_channels, channel_mult and conditioning flags")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    print(f"[convert] written to {args.output}")


if __name__ == "__main__":
    main()

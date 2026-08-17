#!/usr/bin/env bash
# Bridge parameterisation ablation: Ay+b, f(y), Ay+b+f(y), and omega = 0.
set -e
CONFIG=configs/deblur_div2k.yaml

for VARIANT in linear nonlinear hybrid; do
  python tools/train.py --config $CONFIG --stage bridge \
      --set bridge.param_type=$VARIANT exp_name=deblur_div2k_$VARIANT \
            bridge.backbone_ckpt=runs/deblur_div2k/checkpoints/backbone_latest.pth
done

python tools/train.py --config $CONFIG --stage bridge \
    --set bridge.omega=0.0 exp_name=deblur_div2k_omega0 \
          bridge.backbone_ckpt=runs/deblur_div2k/checkpoints/backbone_latest.pth

#!/usr/bin/env bash
# Full inpainting pipeline.
set -e
CONFIG=configs/inpaint_ffhq256.yaml

python tools/train.py --config $CONFIG --stage backbone
python tools/train.py --config $CONFIG --stage bridge

for METHOD in ddim100 ddim50 dpm20 tdpm; do
  python tools/evaluate.py --config $CONFIG --methods $METHOD \
      --num 200 --export_dir results/inpaint_$METHOD
  python tools/score_folder.py --gt results/inpaint_$METHOD/gt \
      --pred results/inpaint_$METHOD/pred --name inpaint_$METHOD
done

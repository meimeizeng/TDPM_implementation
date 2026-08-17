#!/usr/bin/env bash
# Full super-resolution pipeline.
set -e
CONFIG=configs/sr_imagenet.yaml

python tools/train.py --config $CONFIG --stage backbone
python tools/train.py --config $CONFIG --stage bridge

for METHOD in ddim50 dpm20 tdpm; do
  python tools/evaluate.py --config $CONFIG --methods $METHOD \
      --num 200 --export_dir results/sr_$METHOD
  python tools/score_folder.py --gt results/sr_$METHOD/gt \
      --pred results/sr_$METHOD/pred --name sr_$METHOD
done

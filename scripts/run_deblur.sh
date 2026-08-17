#!/usr/bin/env bash
# Full deblurring pipeline: two training stages, benchmark, scoring.
set -e
CONFIG=configs/deblur_div2k.yaml
BENCH=benchmarks/div2k_gauss

python tools/train.py --config $CONFIG --stage backbone
python tools/train.py --config $CONFIG --stage bridge

python tools/make_deblur_benchmark.py \
    --source data/DIV2K/DIV2K_valid_HR --out $BENCH --num 100
python tools/check_deblur_operator.py --bench_dir $BENCH

for METHOD in ddim50 dpm20 tdpm; do
  python tools/evaluate.py --config $CONFIG --methods $METHOD \
      --bench_dir $BENCH --export_dir results/deblur_$METHOD
  python tools/score_folder.py --gt $BENCH/gt \
      --pred results/deblur_$METHOD/pred --name deblur_$METHOD
done

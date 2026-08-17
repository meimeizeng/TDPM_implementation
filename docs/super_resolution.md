# 4x super-resolution

SR3-style conditioning: the observation is bicubically downsampled and
upsampled back to the target resolution, then concatenated to the noisy image.

## Data

Any folder of images works; point `data.root` at it. Images are resized and
centre-cropped to `data.image_size`, and the low-resolution view is produced on
the fly, so no paired dataset needs to be prepared.

```
data/imagenet/**/*.JPEG
```

## Training

```bash
python tools/train.py --config configs/sr_imagenet.yaml --stage backbone
python tools/train.py --config configs/sr_imagenet.yaml --stage bridge
```

## Choosing t*

The upsampled observation already fixes almost all of the low-frequency content
of the target, so the admissible region reaches down to very small truncation
steps: the default is `t_star = 35` out of T = 2000 with 5 reverse steps. This
is the clearest illustration of the information-gain argument in the paper -
the more the observation reveals, the earlier the reverse process can be
entered.

Sweep it to reproduce the corresponding figure:

```bash
for t in 20 35 60 100 200; do
  python tools/train.py --config configs/sr_imagenet.yaml --stage bridge \
      --set truncation.t_star=$t exp_name=sr_t$t
done
```

## Evaluation

```bash
python tools/evaluate.py --config configs/sr_imagenet.yaml \
    --methods ddim50 dpm20 tdpm --num 16

python tools/evaluate.py --config configs/sr_imagenet.yaml \
    --methods tdpm --num 200 --export_dir results/sr_tdpm
python tools/score_folder.py --gt results/sr_tdpm/gt \
    --pred results/sr_tdpm/pred --name sr_tdpm
```

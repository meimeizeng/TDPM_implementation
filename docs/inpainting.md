# Box inpainting on FFHQ-256

Random square holes between 32 and 128 pixels on 256x256 faces.

## Data

```bash
huggingface-cli download merkol/ffhq-256 --repo-type dataset \
    --local-dir data/ffhq-256
```

The loader reads `data/ffhq-256/data/*.parquet` directly, so training runs
without network access. If no shards are found it falls back to the Hub.

## Conditioning and data consistency

The network input is the noisy image, the masked image and the binary mask
(3 + 3 + 1 channels). During sampling the observed pixels are overwritten at
every reverse step with a correctly noised version of the observation, so the
model only synthesises the missing box while staying consistent with its
surroundings. Disable with `--set inpaint.repaint=false`.

Both stages weight the loss inside the hole (`region_weight`). The weighting is
normalised by the total weight rather than by pixel count, so the term is not
diluted when the hole is small.

## Training

```bash
python tools/train.py --config configs/inpaint_ffhq256.yaml --stage backbone
python tools/train.py --config configs/inpaint_ffhq256.yaml --stage bridge
```

## Choosing t*

Inpainting is the task with the least informative observation: the missing
content has to be generated, not recovered. Under the default schedule
(T = 2000, linear):

| t | alpha_bar | comment |
| --- | --- | --- |
| 100 | 0.941 | practically a clean image; the bridge would do all the work |
| 400 | 0.432 | signal and noise comparable |
| 600 | 0.156 | default; enough room left for the prior to generate |
| 800 | 0.038 | close to pure noise; little is saved |

The bridge for this task applies its low-rank map at 64x64
(`bridge.work_size`), and its correction network uses three downsampling stages
so that the receptive field covers the largest 128 px hole.

## Evaluation

```bash
python tools/evaluate.py --config configs/inpaint_ffhq256.yaml \
    --methods ddim100 ddim50 dpm20 tdpm --num 16

python tools/evaluate.py --config configs/inpaint_ffhq256.yaml \
    --methods tdpm --num 200 --export_dir results/inpaint_tdpm
python tools/score_folder.py --gt results/inpaint_tdpm/gt \
    --pred results/inpaint_tdpm/pred --name inpaint_tdpm
```

Validation masks are deterministic (seeded per index), so repeated runs and
different methods see exactly the same holes.

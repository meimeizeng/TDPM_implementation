# Truncated Diffusion Probabilistic Models for Image Restoration

Reference implementation of **Truncated Diffusion Probabilistic Models (TDPM)**
for image restoration under linear degradations in pixel space.

Reverse diffusion is expensive mainly because it starts from pure noise and has
to travel the whole trajectory. When a degraded observation `y` is available,
most of that trajectory is redundant: `y` already determines the low-frequency
content of the target. TDPM learns a **quasi-linear map from the observation to
an intermediate latent state** at a truncation step `t*`, and runs the reverse
process only over `[0, t*]`. The truncation step is chosen from an admissible
region derived from the initialisation-mismatch bound, so the shortened
trajectory is provably non-inferior to the full one.

Three restoration tasks are released here, each with the same two-stage recipe:

| Task | Dataset | Degradation | Config |
| --- | --- | --- | --- |
| Super-resolution | ImageNet / any image folder | 4x bicubic downsampling | `configs/sr_imagenet.yaml` |
| Inpainting | FFHQ-256 | random box, 32-128 px | `configs/inpaint_ffhq256.yaml` |
| Deblurring | DIV2K | Gaussian blur, sigma in [2.5, 10], noise 0.05 | `configs/deblur_div2k.yaml` |

---

## Method in two stages

**Stage 1 - backbone.** A standard conditional diffusion model
`eps_theta(x_t, y, t)` is trained on the task. Nothing about truncation is
involved here, so an existing checkpoint can be reused.

**Stage 2 - truncation bridge.** The backbone is *frozen* and a small network
is trained to map the observation to the latent at `t*`:

```
L = L_rec + omega * L_con
L_rec = || g(y) - E[x_{t*} | x_0] ||_smooth-l1
L_con = || D(g(y) + sigma_{t*} z) - x_0 ||_2
```

`D` is the one-step clean-image estimate produced by the frozen backbone. The
bridge regresses the **mean** of `q(x_{t*} | x_0)`; the stochastic component is
added back at sampling time, so the network never has to fit unpredictable
noise and sample diversity is preserved.

The bridge has three parameterisations, matching the ablation in the paper:

| `bridge.param_type` | Form | Notes |
| --- | --- | --- |
| `linear` | `A y + b` | `A = U V^T`, trained by gradient descent |
| `nonlinear` | `f(y)` | small UNet only |
| `hybrid` | `A y + b + f(y)` | default; `A y + b` is fitted first, then frozen |

At high resolution a dense `A` is not storable (a 256x256x3 to 256x256x3 map has
about 1.5e10 entries), so `A` is kept in the factored form `A = U V^T` and
applied at a lower working resolution (`bridge.work_size`); a rank-256 map at
64x64 needs roughly 7e6 parameters.

---

## Installation

```bash
git clone https://github.com/<user>/tdpm-restoration.git
cd tdpm-restoration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is intentionally left out of `requirements.txt`: install the build that
matches your CUDA version first, following https://pytorch.org.

Tested with Python 3.10 and PyTorch 2.1 on a single 24 GB GPU.

---

## Data

```
data/
  DIV2K/
    DIV2K_train_HR/*.png
    DIV2K_valid_HR/*.png
  ffhq-256/
    data/*.parquet            # huggingface-cli download merkol/ffhq-256 --local-dir data/ffhq-256
  imagenet/                   # any folder of images works for super-resolution
```

Details for each task are in [`docs/`](docs/).

---

## Training

Both stages share one entry point. Every configuration field can be overridden
on the command line.

```bash
# Stage 1: the conditional diffusion backbone
python tools/train.py --config configs/deblur_div2k.yaml --stage backbone

# Stage 2: the truncation bridge (backbone frozen)
python tools/train.py --config configs/deblur_div2k.yaml --stage bridge

# Both, followed by a validation self-check
python tools/train.py --config configs/deblur_div2k.yaml --stage all

# Override anything
python tools/train.py --config configs/deblur_div2k.yaml --stage bridge \
    --set truncation.t_star=300 bridge.param_type=linear bridge.omega=0.0
```

Stage 2 writes `bridge_*.pth` next to, but never over, `backbone_*.pth`, so the
two stages can be iterated independently. Both stages resume automatically from
`runs/<exp_name>/checkpoints/*_latest.pth`.

Output layout:

```
runs/<exp_name>/
  checkpoints/   backbone_latest.pth, bridge_latest.pth, periodic snapshots
  vis/           sample grids written during training
  logs/          loss and validation curves as CSV, eval_results.json
  config_<stage>.json
```

---

## Evaluation

Compare samplers under identical conditioning:

```bash
python tools/evaluate.py --config configs/deblur_div2k.yaml \
    --methods ddim50 dpm20 tdpm --num 16
```

Method strings: `ancestral`, `ddim<N>`, `dpm<N>`, `tdpm`, `tdpm_dpm`. The last
two start from the bridge output at `t*` and use `truncation.num_steps`
reverse steps.

The numbers reported in the paper come from exported images scored by one
shared script, so that every method goes through the same PSNR/SSIM/LPIPS
protocol:

```bash
python tools/evaluate.py --config configs/inpaint_ffhq256.yaml \
    --methods tdpm --num 200 --export_dir results/inpaint_tdpm

python tools/score_folder.py --gt results/inpaint_tdpm/gt \
    --pred results/inpaint_tdpm/pred --name tdpm --note "t*=600, 10 steps"
```

`score_folder.py` computes PSNR and SSIM on uint8 RGB over the full image (no
border cropping, no Y-channel conversion) and LPIPS with a single backbone, and
appends one row per run to `results/leaderboard.csv`.

For deblurring, build the fixed benchmark once and run every method on it:

```bash
python tools/make_deblur_benchmark.py --source data/DIV2K/DIV2K_valid_HR \
    --out benchmarks/div2k_gauss --num 100
python tools/check_deblur_operator.py --bench_dir benchmarks/div2k_gauss
python tools/evaluate.py --config configs/deblur_div2k.yaml --methods tdpm \
    --bench_dir benchmarks/div2k_gauss --export_dir results/deblur_tdpm
```

---

## Choosing the truncation step

`truncation.t_star` is the one setting that matters most. The paper derives two
nested ranges: an **admissible region** where truncated sampling is
non-inferior to full sampling, and a **preferred interval** inside it that also
maximises the speedup. In practice:

- too small - the bridge is asked to do the restoration alone and the diffusion
  prior contributes almost nothing;
- too large - little of the trajectory is skipped, so the acceleration vanishes;
- the more informative the observation, the smaller `t*` can be. This is why
  4x super-resolution tolerates a very small `t*` while box inpainting, where
  the missing content has to be synthesised, needs a much larger one.

A sweep is one command per value:

```bash
for t in 200 300 400 500; do
  python tools/train.py --config configs/deblur_div2k.yaml --stage bridge \
      --set truncation.t_star=$t exp_name=deblur_t$t
done
```

---

## Sanity checks

```bash
# shapes and wiring for all three tasks, no data required, runs on CPU
python tools/smoke_test.py

# the deblurring forward operator against scipy and against a benchmark
python tools/check_deblur_operator.py --bench_dir benchmarks/div2k_gauss
```

## Reusing checkpoints from the original scripts

Weights trained with the earlier single-file scripts are still valid; only
module names changed. Remap them once:

```bash
python tools/convert_legacy_checkpoint.py --kind backbone \
    --input old/inpaint_unet_256_step_475000.pth \
    --output runs/inpaint_ffhq256/checkpoints/backbone_latest.pth \
    --config configs/inpaint_ffhq256.yaml
```

Passing `--config` verifies that the converted state dict loads cleanly and
reports any key that does not match.

## Repository layout

```
tdpm/
  config.py            YAML configs with dotted-key overrides
  modules.py           shared UNet blocks and the conditional UNet
  diffusion.py         schedules, parameterisations, DDIM / DPM-Solver++ / ancestral samplers
  bridge.py            truncation bridge: A y + b, f(y), A y + b + f(y)
  losses.py            reconstruction and consistency terms
  metrics.py           PSNR / SSIM / LPIPS
  data/                one dataset per task
  degradations/        the deblurring forward operator and its adjoint
  tasks/               per-task conditioning, models and data-consistency rules
  engine/              stage 1, stage 2, samplers, evaluation
tools/
  train.py                     stage 1 / stage 2 / both
  evaluate.py                  sampler comparison and image export
  score_folder.py              the single scoring protocol for all methods
  make_deblur_benchmark.py     fixed DIV2K deblurring benchmark
  check_deblur_operator.py     forward-operator diagnostics
  convert_legacy_checkpoint.py remap weights from the original scripts
  smoke_test.py                shape and wiring check, no data needed
configs/               one config per task, plus ablations
scripts/               end-to-end shell pipelines
docs/                  per-task instructions
```

Adding a task means implementing one subclass of `tdpm.tasks.base.BaseTask`
(conditioning, data, optional data-consistency step) and writing a config; the
training engines are task agnostic.

---

## Scope

The method as implemented here assumes a **linear degradation operator in pixel
space**, which covers the three tasks above. Latent-space diffusion and
non-linear degradations are outside the setting analysed in the paper.

---

## Citation

```bibtex
@article{tdpm,
  title   = {Truncated Diffusion Probabilistic Models for Image Restoration},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT. See [LICENSE](LICENSE).

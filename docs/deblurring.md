# Deblurring on DIV2K

Non-blind Gaussian deblurring at 256x256, `sigma` drawn from [2.5, 10], additive
noise with standard deviation 0.05 on the [0,1] scale.

## Data

```
data/DIV2K/DIV2K_train_HR/*.png
data/DIV2K/DIV2K_valid_HR/*.png
```

Both splits are available from the
[DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Patches are cropped on
the fly; `data.repeat` controls how many crops are drawn per image and epoch.

## Forward operator

`tdpm/degradations/deblur.py` defines the operator and must not be changed
without regenerating the benchmark:

1. the kernel comes from `scipy.ndimage.gaussian_filter` applied to a delta
   image, not from an analytic formula;
2. kernel support is 61x61 and the kernel is **not** renormalised - scipy
   truncates at 4 sigma without compensating, so `k.sum()` is slightly below 1
   and that gain is part of the operator;
3. the convolution uses circular boundaries;
4. blur first, then add noise;
5. observations are **not** clipped.

Two equivalent implementations of the circular convolution are available. The
FFT path is faster; the circular-padding path avoids cuFFT, which fails on some
torch/CUDA builds. Selection is automatic and can be forced:

```bash
export DEBLUR_CONV_IMPL=conv     # fft | conv | auto
```

Verify everything before training:

```bash
python tools/check_deblur_operator.py --bench_dir benchmarks/div2k_gauss
```

The residual `mean |A(x) - y|` on the benchmark should land near 0.0798, the
expected absolute value of the noise on the [-1,1] scale.

## Conditioning

The network sees the observation concatenated with the adjoint `A^T y`
(`data.condition_on_adjoint`), and the kernel encoded into the timestep
embedding (`data.condition_on_kernel`), which makes the setting non-blind.

## Training

```bash
python tools/train.py --config configs/deblur_div2k.yaml --stage backbone
python tools/train.py --config configs/deblur_div2k.yaml --stage bridge
```

The backbone predicts `v` rather than `eps`. On restoration tasks the `eps`
parameterisation puts almost no weight on high `t`, which is exactly where
sampling from pure noise starts; `v` weights the objective uniformly in
`x_0` space. `zero_terminal_snr` additionally rescales the schedule so that the
training marginal at `t = T` carries no residual signal.

## Evaluation

```bash
python tools/make_deblur_benchmark.py --source data/DIV2K/DIV2K_valid_HR \
    --out benchmarks/div2k_gauss --num 100

python tools/evaluate.py --config configs/deblur_div2k.yaml --methods tdpm \
    --bench_dir benchmarks/div2k_gauss --export_dir results/deblur_tdpm

python tools/score_folder.py --gt benchmarks/div2k_gauss/gt \
    --pred results/deblur_tdpm/pred --name deblur_tdpm
```

The benchmark stores observations twice: `lq/<stem>.npy` is the true unclipped
input and is what the evaluator reads; `lq/<stem>.png` is clipped and quantised
and exists only for viewing.

Two inference-time corrections are available and must be applied to every
method or to none:

- `--set data.normalize_kernel=true` rescales the kernel to sum to one in the
  network-facing representation. It is needed when benchmark kernels use a
  different scale from the training ones, otherwise both the kernel embedding
  and `A^T y` receive out-of-distribution input, which shows up as a uniform
  colour cast. It does not modify the observations.
- `--dc_fix` projects the prediction onto `mean(x) = mean(y) / k.sum()`. The DC
  gain of a blur kernel is exactly `k.sum()`, so this is a hard data-consistency
  constraint. It fixes brightness drift; it does not restore detail.

Record whichever settings were used in `--note` when scoring.

# Why the kernelgen reference is slower than the eager benchmark

An investigation of one run (2026-10-15, EfficientViT-B1 from timm, batch 4 × 3 × 1088 × 1920, fp32,
NVIDIA GeForce RTX 5090, driver 580.95.05) whose `benchmark.json` reported an eager median of
21.391 ms while the kernelgen report's reference median was 41.582 ms. The two numbers measure
different programs. Nothing in the check's timing loop is wrong, and no default was changed.

## Finding

The kernelgen reference is the Core ATen graph that `yasp-toolkit kernelgen export` writes
(`torch.export` followed by `run_decompositions()`), executed by Gaia as a plain FX graph with the
lifted parameters as inputs. The eager benchmark runs the user's `nn.Module` as written. For this
model the decomposition roughly doubles the GPU work, with bit-identical outputs:

| Cause | Mechanism | Cost (ms, median) |
|---|---|---|
| Hardswish decomposition | Each of the 30 `hardswish_` ops becomes `add`, `clamp`, `clamp`, `mul`, `div`: five memory-bound passes over a large activation instead of one fused kernel. | +16.6 |
| `Dropout(p=0)` as `clone` | timm's `ConvNormAct` has a dropout before every convolution. With torch 2.9.1 the decomposition turns the 59 inactive dropouts into `aten.clone`, a device-to-device copy each. Eager runs nothing. | +4.3 |
| Remainder (7 `contiguous()` clones, 28 `_assert_tensor_metadata`, FX dispatch) | Not separated further. | +0.4 |
| Runtime: torch 2.14.1+cu130/cuDNN 9.24 (check image) versus torch 2.9.1+cu128/cuDNN 9.10 (Gaia worker image) | Both eager and the graph are about 6 % slower in the newer runtime. | +1.3 on both sides |
| Timing protocol: L2 flush, per-call synchronize, interleaving with the candidate | Working set far exceeds L2 at this resolution. | ≤ +0.2 |
| Weights, input values, grad mode | Weights are bit-identical (seeded `__init__`); synthesized inputs and `inference_mode` change nothing. | 0 |

Accounting: eager in the Gaia image 20.13 ms + 16.6 + 4.3 + 0.4 ≈ 41.4 ms, which is what the Gaia
graph measures in the same process (41.47 ms) and what Gaia reported (41.58 ms, replayed here as
41.60 ms). Eager in the check image is 21.43 ms, the 21.39 ms `benchmark.json` reported. The Hardswish
and clone effects were measured as a full 2 × 2 factorial in one process and are additive; the runtime
and protocol effects were measured separately, not jointly with the graph factors.

## Evidence

All timings are medians in ms over 100 to 200 timed calls after 20 warm-up calls, CUDA events around
each call, two or three repeats with the subject order reversed between repeats, no other process on
the GPU, SM clock 2865 to 2895 MHz under load. Outputs of every variant were compared with the eager
output: maximum absolute difference 0.0 in every case.

Gaia worker image (`docker.yasp.ai/yasp-eval/nvidia:1.11.0`, torch 2.9.1+cu128):

| Subject | Kronos timing (block, sync at end) | Gaia timing (L2 flush, sync each call, interleaved) |
|---|---|---|
| eager `Model()` | 20.10 / 20.13 / 20.13 | 20.21 / 20.22 |
| `model.pt2` graph as Gaia runs it | 41.44 / 41.47 / 41.47 | 41.51 / 41.51 |
| `model.pt2`, clones from dropout removed | 37.14 / 37.17 | |
| `model.pt2`, Hardswish re-fused | 24.81 / 24.85 | |
| `model.pt2`, both | 20.55 / 20.58 | |
| fresh `torch.export` in the container, no decomposition | 20.12 / 20.15 | |
| fresh export + `run_decompositions()` (same op histogram as `model.pt2`) | 41.44 / 41.45 | |
| fresh export, decompositions minus Hardswish | 24.82 / 24.83 | |
| fresh export, minus Hardswish, clones removed | 20.56 / 20.58 | |

Check image (`kronos-edge:rtx5090-cuda130`, torch 2.14.1+cu130): eager 21.42 / 21.45 / 21.46; graph
42.71 / 42.73; clones removed 38.52; Hardswish re-fused 26.20; both 21.89; fresh export without
decomposition 21.46. In this torch, `run_decompositions()` no longer emits the dropout clones (38.53).

Faithful replay, `yasp-eval kernel` on the original `model.pt2` and `kernel.py` in the worker image
with no network: reference median 41.601 ms (30 samples, std 0.032), candidate 10.295 ms, speedup
4.043, converged. The original report: 41.582, 10.304, 4.036.

Kernel-level profile (torch profiler, 5 iterations): eager 19.95 GPU ms per call in 236 kernels; the
graph 40.94 GPU ms in 483 kernels. Both are GPU-bound (GPU time equals wall time). Convolution,
batch-norm (`cudnn::bn_fw_inf` 4.70 ms versus `batch_norm_transform_input` 4.68 ms), attention and
concatenation kernels take the same time on both sides. The Hardswish kernels take 3.17 ms in eager
and 19.83 ms decomposed (`clamp` 7.26, `mul` 5.52, `div` 3.62, `add` 3.43); the clones add 4.23 ms of
`Memcpy DtoD`.

Other factors, measured on both subjects in the worker image: synchronize after every call +0.15 ms;
256 MiB L2 flush before every call +0.05 ms; call-by-call interleaving +0.02 ms; synthesized weights
and inputs with the same statistics 0; `inference_mode` instead of `no_grad` 0; autograd enabled
(which neither side does) +4.2 ms on eager only.

## Consequences for reading the numbers

- The check's eager latency is the latency of the user's model as written. The kernelgen report's
  reference latency is the latency of the normalized graph that kernelgen optimizes. Its speedup
  (4.04×) is relative to that graph. Against eager in the same runtime the same kernel is 1.95×
  (10.30 versus 20.13 ms); against the check's eager number it is 2.08×.
- The TensorRT figure (4.84 ms) is fp16 while the kernelgen candidate is fp32. The two are not
  comparable with each other, only each with its own baseline.
- Any reference exported with a torch version carries that version's decompositions: with torch
  2.14.1 the same model's reference would be 38.5 ms instead of 42.8 ms in the check image.

## Measuring going forward

- Name the quantity. End-to-end latency of the model as written (the check's eager number) and the
  compiler's normalized-graph baseline (the kernelgen reference) are both valid and should be
  reported side by side, never substituted for one another.
- Compare like with like: same runtime (torch, CUDA, cuDNN, driver), same precision, same weights
  (seed the model's `__init__`, as `model.py` does), same input shapes. Report the runtime with the
  numbers; a 6 % difference came from torch 2.9.1 versus 2.14.1 alone.
- State warm or cold. Flushing L2 and synchronizing per call costs nothing here because the working
  set is hundreds of MB, but on small kernels it is the whole story. The check keeps warm, in-order,
  device-resident timing; Gaia keeps cold-L2, per-call-synchronized, interleaved timing. Both are fine
  when labelled.
- Keep reporting distributions: mean, median, p95, sample count and repeats with reversed order,
  as the check does. Add the SM clock, temperature and power sampled during the run, and confirm no
  other process holds the GPU before and after.
- Make the CPU boundary explicit. CUDA events bracket GPU time; both programs here are GPU-bound
  (GPU time equals wall time), so events are the right tool. For small or launch-bound models time
  wall-clock with a synchronize as well, and treat CUDA-graph replay as a separate, labelled number.
- Record the backend flags with every measurement. Neither side sets them; the defaults in both
  images are `matmul.allow_tf32=False`, `cudnn.allow_tf32=True`, `cudnn.benchmark=False`,
  `float32_matmul_precision=highest`.
- Treat the diagnostic transforms in `tools/reference-timing/` (re-fusing Hardswish, dropping the
  dropout clones) as instruments for attributing time, never as an optimization result.

References: the PyTorch benchmark recipe
(https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) and the TensorRT benchmarking
guide (https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html).

## Reproducing

`tools/reference-timing/run_in.sh` runs `tools/reference-timing/bench.py` in a disposable container
with the model, input, and artifact directory mounted read-only and samples the GPU before, during,
and after. Build the two derived images once (timm pinned under each image's own constraints):

```bash
docker build -t reftiming-eval-timm:local -f tools/reference-timing/Dockerfile.eval-timm .
docker build -t reftiming-kronos-timm:local -f tools/reference-timing/Dockerfile.kronos-timm .
MODEL=path/to/model.py NPZ=path/to/input.npz ARTIFACT=path/to/.artifact OUT=/tmp/reftiming \
  tools/reference-timing/run_in.sh reftiming-eval-timm:local ablations -- \
  python /work/scripts/bench.py --subjects eager pt2 pt2+noclone pt2+fusehs pt2+fusehs+noclone \
  exp_nodecomp exp_decomp --trials 2 --check --out /work/out/ablations.json
MODEL=... tools/reference-timing/run_in.sh reftiming-eval-timm:local profile -- \
  python /work/scripts/bench.py --subjects eager pt2 --profile 5 --out /work/out/profile.json
```

`--sync each --flush l2 --mode pair` reproduces Gaia's protocol; `--grad`, `--inputs`, `--weights`
select the other factors. The faithful replay is the worker image's own command:

```bash
docker run --rm --gpus all --network none -v "$ARTIFACT":/work/artifact:ro -v "$PWD/kernel":/work/kernel \
  docker.yasp.ai/yasp-eval/nvidia:1.11.0 yasp-eval kernel --reference /work/artifact/model.pt2 \
  --optimized /work/kernel --report /dev/stdout
```

where `kernel/module_source.py` is a copy of the artifact directory's `kernel.py`.

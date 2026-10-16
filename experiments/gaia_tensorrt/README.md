# Gaia kernels as TensorRT plugins: a model-specific experiment

An isolated experiment, not a feature of the check: can the CUDA kernels Gaia
generated for one model (the EfficientViT-B1 `kernel.py` in the artifact
directory) run inside a TensorRT engine, with native TensorRT for everything
else, and does that engine run without Python? It can, and it does; it is
slower than both pure TensorRT and the Gaia kernel in PyTorch. Nothing under
`src/` changes.

## What is built

- **Four `IPluginV3` plugins** (`plugins/`), one per Gaia kernel:
  `GaiaDepthwiseHswish` (`dw_kernel<K,S>`: depthwise conv fused with the
  preceding BN/bias + hardswish and the following BN/bias + hardswish, with the
  stem's second output of the pre-activated input), `GaiaEpilogue`
  (`epilogue_kernel`: BN or bias, optional hardswish, optional residual),
  `GaiaAttnPrep` and `GaiaAttnFinal` (the LiteMLA layout kernels around two
  batched matmuls). Namespace `gaia_effvit`, version `1`, INT32 fields only,
  FP32 linear static shapes only; anything else is rejected at build time and
  again at deserialization with a reason on stderr.
- **The device code is Gaia's, byte for byte.** `plugins/gaia_device_code.cuh`
  is extracted from `kernel.py` by `exporter/extract_gaia_device_code.py`;
  `plugins/provenance.json` records the hashes and a CPU test checks them. The
  one adaptation is `gaia_adapted_epilogue.cuh`: the generated epilogue updates
  its tensor in place, a plugin must not write its input, so the copy reads
  `yin` and writes `yout` with the arithmetic unchanged (a test verifies the
  diff is exactly that). The plugins' launch logic reproduces the host
  wrappers of `kernel.py` (`dw_forward`, `epilogue_`, `attn_prep`,
  `attn_final`) and is ours, not Gaia's.
- **The network** (`exporter/schedule.py`, `exporter/build_hybrid_engine.py`)
  is the op sequence of `kernel.py`'s `forward` as data: 51 native
  convolutions (1x1, the 3x3 stride-2 stem, the grouped 1x1 of the attention
  aggregation), 14 native batched matmuls, 58 plugin layers. Weights and the
  batch-norm inverse standard deviations come from `model.pt2`, the latter
  computed exactly as `kernel.py` computes them at run time, on the GPU. The
  schedule's torch-free shape inference validates the input and every weight
  before TensorRT sees them.
- **A C++ runner** (`runner/runner.cpp`) that links libnvinfer, libcudart and
  the plugin library: deserializes an engine, runs it on a raw FP32 input, and
  times two engines with the product's benchmark protocol. It runs in
  `docker/runtime/Dockerfile`: the CUDA base image, the TensorRT runtime
  package, the runner and the plugin library. No Python, no torch, no
  compiler.
- **Build-time tooling** runs only in `docker/dev/Dockerfile`: the check's
  own environment image (`kronos-edge:rtx5090-cuda130`, from the tracked
  `docker/rtx5090-cuda130/Dockerfile`) plus, pinned from NVIDIA's apt
  repository, the CUDA 13.0 compiler, build tools, Python headers, and the
  official TensorRT 11.4.0.106 C++ headers, the exact build the wheels
  implement. No experiment script imports anything beyond that image's
  packages (no `timm`: the model arrives as the exported `model.pt2`).

## Reproduce

Prerequisites: Docker with the NVIDIA runtime on an RTX 5090 with driver
580.95.05, the check's environment image built as the root `README.md`
describes (`docker build -f docker/rtx5090-cuda130/Dockerfile -t
kronos-edge:rtx5090-cuda130 .`), and a directory holding the `model.pt2` and
`kernel.py` the check produced. Then:

```bash
export ARTIFACT_DIR=/path/with/model.pt2/and/kernel.py OUT_DIR=/some/writable/dir
experiments/gaia_tensorrt/scripts/run_all.sh all       # or one stage: images extract reference engines runtime validate bench
```

`run_all.sh` builds the images, compiles the plugins and runner, extracts
the weights and the saved example input from `model.pt2`, records the
unmodified `kernel.py` output, builds four engines (hybrid and pure TensorRT,
each with TensorRT's default TF32 and with TF32 cleared), validates every
engine against the Gaia output through the TensorRT Python runtime and
through the Python-free runner, and benchmarks. Every stage that touches the
model stack runs in a container; the host runs docker and stdlib Python. The
CPU tests run with the repository's suite:

```bash
.venv/bin/pytest experiments/gaia_tensorrt/tests
```

## Results (RTX 5090, driver 580.95.05, TensorRT 11.4.0.106, CUDA 13.0, torch 2.14.1+cu130)

Input: the example input saved in `model.pt2` (`[4, 3, 1088, 1920]` FP32),
not `input.npz`. Reference: the unmodified `kernel.py` on the PT2 parameters
and buffers, deterministic over repeated runs and bitwise equal to the
decomposed PT2 graph run on the GPU.

Correctness of the hybrid engine against the Gaia output, before and after
timing, through the Python runtime and the C++ runner (identical values):

| Setting | cosine | max abs | max rel to ref max | allclose(1e-4, 1e-5) | deterministic |
|---|---:|---:|---:|---|---|
| TF32 default | 1.0 | 9.35e-5 | 2.6e-4 | no | 5/5 |
| TF32 cleared | 0.99999988 | 1.30e-6 | 3.6e-6 | yes | 5/5 |

The pure TensorRT engine against the same reference: max abs 1.01e-4 (TF32)
and 1.47e-6 (cleared). The hybrid engine's differences are of the same size
as the pure engine's in both settings, consistent with the native
convolutions and matmuls using different algorithms than cuDNN and cuBLAS;
the plugins' own contribution was not isolated. The setting without TF32 is
within single-precision rounding.

Latency, medians of 600 samples (3 trials x 200, 20 warm-ups each, reversed
order on the second trial, no cache flush):

| Setting | Subject | Timer | Median ms |
|---|---|---|---:|
| TF32 default | hybrid engine | C++ runner, non-blocking stream | 13.33 |
| TF32 default | pure TensorRT FP32 | C++ runner, non-blocking stream | 11.36 |
| TF32 default | Gaia kernel in PyTorch | `kronos_edge.benchmark_check.timed` | 11.56 |
| TF32 default | hybrid engine | `kronos_edge` EngineRunner (default stream) | 13.42 |
| TF32 default | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 11.41 |
| TF32 cleared | hybrid engine | C++ runner, non-blocking stream | 13.78 |
| TF32 cleared | pure TensorRT FP32 | C++ runner, non-blocking stream | 12.62 |
| TF32 cleared | Gaia kernel in PyTorch | `kronos_edge.benchmark_check.timed` | 11.93 |
| TF32 cleared | hybrid engine | `kronos_edge` EngineRunner (default stream) | 13.83 |
| TF32 cleared | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 12.75 |

The legacy default stream in the C++ runner adds about 0.01 ms to either
engine; the Python-side figures (default stream, torch's event timer) are
0.05 to 0.13 ms above the C++ ones, so the earlier default-stream Python
measurements of the pure engine stand.

The hybrid engine is 17% slower than pure TensorRT with TF32 and 9% slower
without, and 15% slower than the Gaia kernel in PyTorch. Per-layer profiling
(`exporter/profile_layers.py`) attributes it: the 58 plugin layers take about
4.0 ms of the 13.5 ms layer sum, and the native layers take 9.5 ms against
11.5 ms for the whole pure engine: the hybrid engine contains transpose
layers that the pure engine does not have, placed between TensorRT's
convolutions and the plugins, which accept only the linear layout, so
TensorRT changes layout on the way in and out of every plugin; the eight
largest transposes alone sum to over 2.5 ms. The Gaia kernels' advantage in PyTorch came from fusing work
PyTorch ran as separate kernels; TensorRT already fuses those into its
convolutions, so the plugins replace fused native kernels with separate
passes and add layout changes.

## Limitations

- One model, one input shape, FP32 only. The plugins reject everything else.
- The plugins accept only the linear layout. Supporting the formats TensorRT
  chooses for its convolutions could reduce the layout conversions around the
  plugins; whether and by how much was not measured, and that optimization is
  outside this experiment.
- The hybrid network is built from a hand-written schedule of this model's
  `forward`, not derived from the graph; the pure baseline is built from the
  decomposed PT2 graph, not the original eager `model.py`.
- `IProfiler` serializes layer execution, so per-layer times are for
  attribution, not for latency.
- The runtime image carries the full `libnvinfer11` package (about 3 GB); a
  lean-runtime build was not attempted.

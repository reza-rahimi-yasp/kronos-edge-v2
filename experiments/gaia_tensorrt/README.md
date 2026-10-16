# Gaia kernels as TensorRT plugins: a model-specific experiment

An isolated experiment, not a feature of the check: can the CUDA kernels Gaia
generated for one model (the EfficientViT-B1 `kernel.py` in the artifact
directory) run inside a TensorRT engine, with native TensorRT for everything
else, and does that engine run without Python? It can, and it does. The first
version (branch `experiment/gaia-tensorrt-plugins`) accepted only the linear
NCHW layout and was slower than both pure TensorRT and the Gaia kernel in
PyTorch, because TensorRT runs this model's convolutions in NHWC and had to
transpose every tensor on its way into and out of the plugins. This version
(branch `experiment/gaia-trt-layouts`) lets the plugins take TensorRT's
`kHWC` layout directly, with channel-last counterparts of the Gaia kernels
that keep the arithmetic unchanged. With the layout forced to `kHWC`, the
engine is faster than pure TensorRT and than the Gaia kernel in PyTorch (24%
and 25% at TensorRT's default TF32 setting); left to choose, TensorRT keeps
the linear layout and, for this network, builds a wrong engine. Nothing under
`src/` changes.

## What is built

- **Four `IPluginV3` plugins** (`plugins/`), one per Gaia kernel:
  `GaiaDepthwiseHswish` (`dw_kernel<K,S>`: depthwise conv fused with the
  preceding BN/bias + hardswish and the following BN/bias + hardswish, with the
  stem's second output of the pre-activated input), `GaiaEpilogue`
  (`epilogue_kernel`: BN or bias, optional hardswish, optional residual),
  `GaiaAttnPrep` and `GaiaAttnFinal` (the LiteMLA layout kernels around two
  batched matmuls). Namespace `gaia_effvit_layout`, version `1`, library
  `libgaia_trt_layout_plugins.so`, registration symbol
  `gaia_trt_layout_register_plugins`, built with hidden visibility: distinct
  from the first version's `gaia_effvit` / `libgaia_trt_plugins.so`, so both
  libraries and both kinds of engine load into one process. INT32 fields
  only, FP32 static shapes only; anything else is rejected at build time and
  again at deserialization with a reason on stderr.
- **Layouts.** Every plugin's activation tensors (its 4-D input, its residual,
  its 4-D outputs) may be in the linear NCHW layout or in TensorRT's `kHWC`
  layout (`[N][H][W][C]` in memory; `TensorFormat::kHWC`, "requires FP32,
  FP16, or UINT8 and at least three dimensions" in the 11.4 headers); within
  one plugin they share a layout. Weights, affine vectors and the 3-D
  attention operands between the two batched matmuls stay linear, since the
  matmuls take linear operands. `supportsFormatCombination` states the rule
  per tensor position, `configurePlugin` and `onShapeChange` replay it over
  the whole combination, and `enqueue` reads the layout from the tensor
  descriptors and launches the matching kernel. Each plugin has a `layout`
  field (0 = offer both and let TensorRT choose, 1 = linear only, 2 = `kHWC`
  only) that the exporter sets for the whole engine.
- **The device code for the linear layout is Gaia's, byte for byte.**
  `plugins/gaia_device_code.cuh` is extracted from `kernel.py` by
  `exporter/extract_gaia_device_code.py`; `plugins/provenance.json` records
  the hashes and a CPU test checks them. The one adaptation is
  `gaia_adapted_epilogue.cuh`: the generated epilogue updates its tensor in
  place, a plugin must not write its input, so the copy reads `yin` and writes
  `yout` with the arithmetic unchanged (a test verifies the diff is exactly
  that). The plugins' launch logic reproduces the host wrappers of
  `kernel.py` (`dw_forward`, `epilogue_`, `attn_prep`, `attn_final`) and is
  ours, not Gaia's.
- **The channel-last kernels are ours.** `plugins/gaia_hwc_kernels.cuh`
  (`dw_kernel_hwc<K,S,CT>`, `epilogue_kernel_hwc`, `attn_prep_kernel_hwc`,
  `attn_final_kernel_hwc`) was written for this experiment and is labelled
  so; it is not Gaia output. The kernels include Gaia's `Aff`, `hsw` and
  `bnf` helpers rather than redefining them, use the same rounding-controlled
  intrinsics, the same accumulation order (kh outer, kw inner, zero padding
  multiplied in), the same pre-activation rule (in-bounds loads only), the
  same `fmaxf` relu, the same `1.f` padding column and the same
  `__fdiv_rn(r, __fadd_rn(r16, 1e-5f))`; only the memory indexing and the
  thread-to-element mapping differ, so that neighbouring lanes touch
  neighbouring channels of one pixel, the contiguous direction of `kHWC`.
  The depthwise kernel tiles an input window per block into shared memory
  with lanes strided along C (32 channels, or 16 when C is not a multiple
  of 32: the stem), the epilogue's `float4` carries four channels of one
  pixel (needs C % 4 == 0, which the format rule checks), the attention prep
  loads a pixel's 48 contiguous channels and stores along pixels through
  shared memory, and the attention final writes a pixel's 16 values of a
  head contiguously. A CPU test checks the labelling, the reuse of the
  helpers and the intrinsics.
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
- **Layout inspection** (`exporter/inspect_engine.py`): the engine inspector's
  detailed layer JSON (an engine built with `ProfilingVerbosity.DETAILED`,
  which `build_hybrid_engine.py ... inspect` writes as a separate
  `.inspect.engine`; the benchmarked engines are built at the default
  verbosity, like the product's) summarized into the formats at every plugin
  boundary, every reformat layer with the formats and element counts it
  converts, and the tactics of the native convolutions and matmuls. Two
  diagnostics help explain a build: `GAIA_TRT_TRACE_FORMATS=1` makes the
  plugins print every format query, configuration and enqueue layout on
  stderr, and `GAIA_TRT_VERBOSE=1` swaps in TensorRT's verbose logger so the
  builder's per-tactic timings appear.

## Reproduce

Prerequisites: Docker with the NVIDIA runtime on an RTX 5090 with driver
580.95.05, the check's environment image built as the root `README.md`
describes (`docker build -f docker/rtx5090-cuda130/Dockerfile -t
kronos-edge:rtx5090-cuda130 .`), and a directory holding the `model.pt2` and
`kernel.py` the check produced. Then:

```bash
export ARTIFACT_DIR=/path/with/model.pt2/and/kernel.py OUT_DIR=/some/writable/dir
export GPU_LOCK=/some/lock/file                        # optional: every GPU container runs under flock(1) on it
experiments/gaia_tensorrt/scripts/run_all.sh all       # or one stage: images extract reference engines runtime validate bench
```

`run_all.sh` builds the images (`gaia-trt-layouts-dev:local`,
`gaia-trt-layouts-runtime:local`), compiles the plugins and runner, extracts
the weights and the saved example input from `model.pt2`, records the
unmodified `kernel.py` output, builds the hybrid engines for every layout
variant in `VARIANTS` (default `any hwc linear`: TensorRT's choice, `kHWC`
forced, linear forced, as `hybrid_<setting>`, `hybrid_hwc_<setting>`,
`hybrid_linear_<setting>`) with a detailed copy and its inspection each,
builds the pure TensorRT engines, validates every engine against the Gaia
output through the TensorRT Python runtime and the headline variant
(`HEADLINE`, default `hybrid_hwc`) through the Python-free runner, and
benchmarks the headline variant against pure TensorRT and against the
forced-linear variant, then profiles. Both settings (TensorRT's default TF32
and TF32 cleared) throughout. Every stage that touches the model stack runs
in a container; the host runs docker and stdlib Python. With `GPU_LOCK` set,
each GPU container holds `flock` on that file for its whole lifetime, so a
second experiment sharing the GPU never measures concurrently. The CPU tests
run with the repository's suite:

```bash
.venv/bin/pytest experiments/gaia_tensorrt/tests
```

## Results (RTX 5090, driver 580.95.05, TensorRT 11.4.0.106, CUDA 13.0, torch 2.14.1+cu130)

Input: the example input saved in `model.pt2` (`[4, 3, 1088, 1920]` FP32),
not `input.npz`. Reference: the unmodified `kernel.py` on the PT2 parameters
and buffers, deterministic over repeated runs and bitwise equal to the
decomposed PT2 graph run on the GPU. The pure TensorRT engines and the
reference outputs are the first version's (copied with their hashes
verified), so the comparison is against the original pure engine.

**What TensorRT negotiated** (engine inspector, detailed copies). The first
version's engine: 227 layers, 51 convolutions all in `(NHWC) Channel major
format` with `..._nhwckrsc_nhwc_...` TF32 implicit-GEMM tactics, the network
input moved to NHWC first, and 101 reformat layers (`__myl_Tran`,
1.32 G elements converted) around the 58 plugins, whose 156 tensor boundaries
are all linear. With the layout forced to `kHWC` (`hybrid_hwc`): 128 layers,
128 of the 156 plugin boundaries NHWC and the other 28 linear (the 3-D
attention operands, by design), and 3 reformats left at TF32 default (the
network input to NHWC, 25 M elements; one NHWC-to-NHWC copy of the stem's
second output, 33 M; the network output back to linear, 2 M) or 5 with TF32
cleared (the same plus one NCHW convolution tactic TensorRT preferred for the
stage-0 projection, with a transpose on each side, 33 M + 17 M). Left to
choose (`hybrid`, layout 0), TensorRT queried `kHWC` for every activation
tensor, accepted it, timed both layouts of every plugin (per its verbose log
the two are within a few percent of each other) and still chose linear for
all 58 plugins, keeping the reformats; it also assigned NHWC to the stem
plugin's second output and to the stem epilogue's residual while their
companion tensors stayed linear, a combination the plugins' rule forbids and
TensorRT never queried as such, and that engine's output is wrong (cosine
0.244 to the reference in both settings; the plugins report consistent
linear descriptors at run time). The forced layouts are therefore the
measured ones; `hybrid_linear` reproduces the first version's engine from
this library.

Correctness of the forced-`kHWC` engine against the Gaia output, before and
after timing, through the Python runtime and the C++ runner (identical
values):

| Setting | cosine | max abs | max rel to ref max | allclose(1e-4, 1e-5) | deterministic |
|---|---:|---:|---:|---|---|
| TF32 default | 1.0 | 9.37e-5 | 2.6e-4 | no | 5/5 |
| TF32 cleared | 0.99999988 | 1.30e-6 | 3.6e-6 | yes | 5/5 |

The same figures as the first version's linear engine (9.35e-5 and 1.30e-6)
and as `hybrid_linear` from this library (9.07e-5, 1.30e-6); the pure
TensorRT engine against the same reference: 1.01e-4 and 1.47e-6. The
channel-last kernels are consistent with the linear ones to the rounding of
TensorRT's own convolution tactics; a cosine of 1.0 does not imply allclose,
which only the TF32-cleared setting reaches.

Latency, medians of 600 samples (3 trials x 200, 20 warm-ups each, reversed
order on the second trial, CUDA events, no cache flush, no CUDA graphs):

| Setting | Subject | Timer | Median ms | Mean ms | p95 ms |
|---|---|---|---:|---:|---:|
| TF32 default | hybrid_hwc (Gaia plugins in kHWC) | C++ runner, non-blocking stream | 8.603 | 8.594 | 8.625 |
| TF32 default | pure TensorRT FP32 (original engine) | C++ runner, non-blocking stream | 11.368 | 11.368 | 11.384 |
| TF32 default | hybrid_hwc | C++ runner, legacy stream | 8.628 | 8.620 | 8.653 |
| TF32 default | pure TensorRT FP32 | C++ runner, legacy stream | 11.375 | 11.375 | 11.390 |
| TF32 default | hybrid_hwc (control run) | C++ runner, non-blocking stream | 8.653 | 8.641 | 8.677 |
| TF32 default | hybrid_linear (forced linear, same library) | C++ runner, non-blocking stream | 13.319 | 13.319 | 13.336 |
| TF32 default | Gaia kernel in PyTorch (unmodified) | `kronos_edge.benchmark_check.timed` | 11.505 | 11.491 | 11.533 |
| TF32 default | hybrid_hwc | `kronos_edge` EngineRunner (default stream) | 8.687 | 8.689 | 8.724 |
| TF32 default | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 11.417 | 11.419 | 11.440 |
| TF32 cleared | hybrid_hwc | C++ runner, non-blocking stream | 10.097 | 10.086 | 10.119 |
| TF32 cleared | pure TensorRT FP32 (original engine) | C++ runner, non-blocking stream | 12.858 | 12.859 | 12.884 |
| TF32 cleared | hybrid_hwc | C++ runner, legacy stream | 10.110 | 10.103 | 10.130 |
| TF32 cleared | pure TensorRT FP32 | C++ runner, legacy stream | 12.857 | 12.859 | 12.887 |
| TF32 cleared | hybrid_hwc (control run) | C++ runner, non-blocking stream | 10.111 | 10.100 | 10.130 |
| TF32 cleared | hybrid_linear (forced linear, same library) | C++ runner, non-blocking stream | 13.773 | 13.777 | 13.814 |
| TF32 cleared | Gaia kernel in PyTorch (unmodified) | `kronos_edge.benchmark_check.timed` | 11.904 | 11.894 | 11.927 |
| TF32 cleared | hybrid_hwc | `kronos_edge` EngineRunner (default stream) | 10.159 | 10.159 | 10.187 |
| TF32 cleared | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 12.937 | 12.938 | 12.975 |

Ratios of medians (C++ runner, non-blocking): pure / hybrid_hwc 1.32 with
TF32 and 1.27 without; hybrid_linear / hybrid_hwc 1.54 and 1.36. In Python,
Gaia / hybrid_hwc 1.32 and 1.17, pure / hybrid_hwc 1.31 and 1.27. The
first version's linear engine measured 13.33 and 13.78 ms in the same
runner; `hybrid_linear` reproduces it (13.32, 13.77).

Per-layer attribution (`exporter/profile_layers.py`, `IProfiler`, which
serializes layers, so sums exceed the latencies): hybrid_hwc at TF32 default
sums to 8.65 ms, 4.09 ms in the 58 plugins and 4.56 ms in 70 native layers;
the first version's engine summed to 13.55 ms, with 9.52 ms in 169 native
layers, of which the transposes were the largest items. The plugins' own
times did not change (4.03 ms linear in the first version; TensorRT's
builder timed the two layouts of each plugin within a few percent of each
other). So the difference to the first version is the removed reformats, and
the difference to pure TensorRT (11.54 ms over 156 layers, the largest
being its separate `__myl_AddMaxMinMulMul` elementwise kernels for the
BN + hardswish stages) is that Gaia's depthwise kernels carry those
elementwise stages fused on both sides while TensorRT runs them as separate
passes. These are attributions from a serializing profiler, not a causal
proof.

## Limitations

- One model, one input shape, FP32 only. The plugins reject everything else.
- The layout is chosen per engine by the exporter (`hwc`, `linear`, or
  TensorRT's choice); within a plugin all activation tensors share it, so a
  plugin cannot serve a mixed assignment. TensorRT's own choice produced a
  mixed assignment on the stem that the plugins never agreed to and a wrong
  engine; why its selection ignores the reformats it then inserts was not
  investigated beyond its verbose log.
- The inspected engines are separate builds of the same network at detailed
  verbosity; the benchmarked engines are built at the default verbosity.
  With a forced layout the plugin boundaries are the same in both, but the
  tactics TensorRT picks for its own layers can differ between builds.
- The hybrid network is built from a hand-written schedule of this model's
  `forward`, not derived from the graph; the pure baseline is built from the
  decomposed PT2 graph, not the original eager `model.py`.
- `IProfiler` serializes layer execution, so per-layer times are for
  attribution, not for latency.
- The runtime image carries the full `libnvinfer11` package (about 3 GB); a
  lean-runtime build was not attempted.

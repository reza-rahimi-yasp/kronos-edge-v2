# The whole Gaia kernel in one TensorRT plugin

An isolated experiment on top of the hybrid one in the parent directory: instead
of one plugin per Gaia kernel with native TensorRT in between, the entire forward
of the Gaia-generated `kernel.py` (EfficientViT-B1) runs inside **one**
`IPluginV3`. The four Gaia CUDA kernels run exactly as generated (the epilogue
in place), the 51 convolutions go through cuDNN's backend graph API and the 14
batched matrix multiplies through cuBLAS, driven the way torch 2.14.1 drives
them for `F.conv2d` and `torch.bmm`. The result is an engine whose output is
bitwise equal to the unmodified `kernel.py` in both settings, runs without
Python, and matches the Gaia kernel's PyTorch latency. Nothing under `src/`
changes, and nothing here is a product feature.

## What is built

- **One plugin**, `GaiaEffVitWholeModel` (namespace `gaia_effvit_whole`,
  version `1`; `plugin/`). Its fields are the op program, every weight of the
  model as one FP32 blob, and two switches, `cudnn_tf32` and `cublas_tf32`,
  which mirror `torch.backends.cudnn.allow_tf32` and
  `torch.backends.cuda.matmul.allow_tf32`. The program (`exporter/whole_program.py`)
  is the hybrid experiment's `schedule.build`, laid out as fixed-width INT32
  records with tensor ids and weight offsets; an epilogue's output aliases its
  input, so the generated in-place `epilogue_kernel` runs unchanged. The plugin
  decodes and validates the program itself (every shape recomputed, every weight
  offset bounded) and rejects any input shape but the one it was built for.
  Each execution context (`attachToContext`) gets its own cuDNN and cuBLAS
  handles, the weights on the device, an activation arena planned once from
  tensor lifetimes, the workspaces, and one cuDNN execution plan per distinct
  convolution. Symbols are hidden except the entry point
  `gaia_trt_whole_model_register_plugins`, so this library co-loads with the
  hybrid `libgaia_trt_plugins.so` (`runner_whole` registers both).
- **The device code is Gaia's, byte for byte**: `plugin/gaia_whole_model_plugin.cu`
  includes the parent experiment's extracted `gaia_device_code.cuh` (hash-checked
  by its tests) inside this library's namespace, and launches the kernels with
  the host logic of `kernel.py`'s wrappers.
- **The libraries are torch's own.** The dev image's venv carries the cuDNN
  (9.24.0.43) and cuBLAS (13.1.1.3) builds torch 2.14.1+cu130 links; the
  Makefile compiles and links against them, and the runtime image
  (`docker/runtime/Dockerfile`) pins the same builds from NVIDIA's apt
  repository. No interpreter, no torch, no compiler, no NVRTC in the runtime image.
- **What is mirrored from PyTorch**, measured on the unmodified `kernel.py` with
  cuDNN's and cuBLAS's API logging (`CUDNN_LOGLEVEL_DBG=3`, `CUBLAS_LOGINFO_DBG=1`)
  and read against the PyTorch source:
  - `F.conv2d`: the cuDNN backend graph API with FP32 NCHW tensors of 32-byte
    alignment, compute type float, cross-correlation, alpha 1 and beta 0;
    engine heuristics `CUDNN_HEUR_MODE_INSTANT`, then the fallback list; configs
    with `CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS` skipped, and with
    `CUDNN_NUMERICAL_NOTE_TENSOR_CORE` when `cudnn_tf32` is off; ATen's hardcoded
    errata for this GPU (compute capability 12.0 with cuBLAS older than 13.6.1:
    engines 58 and 63 never run); the first remaining config whose plan finalizes
    and executes is kept, one plan per distinct convolution.
  - `torch.bmm`: `cublasSgemmStridedBatched` on a handle with a 32 MiB workspace
    and `CUBLAS_DEFAULT_MATH` (`CUBLAS_TF32_TENSOR_OP_MATH` when `cublas_tf32` is
    on), the operands in ATen's swapped row-major form, the transposed `q` as
    `CUBLAS_OP_T`.
  With `GAIA_TRT_WHOLE_LOG_PLANS=1` the plugin logs the engine, knobs and notes
  of every convolution; those logs equal PyTorch's choices for all 51
  convolutions in both settings.

## Reproduce

Prerequisites are the hybrid experiment's: the check's environment image, a
directory with `model.pt2` and `kernel.py`, and the hybrid experiment's output
directory for the shared inputs (`weights.npz`, the saved input, the Gaia
reference outputs, the pure TensorRT engines), which are copied and
hash-recorded rather than regenerated. Every GPU stage holds the `flock`
`GPU_LOCK` for its whole lifetime.

```bash
export ARTIFACT_DIR=/path/with/model.pt2/and/kernel.py BASELINE_DIR=/hybrid/experiment/out OUT_DIR=/some/writable/dir
experiments/gaia_tensorrt/whole_model/scripts/run_whole_model.sh all   # or: images baseline engines runtime validate bench
```

The CPU tests run with the repository's suite:

```bash
.venv/bin/pytest experiments/gaia_tensorrt/whole_model/tests
```

## Results (RTX 5090, driver 580.95.05, TensorRT 11.4.0.106, CUDA 13.0, torch 2.14.1+cu130, cuDNN 9.24.0.43, cuBLAS 13.1.1.3)

Input and reference as in the hybrid experiment: the example input saved in
`model.pt2` (`[4, 3, 1088, 1920]` FP32) and the unmodified `kernel.py` output on
the PT2 parameters and buffers. Settings: `default` is torch's defaults as the
harness ran `kernel.py` (cuDNN TF32 on, matmul TF32 off; the pure engine with
TensorRT's TF32 flag), `notf32` is both off (the pure engine with the flag cleared).

Correctness of the whole-model engine against the Gaia output, before and after
timing, through the TensorRT Python runtime and through the interpreter-free C++
runner (identical bytes):

| Setting | cosine | max abs | allclose(1e-4, 1e-5) | bitwise equal | deterministic |
|---|---:|---:|---|---|---|
| default | 1.0 | 0.0 | yes | **yes** | 5/5 |
| notf32 | 0.99999988 | 0.0 | yes | **yes** | 5/5 |

(0.99999988 is the float32 cosine of the reference with itself; the outputs are
byte-identical.) The pure TensorRT engine against the same reference: max abs
1.01e-4 (default) and 1.47e-6 (notf32), bitwise equal in neither.

Latency, medians of 600 samples (3 trials x 200, 20 warm-ups each, reversed
order on the second trial, no cache flush, CUDA-event timing):

| Setting | Subject | Timer | Median ms |
|---|---|---|---:|
| default | whole-model engine | C++ runner_whole, non-blocking stream | 11.49 |
| default | pure TensorRT FP32 | C++ runner_whole, non-blocking stream | 11.45 |
| default | whole-model engine | C++ runner_whole, legacy stream | 11.54 |
| default | pure TensorRT FP32 | C++ runner_whole, legacy stream | 11.46 |
| default | Gaia kernel in PyTorch | `kronos_edge.benchmark_check.timed` | 11.55 |
| default | whole-model engine | `kronos_edge` EngineRunner (default stream) | 11.55 |
| default | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 11.46 |
| notf32 | whole-model engine | C++ runner_whole, non-blocking stream | 11.94 |
| notf32 | pure TensorRT FP32 | C++ runner_whole, non-blocking stream | 12.95 |
| notf32 | whole-model engine | C++ runner_whole, legacy stream | 11.96 |
| notf32 | pure TensorRT FP32 | C++ runner_whole, legacy stream | 12.92 |
| notf32 | Gaia kernel in PyTorch | `kronos_edge.benchmark_check.timed` | 11.92 |
| notf32 | whole-model engine | `kronos_edge` EngineRunner (default stream) | 11.95 |
| notf32 | pure TensorRT FP32 | `kronos_edge` EngineRunner (default stream) | 12.95 |

The whole-model engine runs within 0.4% of the Gaia kernel in PyTorch in both
settings (it is the same kernels and library calls), 0.3% to 0.8% slower than
pure TensorRT with TF32, and 8% faster than pure TensorRT without TF32. The
hybrid engine of the parent experiment, for comparison, was 17% and 9% slower
than pure TensorRT. The whole-model engine has no layout conversions and no
per-layer TensorRT overhead: TensorRT sees one layer.

## Differences from PyTorch, and limitations

- PyTorch tries the top heuristic config first, then the full list minus that
  one, then the fallback list; the plugin walks the full list once, then the
  fallback list. Both pick the first acceptable config, and the logged choices
  agree for every convolution of this model.
- PyTorch's errata also skips engine 5 for cuDNN 9.23.1 to 9.24.0; the plugin
  does not, and the logs show engine 5 was never proposed for this model. The
  plugin additionally skips engines flagged `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`
  (the runtime has no NVRTC); none was proposed for this model either.
- PyTorch allocates the cuDNN workspace per call from its caching allocator;
  the plugin keeps one buffer sized to the largest plan. PyTorch keeps one cuDNN
  handle per thread; the plugin keeps one per execution context.
- One model, one input shape, FP32 only, one GPU (`sm_120`). The engine file
  carries the weights (37 MB) and no cuDNN plan: plans are selected when an
  execution context is created, on whatever cuDNN build the runtime has, which
  is why the runtime image pins torch's.
- The TensorRT builder does not run a one-plugin network, so its TF32 flag,
  timing and tactics play no part; the engine builds in under a second.
- The in-place epilogue, the residual reads and the stem's second output rely on
  the program's aliasing and lifetimes, which the plugin validates but which
  were written for this model's schedule.

## API documentation used

- cuDNN backend (graph) API: descriptors, attributes, heuristics, execution
  plans, variant packs, numerical and behavior notes:
  https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html
- cuDNN API logging (`CUDNN_LOGLEVEL_DBG`, `CUDNN_LOGDEST_DBG`):
  https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/troubleshooting.html
- cuBLAS `cublasSgemmStridedBatched`, `cublasSetMathMode`, `cublasSetWorkspace`,
  `cublasSetStream`: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched,
  https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode,
  https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace,
  https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
- TensorRT `IPluginV3` in C++ (creators, namespaces, serialization, lifecycle):
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html,
  and the `IPluginV3OneRuntime::attachToContext` contract in `NvInferRuntime.h`
  of the installed TensorRT 11.4.0.106 headers.
- PyTorch's cuDNN convolution path (heuristics mode, filters, errata):
  https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp

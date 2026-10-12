# kronos-edge-v2

Checks whether a PyTorch model runs in eager mode with the inputs you supply,
in an environment we support. You give it a `config.yaml` naming the GPU and
software versions you want, a `model.py`, and an `input.npz`. It matches the
config to a supported environment, then runs the check inside that
environment's Docker container on the GPU: it loads the model, validates the
inputs against the model's own reference inputs, runs one forward pass,
exports the model to ONNX with the dynamo exporter, simplifies the ONNX with
onnxsim, builds a TensorRT engine at the precision you ask for, benchmarks
the engine against eager mode, checks their outputs agree, and prints either
the benchmark result or a message saying what to fix. The ONNX files, the
engine, and the benchmark numbers land in an artifact directory.

## Requirements

- Python 3.12 or newer
- The packages in `requirements.txt` (`langgraph`, `numpy`, `onnx`, `onnxscript`, `onnxsim`, `pydantic`, `pyyaml`, `torch`)
- Docker with the NVIDIA container runtime, and `nvidia-smi` on the path

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The package lives under `src/`, so set `PYTHONPATH=src` when running it from
a checkout. The examples below do that.

## Prepare your config file

`config.yaml` names the environment you want and the settings for the
checks. All eight fields are required and no other fields are allowed.
Version numbers must be quoted so YAML does not read them as numbers;
`threshold` is a number.

| Field | Type | Example |
|---|---|---|
| `gpu` | non-empty string | `NVIDIA GeForce RTX 5090` |
| `driver` | dotted version string | `"580.95.05"` |
| `cuda` | dotted version string | `"13.0"` |
| `tensorrt` | dotted version string | `"11.4.0"` |
| `torch` | dotted version string | `"2.14.1"` |
| `precision` | `fp32` or `fp16` | `fp16` |
| `metric` | `cosine` | `cosine` |
| `threshold` | number from 0 to 1 | `0.99` |

The first five fields must match one supported environment exactly. The
supported environments are listed in `src/kronos_edge/environments.yaml`.
There is one today:

```yaml
gpu: NVIDIA GeForce RTX 5090
driver: "580.95.05"
cuda: "13.0"
tensorrt: "11.4.0"
torch: "2.14.1"
```

`precision` is a setting, not part of the environment. Before the ONNX
export, a copy of the model and its floating-point inputs is cast to the
precision you chose, whatever precision you wrote the model in, so
`model.onnx`, `model_simplified.onnx`, and `model.engine` all carry it;
integer inputs are left alone. The eager run always uses the model as you
wrote it. TensorRT builds a strongly typed engine, so the engine
computes in exactly the types the ONNX carries. Changing `precision` and
running again rewrites all three files in the same artifact directory.

`metric` names how the engine's outputs are compared with the eager outputs.
Only cosine similarity is implemented, so only `cosine` is accepted.
`threshold` is the minimum value every output must reach for the check to
pass. Pick it for your model: `0.99` is a reasonable start for FP32, and FP16
models may need a little slack.

## Prepare your model file

`model.py` must define:

| Name | Required | What it is |
|---|---|---|
| `Model` | yes | A `torch.nn.Module` subclass. |
| `get_inputs()` or `get_fwd_args()` | yes, either name | Returns a list or tuple of tensors, one per `forward` argument, with the shapes and dtypes you expect. |
| `get_init_inputs()` | only if `Model.__init__` takes arguments | Returns the constructor arguments. The model is built as `Model(*get_init_inputs())`. |

Example:

```python
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
```

## Prepare your input file

`input.npz` holds one array per tensor returned by `get_inputs()`, in the same
order, under the keys `arr_0`, `arr_1`, and so on. Each array must match the
reference tensor at that position in shape and dtype exactly. This is what
`numpy.savez` writes when you pass the arrays positionally:

```python
import numpy as np

np.savez("input.npz", np.zeros((1, 3, 8, 8), dtype=np.float32))
```

## Build the environment image

Each supported environment has a Dockerfile under `docker/` and a Docker image
tag in `environments.yaml`. The eager check runs inside that image with the
GPU attached, so the check refuses to run until the image exists on this
machine. Build it once, and rebuild it after changing anything under `src/`:

```bash
docker build -f docker/rtx5090-cuda130/Dockerfile -t kronos-edge:rtx5090-cuda130 .
```

## Run the check

```bash
PYTHONPATH=src python -m kronos_edge.workflow path/to/config.yaml path/to/model.py path/to/input.npz
```

On success it prints the benchmark, for example:

```
Eager run, ONNX export, ONNX simplification, TensorRT engine build (fp16), and benchmark succeeded.
Latency per run, eager: mean 0.047 ms, median 0.046 ms, p95 0.052 ms.
Latency per run, TensorRT: mean 0.015 ms, median 0.015 ms, p95 0.017 ms.
TensorRT is 3.0x faster than eager (median). Cosine similarity: 0.9998 (threshold 0.99).
```

The benchmark warms each side up with 20 runs, then times 200 runs one by
one and reports the mean, median, and 95th percentile of those latencies.
The speedup is the ratio of the medians. Eager runs the model as you wrote
it; TensorRT runs the engine at the configured precision. Each output of the
model is compared with the engine's output at the same position by the
configured metric, and every value is printed.

The ONNX export runs `torch.onnx.export(..., dynamo=True)` on the same model
and inputs the forward pass used, `onnxsim` simplifies the result, and
TensorRT builds a serialized engine from the simplified model at the
configured precision. If the model uses an input's values as a shape, for
example `x.reshape(shape.tolist())`, the engine is built for exactly the
values in your `input.npz`. Engine builds take seconds for small models and
can take minutes for large ones; there is no timeout.

## Artifact directory

The files the checks produce are written to `.artifact` in the current
directory. Pass `--artifacts path/to/dir` to use another directory. It is
created if missing and mounted into the container, so the files are owned by
the container's user. Each run first deletes the files listed below from a
previous run and nothing else, so your own files in that directory are safe.
On a failure, the files produced before it are kept.

| File | Written by |
|---|---|
| `model.onnx` | The ONNX export. |
| `model_simplified.onnx` | The onnxsim simplification. |
| `model.engine` | The serialized TensorRT engine, at the configured precision. Specific to this GPU and TensorRT version. |
| `benchmark.json` | The latency statistics of both sides, the speedup, and the metric values, as printed. |

When a model's weights exceed the 2 GB protobuf limit they are stored next to
the `.onnx` file in a companion `.data` file of the same name, for example
`model.onnx.data`. Keep the two files together.

and exits with code 0. On failure it prints a message naming the problem,
followed by the underlying exception, and exits with code 1. For example:

```
Your model cannot be run through eager mode. Either fix it or contact support.
AttributeError: 'Model' object has no attribute 'relo'
```

Log lines go to stderr, including the ones from inside the container. Pick
the level with `--log-level`; the levels are
`TRACE`, `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. `TRACE` prints the shape
and dtype of every input, `DEBUG` adds the full traceback of a failure.

## Failure messages

| Message | What to do |
|---|---|
| config.yaml could not be read. | Check the path and that the file is valid YAML. |
| config.yaml does not match the expected schema. | The lines below it name each field and what is wrong with it. |
| The supported environment registry could not be loaded. Contact support. | Not something you can fix from the config. |
| No supported environment matches config.yaml. | The detail line shows what you asked for and every supported environment. |
| The GPU in this machine does not match the requested environment. | The environment exists but this machine cannot provide it. |
| The Docker image for the requested environment is not built. | Run the `docker build` command shown in the detail line. |
| The check could not be run in the environment container. Contact support. | `docker run` failed before the check could report. |
| The GPU could not be accessed inside the environment container. Contact support. | The container started but torch could not see the GPU. |
| model.py could not be loaded. | Fix the syntax or import error shown below it, or check the path. |
| model.py does not define a 'Model' class. | Add a `Model` class that subclasses `torch.nn.Module`. |
| model.py does not define get_inputs() or get_fwd_args(). | Add one of them, returning the reference inputs. |
| Model could not be constructed. | Add `get_init_inputs()` returning the constructor arguments named in the detail line. |
| get_inputs() raised an exception. | Fix `get_inputs()`. |
| input.npz could not be read. | Check the path, and that the file was written with `numpy.savez`. |
| input.npz does not match the inputs returned by get_inputs(). | The detail line names the key and the expected versus found shape and dtype. |
| Your model cannot be run through eager mode. Either fix it or contact support. | The forward pass raised. The detail line shows the exception. |
| The ONNX exporter is not available inside the environment container. Contact support. | The image lacks `onnx` or `onnxscript`. |
| Your model could not be cast to the requested precision for export. Either fix it or contact support. | Casting a copy of the model to half raised; the detail shows the exception. |
| Your model could not be traced by torch.export for ONNX export. Fix the construct named below. | `torch.export` could not capture the model, for example a branch on a tensor value. The detail names the construct and the line in `model.py`. |
| Your model was traced but an operator has no ONNX translation. Replace the operator named below or contact support. | The detail names the operator the dynamo exporter cannot translate. |
| The exported ONNX model could not be serialized or written to the artifact directory. | Check the directory is writable and the machine has memory for the model; the detail shows the error. |
| The ONNX simplifier is not available inside the environment container. Contact support. | The image lacks `onnxsim`. |
| The ONNX model could not be simplified by onnxsim. Either fix the model or contact support. | onnxsim raised. The detail shows the exception. |
| onnxsim produced a simplified model that does not pass its check or the ONNX checker. Contact support. | Not something you can fix from the model. |
| The simplified ONNX model could not be written to the artifact directory. | Check the directory is writable; the detail shows the OS error. |
| TensorRT is not available inside the environment container. Contact support. | The image lacks `tensorrt`. |
| TensorRT could not parse the ONNX model. Replace the operator or node named below or contact support. | The detail lists the parser's errors, typically an operator TensorRT has no importer for. |
| TensorRT could not build an engine from the model. The errors it reported are below. | The detail lists what TensorRT logged, for example an unsupported layer configuration or running out of GPU memory. |
| The TensorRT engine could not be written to the artifact directory. | Check the directory is writable; the detail shows the OS error. |
| Your model failed while being run repeatedly in eager mode for timing. Either fix it or contact support. | The forward pass passed once but raised when repeated; the detail shows the exception. |
| The TensorRT runtime is not available inside the environment container. Contact support. | The image lacks `tensorrt`. |
| The TensorRT engine could not be loaded or run. The errors it reported are below. | The detail lists what TensorRT logged. |
| The TensorRT engine's outputs do not have the same structure as the eager outputs. Contact support. | The count or a shape differs; the detail names the output. |
| The TensorRT engine's outputs do not match the eager outputs within the threshold in config.yaml. | The detail names each output below the threshold with its value. Lower the threshold if the gap is acceptable for your model, or use `fp32`. |

## Run the tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest
```

`pyproject.toml` points pytest at `src/`, so no install of the package is
needed. The tests write their model and input files to a temporary directory;
nothing binary is checked in.

See `architecture.md` for how the pieces fit together.

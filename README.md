# kronos-edge-v2

Checks whether a PyTorch model runs in eager mode with the inputs you supply,
in an environment we support. You give it a `config.yaml` naming the GPU and
software versions you want, a `model.py`, and an `input.npz`. It matches the
config to a supported environment, then runs the check inside that
environment's Docker container on the GPU: it loads the model, validates the
inputs against the model's own reference inputs, runs one forward pass, and
prints either a success line or a message saying what to fix.

## Requirements

- Python 3.12 or newer
- The packages in `requirements.txt` (`langgraph`, `numpy`, `pydantic`, `pyyaml`, `torch`)
- Docker with the NVIDIA container runtime, and `nvidia-smi` on the path

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The package lives under `src/`, so set `PYTHONPATH=src` when running it from
a checkout. The examples below do that.

## Prepare your config file

`config.yaml` names the environment you want. All five fields are required,
all values are strings, and no other fields are allowed. Version numbers must
be quoted so YAML does not read them as numbers.

| Field | Type | Example |
|---|---|---|
| `gpu` | non-empty string | `NVIDIA GeForce RTX 5090` |
| `driver` | dotted version string | `"580.95.05"` |
| `cuda` | dotted version string | `"13.0"` |
| `tensorrt` | dotted version string | `"11.4.0"` |
| `torch` | dotted version string | `"2.14.1"` |

The config must match one supported environment exactly on every field. The
supported environments are listed in `src/kronos_edge/environments.yaml`.
There is one today:

```yaml
gpu: NVIDIA GeForce RTX 5090
driver: "580.95.05"
cuda: "13.0"
tensorrt: "11.4.0"
torch: "2.14.1"
```

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

On success it prints:

```
Eager mode run succeeded.
```

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

## Run the tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest
```

`pyproject.toml` points pytest at `src/`, so no install of the package is
needed. The tests write their model and input files to a temporary directory;
nothing binary is checked in.

See `architecture.md` for how the pieces fit together.

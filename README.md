# kronos-edge-v2

Checks whether a PyTorch model runs in eager mode with the inputs you supply.
You give it a `model.py` and an `input.npz`; it loads the model, validates the
inputs against the model's own reference inputs, runs one forward pass, and
prints either a success line or a message saying what to fix.

## Requirements

- Python 3.12 or newer
- The packages in `requirements.txt` (`langgraph`, `numpy`, `torch`)

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The package lives under `src/`, so set `PYTHONPATH=src` when running it from
a checkout. The examples below do that.

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

## Run the check

```bash
PYTHONPATH=src python -m kronos_edge.workflow path/to/model.py path/to/input.npz
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

Log lines go to stderr. Pick the level with `--log-level`; the levels are
`TRACE`, `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. `TRACE` prints the shape
and dtype of every input, `DEBUG` adds the full traceback of a failure.

## Failure messages

| Message | What to do |
|---|---|
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

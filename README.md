# kronos-edge-v2

Checks whether a PyTorch model runs in eager mode with the inputs you supply,
in an environment we support, then asks yasp's Gaia for an optimized kernel of
it. You give it a `config.yaml` naming the GPU and software versions you want,
a `model.py`, and an `input.npz`. It matches the config to a supported
environment and verifies, before anything runs, that the Python environment
you started it in is that environment: the torch, CUDA build, and TensorRT
versions, the GPU, and the `yasp-toolkit` command. Then it runs the check in
that environment on the GPU, in a subprocess: it loads the model, validates
the inputs against the model's own reference inputs, runs one forward pass,
exports the model to ONNX with the dynamo exporter, simplifies the ONNX with
onnxsim, builds a TensorRT engine at the precision you ask for, benchmarks the
engine against eager mode, and checks their outputs agree. It then exports the
kernelgen reference with `yasp-toolkit kernelgen export`, confirms your Gaia
worker for the config's GPU and routing key is registered, and runs
`yasp-toolkit kernelgen compile`. It prints the benchmark and the kernelgen
report, or a message saying what to fix. The ONNX files, the engine, the
benchmark numbers, the reference, the kernel, and its report land in an
artifact directory.

The same workflow runs natively or inside a container you built from the
environment's Dockerfile under `docker/`: it checks the environment it finds
itself in and never runs Docker.

Every Gaia step uses only the commands the
[yasp documentation](https://compile.yasp.ai/help/) publishes, following its
[Generate your first kernel](https://compile.yasp.ai/help/gaia/generate-your-first-kernel/)
workflow: start a runner, export, generate, inspect. This repository
orchestrates them; it implements no part of Gaia.

## Requirements

- Python 3.12 or newer
- A supported environment (see "Set up the environment"): the packages in
  `requirements.txt` (`langgraph`, `numpy`, `onnx`, `onnxscript`, `onnxsim`,
  `pydantic`, `pyyaml`, `torch`) with torch's CUDA build, TensorRT, and the
  GPU of the environment you name in `config.yaml`
- `yasp-toolkit`, from `yasp-agent` installed as the yasp docs describe (see "Install")
- A Yasp API key for the kernelgen steps, and a registry key derived from it
  for the install (see below)
- A Gaia worker on this machine, deployed once and registered under your
  config's `eval_routing` (see "Deploy the Gaia worker")

## Install

The check itself installs from PyPI. Install torch from the index of the
environment's CUDA build, not PyPI's default, so its CUDA version is the
one `config.yaml` names (see "Set up the environment"):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`yasp-toolkit` comes from `yasp-agent`, which is published on the yasp package
index, not on PyPI. The check runs it as a command and never imports it, so it
is installed on its own, the way the
[pipx guide](https://compile.yasp.ai/help/advanced/pipx/) of the yasp docs
shows. That index is read with a registry key: a scoped, long-lived key
(`nexus-read`) exchanged from your Yasp API key as the
[install guide](https://compile.yasp.ai/help/install/#get-a-registry-key)
describes (its exchange needs no package installed), or, once the toolkit is
installed, printed by `yasp-toolkit registry key`. Put it in `REGISTRY_KEY` and
hand it to pip through a netrc file, never on the command line, where it would
show in process listings while pip runs. The file lives in a private temporary
path and is removed when the subshell ends, success or failure:

```bash
(
  set -e; umask 077
  NETRC_FILE=$(mktemp); trap 'rm -f "$NETRC_FILE"' EXIT
  printf 'machine nexus.yasp.ai\nlogin token\npassword %s\n' "$REGISTRY_KEY" > "$NETRC_FILE"
  NETRC="$NETRC_FILE" pipx install \
      --pip-args="--extra-index-url https://nexus.yasp.ai/repository/pypi-hosted/simple/" \
      'yasp-agent[gaia,kronos]==0.5.2'
)
```

`NETRC` names the file pip reads credentials from, so the index URL stays
credential-free; the registry key is the HTTP Basic password for the user
`token`, as the [registry reference](https://compile.yasp.ai/help/reference/registry/)
states. The evaluator is not a package to install: it is the Gaia worker
container (see "Deploy the Gaia worker").

The package lives under `src/`, so set `PYTHONPATH=src` when running it from
a checkout. The examples below do that.

## Set up your Yasp API key

`yasp-toolkit` needs a Yasp API key. Either export it in the shell that runs
the check, or store it in the OS keyring once
([keyring reference](https://compile.yasp.ai/help/reference/keyring/)):

```bash
export YASP_API_KEY=...       # the variable yasp-toolkit reads
yasp-toolkit keyring set      # or: keep it in the keyring instead
```

`.env.example` documents the variable. Nothing in the check reads a `.env`
file: if you keep the key in one, load it yourself (`set -a; . ./.env; set +a`)
and never commit it. `.env` is gitignored. Only `yasp-toolkit` reads the key.
Your model never sees it: the model check and `yasp-toolkit kernelgen export`,
the steps that run `model.py`, start without `YASP_API_KEY`, any other `YASP_`
variable, or `REGISTRY_KEY` in their environment, and with Python's `keyring`
package set to its null backend; the export needs no key. The worker listing,
the generation, and the report inspection keep your environment and
authenticate as usual. This withholds the key from your model's environment;
it is not a sandbox, since your model still runs as you.

## Deploy the Gaia worker

Gaia scores every candidate kernel on a Gaia worker, a container that runs on
the target GPU and registers with yasp. The docs deploy it once and reuse it:
"You deploy it once; it stays up and serves all your generations"
([Generate your first kernel, step 1](https://compile.yasp.ai/help/gaia/generate-your-first-kernel/#step-1-start-a-runner)).
Deploy it on this machine by following the
[Gaia Docker guide](https://compile.yasp.ai/help/gaia/workers/docker/) as
written, with one addition: register it in your own routing pool by adding
`--tag routing=<eval_routing>` to its `yasp-eval register` line, as
[Tags & routing](https://compile.yasp.ai/help/gaia/workers/#tags-routing)
describes. For example, with `eval_routing: reza-fractal`:

```bash
... $GAIA_IMAGE yasp-eval register \
    --tag routing=reza-fractal \
    --tag host=$(hostname -s) --tag version=$GAIA_TAG --tag last_update=$(date +%F)
```

Then confirm it is listed with your GPU and routing key:

```bash
yasp-toolkit eval workers --tag routing=reza-fractal
```

The check runs that same listing before any kernelgen job and stops if no
worker advertising exactly your config's `gpu` is registered under its
`eval_routing`. It never starts, stops, or removes a worker, and mints no key:
the worker, its restart policy, health check, and autoheal sidecar are yours,
as the guide sets them up.

## Prepare your config file

`config.yaml` names the environment you want, the settings for the checks,
and the evaluator for the kernelgen kernel. All nine fields are required and
no other fields are allowed. Version numbers must be quoted so YAML does not
read them as numbers; `threshold` is a number.

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
| `eval_routing` | non-empty string | `reza-fractal` |

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

Before anything runs, the check verifies that the Python environment it was
started in is that entry: torch's version and CUDA build
(`torch.version.cuda`), TensorRT's version, and the GPU torch sees must be
the entry's. Versions match by component, so the entry's `11.4.0` accepts
TensorRT's `11.4.0.106` and `2.14.1` accepts torch's `2.14.1+cu130`. The
`driver` field identifies the entry and is not checked against this machine.
Every difference is reported at once, with the entry's Dockerfile as the
recipe (see "Set up the environment"). `yasp-toolkit` must be on the path.

`precision` is a setting, not part of the environment. Before the ONNX
export, a copy of the model and its floating-point inputs is cast to the
precision you chose, whatever precision you wrote the model in, so
`model.onnx`, `model_simplified.onnx`, and `model.engine` all carry it;
integer inputs are left alone. The eager run always uses the model as you
wrote it, and the kernelgen reference (`model.pt2`) is exported from
`model.py` itself, uncast (see "Run the check"). TensorRT builds a
strongly typed engine, so the engine computes in exactly the types the ONNX
carries. Changing `precision` and running again rewrites all the files in the
same artifact directory.

`eval_routing` is also a setting. It is the routing key kernelgen's
evaluations are pinned to (`kernelgen compile --eval-routing`): they go to the
Gaia worker you registered with `--tag routing=<eval_routing>` (see "Deploy the
Gaia worker"). Pick a key of your own, so a job never lands on someone else's
worker. It is required: a worker without a routing tag joins the shared pool,
"where evaluations route unless told otherwise"
([Tags & routing](https://compile.yasp.ai/help/gaia/workers/#tags-routing)),
and the check must evaluate on this machine, so a config that omits it is
rejected before any job is submitted.

Packages `model.py` imports beyond torch are yours to install into the
environment the check runs in (`pip install timm`, say); the check installs
nothing. If `model.py` imports a package that is not installed, the check
stops at the import and names the module. Pick versions that keep the
environment's torch: a package that replaces it changes the environment the
checks were validated in, and the next run's preflight will say so.
`yasp-toolkit kernelgen export` imports `model.py` too, with the toolkit's own
torch (see "Run the check"), so the package must be importable there as well.

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

## Set up the environment

The check runs the model in the Python environment you start it in, and
refuses to run unless that environment is the one `config.yaml` names. Each
supported environment has a Dockerfile under `docker/` (named in
`environments.yaml`) that is the recipe for it; the check never builds or
runs an image. Get the environment either way:

- Natively: install torch from the index of the entry's CUDA build and the
  TensorRT the entry names, as the Dockerfile does (`torch==2.14.1` from
  `https://download.pytorch.org/whl/cu130`, `tensorrt-cu13==11.4.0.106`),
  plus `requirements.txt` and `yasp-agent` (see "Install"), on a machine with
  the entry's GPU.
- In a container: build the Dockerfile and run the check inside it with the
  GPU attached. The image copies `src/` in, so rebuild it after changing
  anything under `src/`. Install `yasp-agent` inside it as "Install" shows,
  and pass your Yasp API key into that container yourself; the check does
  not.

```bash
docker build -f docker/rtx5090-cuda130/Dockerfile -t kronos-edge:rtx5090-cuda130 .
```

A failed preflight lists every difference between your environment and the
entry, for example:

```
This Python environment does not match the requested environment. The differences are below; the Dockerfile named last is the recipe for the environment.
ValueError: torch: requested 2.14.1, found 2.9.1+cpu
cuda: requested 13.0, found a torch without CUDA
tensorrt: requested 11.4.0, not installed
gpu: requested NVIDIA GeForce RTX 5090, none available to torch
the recipe for environment rtx5090-cuda130 is docker/rtx5090-cuda130/Dockerfile
```

## Run the check

```bash
PYTHONPATH=src python -m kronos_edge.workflow path/to/config.yaml path/to/model.py path/to/input.npz
```

The model runs in a subprocess of the same interpreter (`python -m
kronos_edge.eager_check`), so whatever `model.py` does to the process cannot
touch the check's result, which travels in a report file.

On success it prints the benchmark, for example:

```
Eager run, ONNX export, ONNX simplification, TensorRT engine build (fp16), and benchmark succeeded.
Latency per run, eager: mean 0.047 ms, median 0.046 ms, p95 0.052 ms.
Latency per run, TensorRT: mean 0.015 ms, median 0.015 ms, p95 0.017 ms.
TensorRT is 3.0x faster than eager (median). Cosine similarity: 0.9998 (threshold 0.99).
The kernelgen optimization succeeded: the kernel is .artifact/kernel.py. Its report, as yasp-toolkit eval inspect shows it:
.artifact/kernelgen_report.json
  ...
  speedup: 2.000 (CI 2.000 – 2.010) ✓ converged
  correctness: passed (1 output, shapes ✓, dtypes ✓, values ✓)
  ...
```

The benchmark warms each side up with 20 runs, then times 200 runs one by
one and reports the mean, median, and 95th percentile of those latencies.
The speedup is the ratio of the medians. Eager runs the model as you wrote
it; TensorRT runs the engine at the configured precision. Each output of the
model, in the order they are returned (a dict's insertion order), is compared
with the engine's output at the same position by the configured metric, and
every value is printed. A model that changes its inputs in place is fine:
every stage runs it on copies of your inputs.

The ONNX export runs
`torch.onnx.export(..., dynamo=True)` on the model and inputs, `onnxsim`
simplifies the result, and TensorRT builds a serialized engine from the
simplified model at the configured precision. If the model uses an input's
values as a shape, for example `x.reshape(shape.tolist())`, the engine is
built for exactly the values in your `input.npz`. Engine builds take seconds
for small models and can take minutes for large ones; there is no timeout.

Once the engine is built, the check exports the kernelgen reference with the
documented
[`yasp-toolkit kernelgen export`](https://compile.yasp.ai/help/reference/kernelgen/#export)
on your `model.py`, run as a command with its output captured. It builds
`Model(*get_init_inputs())` (or `Model()`), calls `get_inputs()`, traces the
model with `torch.export` in inference mode, and writes `model.pt2`. It runs
with the torch of the environment `yasp-agent` is installed in, on whatever
device that torch and your model choose; the check forces neither the CPU nor
the GPU. Any package `model.py` imports must be importable there too. A
reference traced on the GPU has been rejected by the kernelgen service before
(`Torch not compiled with CUDA enabled`) while CPU references were accepted,
so a model that places tensors on the GPU inside `forward` may need a CPU
torch in the toolkit's environment.

The reference is not built from the model the benchmark ran or from your
`input.npz`: the documented export reads `model.py` alone, and no documented
Gaia command takes weights or input arrays today. Its weights are whatever
`Model(*get_init_inputs())` creates, the same as the benchmarked model only if
construction is deterministic (seed it in `__init__`, as the docs' example
model does); its example inputs are what `get_inputs()` returns, with the
shapes and dtypes `input.npz` matches but not its values. The TensorRT result
and the kernelgen result are separate comparisons with the model; neither is
derived from the other.

Then the check confirms your Gaia worker is registered (see "Deploy the Gaia
worker") and runs the documented generation, the
[reference alone](https://compile.yasp.ai/help/gaia/generate-your-first-kernel/#step-3-generate)
with your config's `gpu` as the target and its in-compile evaluations routed to
your worker:

```bash
yasp-toolkit kernelgen compile --reference .artifact/model.pt2 --gpu-name "<gpu>" \
    --output .artifact/kernel.py --report .artifact/kernelgen_report.json \
    --no-progress --eval-routing <eval_routing>
```

No `--source` is passed, so no Python of yours runs on the service's side.
Kernelgen jobs take minutes; the toolkit's progress goes to the log. When the
job completes, Gaia returns the kernel that measured fastest among those that
passed its worker's checks against the reference. The check prints the job's
report through
[`yasp-toolkit eval inspect --file`](https://compile.yasp.ai/help/reference/eval/#inspect),
speedup and correctness included, and reads none of its fields itself. If the
job fails, the detail shows the toolkit's output with the task id;
`yasp-toolkit kernelgen get <task id> --wait` shows its state.

## Artifact directory

The files the checks produce are written to `.artifact` in the current
directory. Pass `--artifacts path/to/dir` to use another directory. It is
created if missing. Each run first deletes the files listed below from a
previous run, and `kernelgen_eval.json`, which earlier versions of the check wrote,
and nothing else, so your own files in that directory are safe.
On a failure, the files produced before it are kept.

| File | Written by |
|---|---|
| `model.onnx` | The ONNX export. |
| `model_simplified.onnx` | The onnxsim simplification. |
| `model.engine` | The serialized TensorRT engine, at the configured precision. Specific to this GPU and TensorRT version. |
| `benchmark.json` | The latency statistics of both sides, the speedup, and the metric values, as printed. |
| `model.pt2` | The kernelgen reference `yasp-toolkit kernelgen export` wrote from `model.py`. |
| `kernel.py` | The optimized kernel kernelgen returned (its `module_source.py`). |
| `kernelgen_report.json` | The kernelgen job's report, as printed by `yasp-toolkit eval inspect`. |

When a model's weights exceed the 2 GB protobuf limit they are stored next to
the `.onnx` file in a companion `.data` file of the same name, for example
`model.onnx.data`. Keep the two files together.

and exits with code 0. On failure it prints a message naming the problem,
followed by the underlying exception, and exits with code 1. For example:

```
Your model cannot be run through eager mode. Either fix it or contact support.
AttributeError: 'Model' object has no attribute 'relo'
```

Log lines go to stderr, including the ones from the model-check subprocess. Pick
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
| This Python environment could not be inspected. Contact support. | The probe that imports torch and TensorRT in a subprocess failed before reporting; the detail shows its output. |
| This Python environment does not match the requested environment. The differences are below; the Dockerfile named last is the recipe for the environment. | Each line names a package or the GPU, what the environment needs, and what was found. Set the environment up as "Set up the environment" describes, or run the check in a container built from the Dockerfile. |
| The yasp-toolkit command is not available. Install yasp-agent as README.md describes. | Install `yasp-agent` with pipx as "Install" shows, so `yasp-toolkit` is on the path. |
| The model checks could not be run. Contact support. | The subprocess running the model could not be started, or exited without a usable report. |
| The GPU could not be accessed by torch when the model checks started. Contact support. | The preflight saw a GPU but the subprocess did not. |
| model.py could not be loaded. | Fix the syntax or import error shown below it, or check the path. |
| model.py imports a package that is not installed in this Python environment. Install it and run the check again. | The detail names the module `model.py` could not import. Install the pip package that provides it (for `PIL`, that is `pillow`) into this environment. |
| model.py does not define a 'Model' class. | Add a `Model` class that subclasses `torch.nn.Module`. |
| model.py does not define get_inputs() or get_fwd_args(). | Add one of them, returning the reference inputs. |
| Model could not be constructed. | Add `get_init_inputs()` returning the constructor arguments named in the detail line. |
| get_inputs() raised an exception. | Fix `get_inputs()`. |
| input.npz could not be read. | Check the path, and that the file was written with `numpy.savez`. |
| input.npz does not match the inputs returned by get_inputs(). | The detail line names the key and the expected versus found shape and dtype. |
| Your model cannot be run through eager mode. Either fix it or contact support. | The forward pass raised. The detail line shows the exception. |
| The ONNX exporter is not available in this Python environment. Contact support. | The environment lacks `onnx` or `onnxscript`. |
| Your model could not be cast to the requested precision for export. Either fix it or contact support. | Casting a copy of the model to half raised; the detail shows the exception. |
| Your model could not be traced by torch.export for ONNX export. Fix the construct named below. | `torch.export` could not capture the model, for example a branch on a tensor value. The detail names the construct and the line in `model.py`. |
| Your model was traced but an operator has no ONNX translation. Replace the operator named below or contact support. | The detail names the operator the dynamo exporter cannot translate. |
| The exported ONNX model could not be serialized or written to the artifact directory. | Check the directory is writable and the machine has memory for the model; the detail shows the error. |
| The ONNX simplifier is not available in this Python environment. Contact support. | The environment lacks `onnxsim`. |
| The ONNX model could not be simplified by onnxsim. Either fix the model or contact support. | onnxsim raised. The detail shows the exception. |
| onnxsim produced a simplified model that does not pass its check or the ONNX checker. Contact support. | Not something you can fix from the model. |
| The simplified ONNX model could not be written to the artifact directory. | Check the directory is writable; the detail shows the OS error. |
| TensorRT is not available in this Python environment. Contact support. | The environment lacks `tensorrt`. |
| TensorRT could not parse the ONNX model. Replace the operator or node named below or contact support. | The detail lists the parser's errors, typically an operator TensorRT has no importer for. |
| TensorRT could not build an engine from the model. The errors it reported are below. | The detail lists what TensorRT logged, for example an unsupported layer configuration or running out of GPU memory. |
| The TensorRT engine could not be written to the artifact directory. | Check the directory is writable; the detail shows the OS error. |
| Your model failed while being run repeatedly in eager mode for timing. Either fix it or contact support. | The forward pass passed once but raised when repeated; the detail shows the exception. |
| The TensorRT runtime is not available in this Python environment. Contact support. | The environment lacks `tensorrt`. |
| The TensorRT engine could not be loaded or run. The errors it reported are below. | The detail lists what TensorRT logged. |
| The TensorRT engine's outputs do not have the same structure as the eager outputs. Contact support. | The count or a shape differs; the detail names the output. |
| The TensorRT engine's outputs do not match the eager outputs within the threshold in config.yaml. | The detail names each output below the threshold with its value. Lower the threshold if the gap is acceptable for your model, or use `fp32`. |
| yasp-toolkit kernelgen export could not export model.py as the kernelgen reference. The exporter's output is below. | The documented exporter could not import, trace, or save your model, for example a package missing from the toolkit's environment or a branch on a tensor value. If you cannot fix it, send the output (and `model.py` if you can share it) to support@yasp.ai, as the docs ask. |
| The kernelgen reference (model.pt2) is missing from the artifact directory. Contact support. | The export reported success without writing the reference. |
| The TensorRT engine (model.engine) kernelgen waits for is missing from the artifact directory. Contact support. | The model check reported success without writing the engine. |
| The evaluators registered for the kernelgen evaluation could not be listed. The toolkit's output is below. | The service could not be reached or rejected the request; the detail shows the toolkit's output. |
| No Gaia worker advertising your config's gpu is registered under its eval_routing. Deploy one on this machine as README.md describes; the check never starts one. | Deploy the worker as "Deploy the Gaia worker" shows, with `--tag routing=<eval_routing>`, and check that its GPU Name in `yasp-toolkit eval workers` equals your config's `gpu`. |
| The kernelgen optimization did not produce a kernel. The toolkit's output is below; `yasp-toolkit kernelgen get <task id> --wait` shows the job's state. | The kernelgen job failed, the service could not be reached, or the key was missing or rejected; the detail shows the toolkit's output, including the task id. |
| The kernelgen kernel was written, but its report could not be shown with yasp-toolkit eval inspect. The toolkit's output is below. | `kernel.py` and `kernelgen_report.json` are in the artifact directory; the detail shows why the toolkit could not render the report. |

## Run the tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest
```

`pyproject.toml` points pytest at `src/`, so no install of the package is
needed. The tests write their model and input files to a temporary directory;
nothing binary is checked in. They need neither a GPU, TensorRT, nor
`yasp-toolkit`: the preflight probe, the toolkit, and the engine are stubbed,
and the model-check graph runs in-process on the CPU.

See `architecture.md` for how the pieces fit together.

# kronos-edge-v2

Checks whether a PyTorch model runs in eager mode with the inputs you supply,
in an environment we support. You give it a `config.yaml` naming the GPU and
software versions you want, a `model.py`, and an `input.npz`. It matches the
config to a supported environment, then runs the check inside that
environment's Docker container on the GPU: it loads the model, validates the
inputs against the model's own reference inputs, runs one forward pass,
exports the model with `torch.export` as the reference for kernelgen, exports
it to ONNX with the dynamo exporter, simplifies the ONNX with onnxsim, builds
a TensorRT engine at the precision you ask for, benchmarks the engine against
eager mode, and checks their outputs agree. Back on this machine it then makes
sure an evaluator for your GPU and routing key is registered, starting one on
this machine when none is, asks yasp kernelgen for an optimized kernel of the
model, scores that kernel on that evaluator, and prints either the benchmark
result and the kernelgen speedup or a message saying what to fix. The reference, the ONNX
files, the engine, the benchmark numbers, the kernel, and the evaluation
reports land in an artifact directory.

## Requirements

- Python 3.12 or newer
- The packages in `requirements.txt` (`langgraph`, `numpy`, `onnx`, `onnxscript`, `onnxsim`, `pydantic`, `pyyaml`, `torch`, `yasp-agent[gaia]`)
- Docker with the NVIDIA container runtime, and `nvidia-smi` on the path
- A Yasp API key for the kernelgen steps (see below)
- Access to `docker.yasp.ai` and to the local Docker socket
  (`/var/run/docker.sock`), for the Gaia worker the check starts when no
  evaluator is registered (see below). The worker image carries the CUDA
  build toolchain; nothing beyond Docker is needed on this machine for it.

## Install

`yasp-agent[gaia]` is published on the yasp package index, not on PyPI, and
that index is read with a registry key: a scoped, long-lived key
(`nexus-read`) exchanged from your Yasp API key as the
[install guide](https://compile.yasp.ai/help/install/) describes (its
exchange needs no package installed), or, once the toolkit is installed,
printed by `yasp-toolkit registry key`. Put it in `REGISTRY_KEY` and hand it
to pip through a netrc file, never on the command line, where it would show
in process listings while pip runs. The file lives in a private temporary
path and is removed when the subshell ends, success or failure:

```bash
(
  set -e; umask 077
  NETRC_FILE=$(mktemp); trap 'rm -f "$NETRC_FILE"' EXIT
  printf 'machine nexus.yasp.ai\nlogin token\npassword %s\n' "$REGISTRY_KEY" > "$NETRC_FILE"
  python -m venv .venv
  NETRC="$NETRC_FILE" .venv/bin/pip install -r requirements.txt \
      --extra-index-url https://nexus.yasp.ai/repository/pypi-hosted/simple/
)
```

`NETRC` names the file pip reads credentials from, so the index URL stays
credential-free. The same key builds the environment image (see "Build the
environment image"). The index installs the `yasp-toolkit` command the kernelgen
steps run. The evaluator is not a package to install: it runs inside the
official Gaia worker image (see "Set up your Yasp API key").

The package lives under `src/`, so set `PYTHONPATH=src` when running it from
a checkout. The examples below do that.

## Set up your Yasp API key

`yasp-toolkit` needs a Yasp API key. Either export it in the shell that runs
the check, or store it in the OS keyring once:

```bash
export YASP_API_KEY=...            # the variable yasp-toolkit reads
.venv/bin/yasp-toolkit keyring set  # or: keep it in the keyring instead
```

`.env.example` documents the variable. Nothing in the check reads a `.env`
file: if you keep the key in one, load it yourself (`set -a; . ./.env; set +a`)
and never commit it. `.env` is gitignored.

The kernelgen evaluation runs on an evaluator registered under the routing
key of your config (`eval_routing`, below) and advertising your config's
`gpu`. Before kernelgen, the check lists the evaluators registered under that
key (`yasp-toolkit eval workers`). If one advertises your `gpu`, it is used
and left as it is. Otherwise the check starts a Gaia worker on this machine
the way the [Gaia Docker worker guide](https://compile.yasp.ai/help/gaia/workers/docker/)
does: it mints a registry key and a registration key from your API key
(`yasp-toolkit registry key`, `yasp-toolkit eval key`), pulls the newest
release of the official NVIDIA worker image from `docker.yasp.ai` under a
temporary Docker login that leaves your own Docker credentials untouched,
and starts a container named `kronos-eval-<unique id>` running
`yasp-eval register --tag routing=<key>`. `yasp-eval` here is the
executable inside that image, not something installed on this machine: no
evaluator package and no CUDA toolkit are needed on the host. The worker
gets the NVIDIA runtime, the Docker socket, and the dataset cache the guide
lists, so it can run its per-task evaluation containers, and it detects its
own GPU. Only the registration key, scoped to registering, enters the
worker; your API key stays on the host and never enters it or the
model-check container, and no key is written to a log or a command line.
The check waits up to two minutes for the worker to be listed and stops
that one container when the check ends, whether it succeeded, failed, or
was interrupted. The worker is first asked to stop (`docker stop`, up to
30 s) so it cancels its running evaluation, then removed. As started here,
the worker runs each evaluation inside its own container, so the
evaluation ends with it. A worker whose launcher is configured for its
Docker backend runs each evaluation instead in a sibling container under
the shared slot name `yasp-launcher-slot` and tears it down on cancel; the
check then inspects the slot and removes the sibling only if it was this
worker's, which it tells from the `KRONOS_EVALUATOR` marker the worker
forwards into its tasks, and only by the id that inspect returned, so a
sibling started by another worker on this machine is never touched, even
one that takes the name meanwhile. The pulled image and the
`yasp-dataset-cache` volume stay for the next run.
Unlike the guide's deployment, this worker has no restart policy, health
check, or autoheal sidecar: it lives for one check. Its output is kept out
of the check's output and shown only if it exits before registering. The
routing key is required, so the evaluation is always pinned to that
evaluator.

## Prepare your config file

`config.yaml` names the environment you want, the settings for the checks,
the evaluator for the kernelgen kernel, and the packages your model needs.
The first nine fields are required, `dependencies` is optional, and no other
fields are allowed. Version numbers must be quoted so YAML does not read them
as numbers; `threshold` is a number.

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
| `dependencies` | list of pip requirements | `[timm, einops==0.8.0]` |

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
integer inputs are left alone. The eager run and the kernelgen reference
(`model.pt2`) always use the model as you wrote it; the reference is traced
from CPU copies of that uncast model and your inputs (see "Run the check").
TensorRT builds a
strongly typed engine, so the engine computes in exactly the types the ONNX
carries. Changing `precision` and running again rewrites all the files in the
same artifact directory.

`eval_routing` is also a setting. It is the routing key the kernelgen
evaluations are pinned to: kernelgen's in-compile evaluations and the final
evaluation go to the evaluator registered under it that advertises your
`gpu`, which the check starts on this machine when there is none. Pick a key
of your own, so a job never lands on someone else's evaluator. It is required:
without it any evaluator advertising your config's `gpu` could take the job,
and the check must evaluate on this machine, so a config that omits it is
rejected before any job is submitted.

`dependencies` lists the packages `model.py` imports that the environment
image does not carry, as pip requirements (`timm`, `einops==0.8.0`). They
are installed inside the container at the start of every run, before
`model.py` is imported, and are never baked into the image; the image resolves
them against the same package index as its torch build, so a package with
CUDA builds matches the environment. The install is pinned to the versions the
image already carries, so a requirement that would replace torch or another
package of the environment, whether listed here or pulled in by a listed
package, fails the install instead of changing the environment the checks were
validated in. Leave the field out when your model needs
only torch. If `model.py` imports a package that is not installed, the check
stops before anything runs and names the module, so add the package that
provides it here. The install adds to each run's time.

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
machine. The build installs `yasp-compile-common`, the package holding
yasp's exporter, from the yasp package index named in "Install", which needs
a registry key to read. The key goes to the build as a BuildKit secret: a
netrc file the build mounts for that one `pip install` and nothing else, so
it is never a build argument, an environment variable, part of a URL, or a
layer of the image. Write the file with your registry key (`yasp-toolkit
registry key` prints one; the install guide's exchange works before the
toolkit exists) in a private temporary path, build, and let the subshell
remove it, success or failure:

```bash
(
  set -e; umask 077
  NETRC_FILE=$(mktemp); trap 'rm -f "$NETRC_FILE"' EXIT
  printf 'machine nexus.yasp.ai\nlogin token\npassword %s\n' "$(yasp-toolkit registry key)" > "$NETRC_FILE"
  docker build --secret id=yasp-netrc,src="$NETRC_FILE" \
      -f docker/rtx5090-cuda130/Dockerfile -t kronos-edge:rtx5090-cuda130 .
)
```

The build fails without the secret, and the check's "image is not built"
message repeats this command with the secret flag. Running the check needs
no key inside the container: the exporter it installs runs without one.
Build the image once, and rebuild it after changing anything under `src/`.

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
The kernelgen optimization and kernelgen evaluation (speedup 2.00x) succeeded.
```

The benchmark warms each side up with 20 runs, then times 200 runs one by
one and reports the mean, median, and 95th percentile of those latencies.
The speedup is the ratio of the medians. Eager runs the model as you wrote
it; TensorRT runs the engine at the configured precision. Each output of the
model, in the order they are returned (a dict's insertion order), is compared
with the engine's output at the same position by the configured metric, and
every value is printed. A model that changes its inputs in place is fine:
every stage runs it on copies of your inputs.

Right after the forward pass, yasp's own exporter (`yasp.compile.common.exporter`,
the functions `yasp-toolkit kernelgen export` runs, installed in the image
and needing no key) traces independent CPU copies of the same model and
inputs, uncast, into `model.pt2`, the reference kernelgen optimizes. The
copies matter: the archive records the device the trace ran on
in every weight, and a torch built without CUDA cannot load an archive traced
on the GPU (`Torch not compiled with CUDA enabled`). That is the error the
kernelgen service reported for a GPU-traced reference, before any kernel was
attempted, so the service is taken to load references on such a build; its
internals are not observable from here. Tracing on CPU removes that loading
failure, which is what was demonstrated: CPU-traced references passed the
service's loading stage. It does not by itself make kernelgen succeed, which
still depends on its candidates passing the evaluator's checks, and a model
that itself places tensors on the GPU inside `forward` still records CUDA in
the trace. The eager model and your inputs stay on the GPU for the steps that
follow. The ONNX export then runs
`torch.onnx.export(..., dynamo=True)` on the same model and inputs, `onnxsim`
simplifies the result, and TensorRT builds a serialized engine from the
simplified model at the configured precision. If the model uses an input's
values as a shape, for example `x.reshape(shape.tolist())`, the engine is
built for exactly the values in your `input.npz`. Engine builds take seconds
for small models and can take minutes for large ones; there is no timeout.

Once the container has produced the reference and the engine, the check makes
sure an evaluator for your `gpu` and `eval_routing` is registered, starting
a Gaia worker container on this machine when none is (see "Set up your Yasp API
key"). It then runs
`yasp-toolkit kernelgen compile` on this machine with your `model.py` as the
source and your config's `gpu` as the target, and waits for the kernelgen
job; the optimized kernel is written as `kernel.py`. It then runs
`yasp-toolkit eval compare` to score that kernel against the reference on
this machine's evaluator. The kernel must pass the evaluator's correctness
checks; the speedup the evaluator measured is printed, and no expectation is
derived from the TensorRT engine or the eager run. Kernelgen jobs take
minutes; the toolkit's progress goes to the log. A Gaia worker the check
started is removed once the evaluation is over, or as soon as a step fails.

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
| `model.pt2` | The reference of your model, exported by yasp's exporter on CPU from an uncast copy, as kernelgen's `reference.pt2`. |
| `kernel.py` | The optimized kernel kernelgen returned (its `module_source.py`). |
| `kernelgen_report.json` | The kernelgen job's report. |
| `kernelgen_eval.json` | The evaluator's results for `kernel.py`: correctness checks and speedup. |

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
| The Docker image for the requested environment is not built. | Run the `docker build` command shown in the detail line, with the registry-key secret file "Build the environment image" describes. |
| The check could not be run in the environment container. Contact support. | `docker run` failed before the check could report. |
| The GPU could not be accessed inside the environment container. Contact support. | The container started but torch could not see the GPU. |
| A package listed under dependencies in config.yaml could not be installed, or would replace a package the environment image carries. pip's output is below. | Check the requirement's spelling and version against the index, or, if pip reports a conflict with a constraint, pick a version of the package that works with the environment's torch; the detail shows what pip printed. |
| model.py could not be loaded. | Fix the syntax or import error shown below it, or check the path. |
| model.py imports a package that is not installed in the environment. Add the package that provides it to dependencies in config.yaml. | The detail names the module `model.py` could not import. Add the pip package that provides it (for `PIL`, that is `pillow`) to `dependencies`. |
| model.py does not define a 'Model' class. | Add a `Model` class that subclasses `torch.nn.Module`. |
| model.py does not define get_inputs() or get_fwd_args(). | Add one of them, returning the reference inputs. |
| Model could not be constructed. | Add `get_init_inputs()` returning the constructor arguments named in the detail line. |
| get_inputs() raised an exception. | Fix `get_inputs()`. |
| input.npz could not be read. | Check the path, and that the file was written with `numpy.savez`. |
| input.npz does not match the inputs returned by get_inputs(). | The detail line names the key and the expected versus found shape and dtype. |
| Your model cannot be run through eager mode. Either fix it or contact support. | The forward pass raised. The detail line shows the exception. |
| The kernelgen exporter is not available inside the environment container. Contact support. | The image lacks `yasp-compile-common`. |
| Your model could not be traced by torch.export for the kernelgen reference. Fix the construct named below. | `torch.export` could not capture the model, for example a branch on a tensor value. The detail names the construct. |
| The kernelgen reference (model.pt2) could not be written to the artifact directory. | Check the directory is writable; the detail shows the OS error. |
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
| The kernelgen reference (model.pt2) is missing from the artifact directory. Contact support. | The container reported success without writing the reference. |
| The TensorRT engine (model.engine) kernelgen waits for is missing from the artifact directory. Contact support. | The container reported success without writing the engine. |
| The yasp-toolkit command is not available. Install yasp-agent[gaia] into the environment that runs the check. | Install the packages in `requirements.txt` with the yasp index. |
| No Yasp API key was found, or the key was not accepted. Export a valid YASP_API_KEY (see .env.example) or store one with `yasp-toolkit keyring set`. | See "Set up your Yasp API key". If a key is set, the service rejected it: create a new one. |
| The evaluators registered for the kernelgen evaluation could not be listed. The toolkit's output is below. | The service could not be reached or rejected the request; the detail shows the toolkit's output. |
| Docker is not available for starting the Gaia worker: the docker command or /var/run/docker.sock was not found. Install Docker with the NVIDIA container runtime on this machine. | The worker is a container with the NVIDIA runtime and access to `/var/run/docker.sock`. |
| No evaluator for your config's gpu and eval_routing is registered, and the Gaia worker could not be started on this machine or exited before registering. The detail below names the step that failed or shows the worker's output. | The detail is the Docker or toolkit command that failed with what it printed (registry access, the NVIDIA runtime, the socket's permissions), or the worker's own output when it exited. |
| No evaluator for your config's gpu and eval_routing is registered, and the Gaia worker started on this machine did not register in time. Check the gpu it advertises and that the service is reachable. | The worker ran for two minutes without being listed for your `gpu` and `eval_routing`. The detail names both and the container. |
| The Gaia worker container this check started could not be removed. Remove the kronos-eval container named below yourself. | `docker rm -f <container>` for the container the detail names (the worker by name, or its `yasp-launcher-slot` task container by id); evaluators the check did not start are never touched. |
| The kernelgen optimization did not produce a kernel. The toolkit's output is below. | The kernelgen job failed or the service could not be reached; the detail shows the toolkit's output, including the task id. |
| The kernelgen kernel could not be evaluated on this machine. The toolkit's output is below. | The evaluation task failed, or the evaluator dropped out after the check found it registered; the detail shows the toolkit's output. |
| The evaluator's results for the kernelgen kernel could not be read. Contact support. | Not something you can fix from the model. |
| The kernelgen kernel failed the evaluation on this machine: it did not build or did not match your model. The evaluator's report is below. Contact support. | The detail names the failed checks and what the evaluator captured. |

## Run the tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest
```

`pyproject.toml` points pytest at `src/`, so no install of the package is
needed. The tests write their model and input files to a temporary directory;
nothing binary is checked in.

See `architecture.md` for how the pieces fit together.

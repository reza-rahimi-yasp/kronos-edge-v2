# AGENTS.md

Decisions that govern this repository. `README.md` has the user contract and
the commands; `architecture.md` has the data flow. Keep this file to
decisions: do not add file listings, commands, or summaries here.

## Structure

- Two LangGraph graphs: the host graph validates config and environment, the
  container graph runs the model. The host never runs user code.
- The host must import without torch or numpy. Only the container graph's
  modules (`eager_check`, `onnx_check`, `onnxsim_check`, `tensorrt_check`,
  `benchmark_check`) import the model stack. A test
  enforces it.
- Gaia is used only through the commands documented at
  https://compile.yasp.ai/help/, following its "Generate your first kernel"
  workflow (runner, export, generate, inspect). The repository orchestrates
  them and implements no part of Gaia: no `yasp-compile-*` package is ever a
  dependency or an import, no exporter or evaluator is replicated, no service
  is called except through `yasp-toolkit`, and no field of a Yasp report is
  parsed. `yasp-agent` is the only yasp package installed directly; what it
  pulls in is its own.
- Kernelgen runs on the host, after the containers, as `yasp-toolkit`
  subprocesses (`kernelgen compile`, then `eval inspect`): the toolkit needs
  the user's Yasp API key, which never enters a container, and the host never
  imports yasp-agent, which is installed with pipx as the docs show, not from
  `requirements.txt`. The `kernelgen` node requires the export's `model.pt2`
  and the container's `model.engine` before it starts. The toolkit can exit 0
  after an early error, so a node trusts the files it asked for, not the exit
  code.
- The kernelgen reference is produced by the documented `yasp-toolkit
  kernelgen export`, run on the user's `model.py` in a container from the
  export image (`docker/gaia-export/Dockerfile`), because it runs user code
  and the host must not. That image is the docs' install: CPU-only
  `torch==2.9.1`, then `yasp-agent`, nothing else. Its build reads the yasp
  package index with a scoped registry key taken as a BuildKit secret (a
  netrc file mounted for that step alone), never as a build argument,
  environment variable, URL, or layer. The export runs with no GPU and no
  key (observed to need neither); the config's `dependencies` are installed
  there first under the image's constraints. The CPU-only torch traces the
  reference on the CPU: a GPU-traced reference failed on the kernelgen service
  with `Torch not compiled with CUDA enabled`, the error a torch built without
  CUDA raises when loading it, while CPU references were accepted; the
  service's build is inferred from that, not observed. The export reads
  `model.py` alone (`Model(*get_init_inputs())`, `get_inputs()`): the
  reference does not carry the benchmarked instance or `input.npz`, since no
  documented command takes weights or input arrays, and the README says so.
  The model-check image carries no yasp package.
- The kernelgen evaluation runs on the user's Gaia worker on this machine:
  `kernelgen compile --eval-routing` routes its in-compile evaluations to the
  pool the worker joined with `--tag routing=<eval_routing>`. The setting is
  required, so a config without it fails validation before any job is
  submitted; a worker with no routing tag joins the shared pool, where any
  job could take it. The reference alone is uploaded (no `--source`), so no
  user Python runs on the service's side. Gaia returns only a kernel that
  passed its worker's checks; the check shows the job's report through `eval
  inspect` and judges nothing itself. No expectation is derived from TensorRT
  versus the eager run.
- The Gaia worker is the user's persistent deployment, made once as the Gaia
  Docker guide documents, with a routing tag added; it stays up and serves
  every check. A job routed to a pool with no worker waits forever, so before
  kernelgen the host lists `yasp-toolkit eval workers --tag
  routing=<eval_routing>`, bounded, and requires a row whose GPU Name and
  Routing equal the config's `gpu` and `eval_routing` exactly (the listing's
  table is parsed; the toolkit has no machine-readable output). The check
  never starts, stops, or removes a worker and mints no key.
- `config.yaml` has two kinds of fields. The five environment fields must
  equal a registry entry and select the image. Settings (`precision`,
  `metric`, `threshold`) are validated by the same schema but never take part
  in matching; changing a setting re-runs the checks in the same environment
  and overwrites the artifacts. `config.Config` is the environment schema plus
  the settings. A metric is accepted only once it is implemented; the
  threshold is the user's, never a constant in code.
- Packages `model.py` needs beyond the image are the user's to declare, as
  pip requirements under `dependencies` in `config.yaml` (a setting, optional).
  The container pip-installs them into its venv at the start of every run,
  before `model.py` is imported; they are never added to a Dockerfile or
  committed to an image, so the image stays the registry entry's versions and
  nothing else. The image sets `PIP_EXTRA_INDEX_URL` to its torch index so
  CUDA-built packages resolve to the image's CUDA. The install is constrained
  to every distribution already in the venv, torch's CUDA build included, so
  a requirement, direct or transitive, that would replace an image package is
  a resolver conflict reported at `install_dependencies` and nothing is
  installed: the environment the checks run in stays the one that was
  validated. A `ModuleNotFoundError`
  while importing `model.py` is its own stage, `missing_dependency`, naming
  the module, so the user learns which package to declare.
- Precision is applied once, at the ONNX export, by casting a copy of the
  model and its float inputs to the configured dtype for both `fp32` and
  `fp16`, whatever the user wrote the model in. TensorRT 11 has no precision
  builder flags, so the network is strongly typed and the engine computes in
  the ONNX's types. The eager model in state is never cast.
- The engine is built for the user's inputs: shape inputs and dynamic
  dimensions get an optimization profile pinned to the values and shapes in
  `input.npz`. Nothing is generalized beyond what the user supplied.
- TensorRT exists only in the image. The real work is one module-level
  function per node (`tensorrt_check.build`, `benchmark_check.load_engine`)
  that an autouse fixture stubs; tests cover the node around it and the
  import failure, and the real paths are verified by a GPU run before each
  merge. The engine runner keeps the TensorRT `Runtime` alive for as long as
  the engine and reads shape-tensor inputs from host memory; both are
  segfaults otherwise. An output whose shape is still unknown after shape
  inference (data-dependent, such as `nonzero`) gets a TensorRT output
  allocator rather than a preallocated buffer.
- The user's inputs in state are never mutated: every node that runs the
  model runs it on copies, since a model may update its inputs in place, and
  the benchmark compares eager and the engine on the original values.
- Model outputs are flattened with pytree, which is what the exporter does,
  so tuples, lists, and dicts (insertion order) pair up with the ONNX outputs
  by position. A non-finite metric value fails the accuracy check; the host
  rejects a report carrying one.
- Benchmarks report a distribution, never a single number: warm up, time each
  run, report mean, median, and p95 in ms for both sides, speedup as the
  ratio of medians. Eager is timed as the user wrote the model; the engine at
  the configured precision. Outputs are paired by position and every pair
  must reach the threshold.
- The container's success report must carry the benchmark; the host refuses
  a success without one.
- Customer-facing text lives only in `messages.py`, one table per check,
  keyed by stage. Stage names are one flat namespace across all checks;
  `messages.merge()` and a test reject duplicates, and every `stage = "..."`
  in a check must have an entry.
- A registry entry in `environments.yaml`, its Dockerfile under `docker/`,
  and the image labels state the same versions. Change all three together.
- The container talks to the host only through `report.json` in the mounted
  report directory. Never through stdout: the user's model owns stdout. Logs
  go to stderr, which the host inherits.
- Files produced for the user (`.onnx`, later `.engine`) go only to the
  artifact directory: `.artifact` in the working directory unless
  `--artifacts` says otherwise, created by the host and mounted into the
  container. Every such file is listed in `artifacts.py`; the host deletes
  those names, and only those, before each run so a stale file never sits
  next to a fresh one. A name an earlier version wrote stays listed there as
  retired, deleted but never written, so a carried-over directory is cleaned
  too. Each node writes its own file as soon as it has it, so
  a later failure keeps the earlier files. Intermediate models also stay in
  state in memory for the next node.
- Model-stack packages (`onnx`, `onnxsim`, and the exporter's lazy imports)
  are imported inside the node that needs them, never at module top, so a
  missing package in the image reaches the report as a stage instead of
  crashing the container before it can write the report.
- Models above the 2 GB protobuf limit are saved with external data and
  checked by path. The onnxsim step is unverified above that size.
- `yasp-agent` comes from the yasp package index, not PyPI, read with a
  registry key that reaches pip through a temporary netrc file, never a
  command line; the `README.md` install command names the index.
  Credentials are documented in `.env.example` only; nothing reads a `.env`
  file.

## Node conventions

- Every node wraps its body in one `try/except Exception` and returns
  `fail(logger, stage, exc)` on error. Exceptions never escape a node.
- Failure details travel in state (`error`, `error_detail`), never in a shared
  log buffer. Logs are for operators; state is for the workflow.
- Log through `kronos_edge.log.get_logger` and tag records with
  `extra={"stage": ...}`. A module that can run under `python -m` names its
  logger explicitly, because `__name__` becomes `__main__` there.
- Anything that shells out (`docker`, `nvidia-smi`, `yasp-toolkit`) is a
  module-level function so tests can replace it with `monkeypatch`.

## Testing

- Tests come first: add the scenario to the parametrized table, see it fail,
  then implement.
- One test file per check. Fixtures are source strings written to
  `tmp_path`; nothing binary is committed. Tests never shell out: autouse
  fixtures replace the `docker` and `nvidia-smi` probes and `yasp-toolkit`,
  and run the container graph in-process on CPU. No test needs a Yasp key.
- A gitignored `.venv` is expected at the repo root. No lockfiles.

## Working rules

- Work only inside this repository. Do not read, reference, or borrow code
  from any other project on this machine, and do not assume anything about
  the project beyond what is in it.
- Implement the minimum change that satisfies the request. No extra features,
  refactors, abstractions, or "while I'm here" cleanups. Do not add
  dependencies, tooling, or scaffolding unless the request requires it.
- Prefer editing existing files over creating new ones. Match the existing
  style of the code you touch.
- If the request is ambiguous, ask a concise clarifying question before
  writing code. State any small, low-risk assumption explicitly.
- Rebuild the image after changing anything under `src/`; it is copied in.
- Branch off `main`, one PR per concern, commits that each do one thing.
- Before finishing: run the tests and report the result verbatim, verify the
  change does what was asked and nothing more, update `README.md`,
  `architecture.md`, or this file when the change alters the user contract,
  the data flow, or a decision, and report what was changed, what was not
  done, and anything that still needs the user's input.

# AGENTS.md

Decisions that govern this repository. `README.md` has the user contract and
the commands; `architecture.md` has the data flow. Keep this file to
decisions: do not add file listings, commands, or summaries here.

## Structure

- One local workflow, two LangGraph graphs: the workflow graph validates the
  config and the environment, the model-check graph runs the model as a
  subprocess of the same interpreter. The workflow process never runs user
  code in-process and never imports the model stack. The same workflow serves
  a native install and a container the user built from an environment's
  Dockerfile: it verifies whichever Python environment it was started in and
  does not ask which kind it is.
- The workflow must import without torch or numpy. Only the model-check
  graph's modules (`eager_check`, `onnx_check`, `onnxsim_check`,
  `tensorrt_check`, `benchmark_check`) import the model stack, and the
  preflight's probe does so only in its subprocess. A test enforces it.
- Dockerfiles under `docker/` are recipes for the registry's environments and
  validation fixtures: a test compares each entry with its Dockerfile's
  labels. Nothing under `src/` runs `docker` or checks for an image, and no
  image is required at run time; a test enforces that too. There is no
  Docker detection and no driver-version gate.
- The preflight runs before anything else touches the model: a probe
  subprocess reports torch's version and CUDA build, TensorRT's version, and
  the GPU torch sees, compared with the matched entry by version component
  (`11.4.0` accepts `11.4.0.106`, `2.14.1` accepts `2.14.1+cu130`, the GPU
  name must be equal, the driver is not checked), every difference reported
  at once with the entry's Dockerfile as the recipe. It also requires
  `yasp-toolkit` on the path. The config's `driver` field still identifies the
  registry entry; it is documentation of the environment the entry was
  validated with, not a check on this machine.
- Gaia is used only through the commands documented at
  https://compile.yasp.ai/help/, following its "Generate your first kernel"
  workflow (runner, export, generate, inspect). The repository orchestrates
  them and implements no part of Gaia: no `yasp-compile-*` package is ever a
  dependency or an import, no exporter or evaluator is replicated, no service
  is called except through `yasp-toolkit`, and no field of a Yasp report is
  parsed. `yasp-agent` is the only yasp package installed directly, with
  pipx as the docs show, never from `requirements.txt`; what it pulls in is
  its own. It and `yasp-toolkit` are run-time requirements of the check;
  deeper integration than running its commands is deferred.
- Every `yasp-toolkit` command (`kernelgen export`, `eval workers`,
  `kernelgen compile`, `eval inspect`) is a subprocess with its output
  captured, after the model check. The toolkit needs the user's Yasp API key,
  which only it reads. The processes that run the user's model, the
  model-check subprocess and `kernelgen export`, get this environment
  without `YASP_*` and `REGISTRY_KEY` and with Python's keyring set to its
  null backend; the export needs no key. The service commands (`eval
  workers`, `kernelgen compile`, `eval inspect`) keep the full environment
  they authenticate with. This withholds the credentials, it is not a
  sandbox: the model still runs as the user. The `kernelgen` node requires
  the export's `model.pt2` before it starts, and not the TensorRT engine. The
  toolkit can exit
  0 after an early error, so a node trusts the files it asked for, not the
  exit code.
- The kernelgen reference is produced by the documented `yasp-toolkit
  kernelgen export`, run on the user's `model.py` with the toolkit's own
  torch, on whatever device that torch and the model choose: the check forces
  neither the CPU nor the GPU. (Earlier, a GPU-traced reference once failed on
  the kernelgen service with `Torch not compiled with CUDA enabled` while CPU
  references were accepted; that is a known risk of the user's setup, not
  something the check enforces.) The export reads `model.py` alone
  (`Model(*get_init_inputs())`, `get_inputs()`): the reference does not carry
  the benchmarked instance or `input.npz`, since no documented command takes
  weights or input arrays, and the README says so.
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
  kernelgen the workflow lists `yasp-toolkit eval workers --tag
  routing=<eval_routing>`, bounded, and requires a row whose GPU Name and
  Routing equal the config's `gpu` and `eval_routing` exactly (the listing's
  table is parsed; the toolkit has no machine-readable output). The check
  never starts, stops, or removes a worker and mints no key.
- `config.yaml` has two kinds of fields. The five environment fields must
  equal a registry entry, which the preflight then verifies against this
  environment. Settings (`precision`, `metric`, `threshold`, `eval_routing`)
  are validated by the same schema but never take part in matching; changing
  a setting re-runs the checks in the same environment into a new run
  directory. `config.Config` is the environment schema plus the settings. A
  metric is accepted only once it is implemented; the threshold is the
  user's, never a constant in code.
- Packages `model.py` imports are the user's to install into the environment
  the check runs in. The check has no dependency setting and installs
  nothing, so the environment the checks run in stays the one the preflight
  verified. A `ModuleNotFoundError` while importing `model.py` is its own
  stage, `missing_dependency`, naming the module, so the user learns what to
  install.
- Precision is applied once, at the ONNX export, by casting a copy of the
  model and its float inputs to the configured dtype for both `fp32` and
  `fp16`, whatever the user wrote the model in. TensorRT 11 has no precision
  builder flags, so the network is strongly typed and the engine computes in
  the ONNX's types. The eager model in state is never cast.
- The engine is built for the user's inputs: shape inputs and dynamic
  dimensions get an optimization profile pinned to the values and shapes in
  `input.npz`. Nothing is generalized beyond what the user supplied.
- TensorRT needs the GPU and is absent from the test environment. The real
  work is one module-level function per node (`tensorrt_check.build`,
  `benchmark_check.load_engine`) that an autouse fixture stubs; tests cover
  the node around it and the import failure, and the real paths are verified
  by a GPU run before each merge. The engine runner keeps the TensorRT
  `Runtime` alive for as long as the engine and reads shape-tensor inputs
  from host memory; both are segfaults otherwise. An output whose shape is
  still unknown after shape inference (data-dependent, such as `nonzero`)
  gets a TensorRT output allocator rather than a preallocated buffer.
- The user's inputs in state are never mutated: every node that runs the
  model runs it on copies, since a model may update its inputs in place, and
  the benchmark compares eager and the engine on the original values.
- Model outputs are flattened with pytree, which is what the exporter does,
  so tuples, lists, and dicts (insertion order) pair up with the ONNX outputs
  by position. A non-finite metric value fails the accuracy check; the
  workflow rejects a report carrying one.
- Benchmarks report a distribution, never a single number: warm up, time each
  run, report mean, median, and p95 in ms for both sides, speedup as the
  ratio of medians. Eager is timed as the user wrote the model; the engine at
  the configured precision. Outputs are paired by position and every pair
  must reach the threshold.
- The model check's success report must carry the benchmark; the workflow
  refuses a success without one.
- Loading the model, validating `input.npz`, and the eager forward are
  prerequisites: if one fails, or the model check leaves no usable report
  before showing they passed, neither the PT2 export nor any later step runs.
  The subprocess creates a prerequisites marker before the ONNX/TensorRT
  work, so a crash after it (no usable report) is an ONNX/TensorRT failure,
  `model_check_crash`. The workflow graph routes on this explicitly after
  `run_model_check`. Past them, the ONNX/TensorRT
  branch and the PT2/kernelgen branch are independent and run in that order:
  an ONNX/TensorRT failure is kept in `tensorrt_error` and kernelgen still
  runs; a kernelgen failure keeps the ONNX/TensorRT files and benchmark. Either
  failure fails the run, and the message reports both branches.
- Customer-facing text lives only in `messages.py`, one table per check,
  keyed by stage. Stage names are one flat namespace across all checks;
  `messages.merge()` and a test reject duplicates, and every `stage = "..."`
  in a check must have an entry.
- A registry entry in `environments.yaml`, its Dockerfile under `docker/`,
  and the Dockerfile's labels state the same versions. Change all three
  together; a test compares the entry with the labels.
- The model-check subprocess talks to the workflow only through `report.json`
  and the prerequisites marker in a temporary directory. Never through stdout: the user's model owns
  stdout. Logs go to stderr, which the workflow inherits.
- Files produced for the user (`.onnx`, `.engine`, the kernelgen files) go
  only to the run directory: a new directory under the artifact root
  (`.artifact` in the working directory unless `--artifacts` says otherwise),
  named for the run's UTC start time and created exclusively by the workflow
  before the model check, with a numeric suffix if the name is taken. It
  travels in state as `run_path` to the model-check subprocess and the
  kernelgen nodes. Because every run starts in an empty directory of its own,
  no stale file sits next to a fresh one, and the workflow deletes nothing:
  earlier runs and anything else in the root are never touched. Every such
  file is listed in `artifacts.py`. Each node writes its own file as soon as
  it has it, so a later failure keeps the earlier files in its run directory. Intermediate models also stay in
  state in memory for the next node.
- Model-stack packages (`onnx`, `onnxsim`, and the exporter's lazy imports)
  are imported inside the node that needs them, never at module top, so a
  missing package in the environment reaches the report as a stage instead
  of crashing the subprocess before it can write the report.
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
- Anything that shells out (the probe, the model-check subprocess,
  `yasp-toolkit`, the path lookup) is a module-level function so tests can
  replace it with `monkeypatch`.

## Testing

- Tests come first: add the scenario to the parametrized table, see it fail,
  then implement.
- One test file per check. Fixtures are source strings written to
  `tmp_path`; nothing binary is committed. Tests never shell out for the
  workflow: autouse fixtures replace the preflight probe, the toolkit lookup,
  and `yasp-toolkit`, and run the model-check graph in-process on CPU. No
  test needs a Yasp key.
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
- No image is required to run the check. A container built from an
  environment's Dockerfile copies `src/` in, so rebuild it after changing
  `src/` if that container is the environment you run the check in.
- Branch off `main`, one PR per concern, commits that each do one thing.
- Before finishing: run the tests and report the result verbatim, verify the
  change does what was asked and nothing more, update `README.md`,
  `architecture.md`, or this file when the change alters the user contract,
  the data flow, or a decision, and report what was changed, what was not
  done, and anything that still needs the user's input.

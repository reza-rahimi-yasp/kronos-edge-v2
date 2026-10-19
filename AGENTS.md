# AGENTS.md

Rules for changing this repository. Keep this file to rules an agent must
follow; explanation belongs elsewhere.

## Where things are documented

- `README.md` is the user contract: requirements, install, config, inputs,
  commands, output files, and failure messages.
- `architecture.md` is the data flow: both graphs, every node and its stages,
  process boundaries, the run directory, and the reasons behind them.
- Say each thing once, in the file that owns it, and link to it from the
  others. Do not add file listings, command recipes, or implementation
  summaries here.

## Boundaries

- The workflow process imports neither torch nor numpy and never runs user
  code in-process. Only the model-check graph's modules import the model
  stack; the preflight probe does so only in its subprocess. Inside those
  modules, import `onnx`, `onnxsim`, and the exporter's lazy imports inside
  the node that needs them, never at module top.
- No image is needed at run time. Nothing under `src/` runs `docker` or
  checks for an image; there is no Docker detection and no driver-version
  gate. The config's `driver` identifies the registry entry and is not
  checked.
- A registry entry in `environments.yaml`, its Dockerfile under `docker/`, and
  that Dockerfile's labels state the same versions: change all three together.
- Gaia is used only through the `yasp-toolkit` commands documented at
  https://compile.yasp.ai/help/ ("Generate your first kernel": runner, export,
  generate, inspect), each run as a subprocess with its output captured. Never
  depend on or import a `yasp-compile-*` package, replicate an exporter or
  evaluator, call a service except through `yasp-toolkit`, or parse a field of
  a Yasp report. The check shows the job's report through `eval inspect` and
  judges nothing itself. Deeper integration is deferred.
- The kernelgen reference is the documented `kernelgen export` of `model.py`
  alone: do not force its device or feed it the benchmarked instance or
  `input.npz`. Upload the reference alone (no `--source`). `eval_routing` is
  required, and kernelgen never runs until `eval workers` lists a row whose GPU
  Name and Routing equal the config's `gpu` and `eval_routing`.
- Never start, stop, or remove a Gaia worker, and never mint a key.
- The check installs nothing and has no dependency setting. A
  `ModuleNotFoundError` from `model.py` is the `missing_dependency` stage,
  naming the module.
- `config.yaml`'s five environment fields must equal a registry entry;
  settings never take part in matching. Accept a metric only once it is
  implemented; the threshold is the user's, never a constant in code.

## Credentials

- Only `yasp-toolkit` reads the Yasp API key. The processes that run the
  user's model (the model-check subprocess and `kernelgen export`) get the
  environment without `YASP_*` and `REGISTRY_KEY` and with Python's keyring
  set to its null backend. The service commands keep the full environment.
  Do not describe this as a sandbox.
- `yasp-agent` is the only yasp package installed directly, with pipx from the
  yasp package index, never from `requirements.txt`. The registry key reaches
  pip through a temporary netrc file, never a command line.
- Credentials are documented in `.env.example` only; nothing reads a `.env`
  file.

## Behaviour that must hold

- The preflight runs before anything touches the model: it compares this
  environment with the matched entry by version component and reports every
  difference at once, and it requires `yasp-toolkit` on the path.
- Prerequisites gate everything: loading the model, validating `input.npz`,
  and the eager forward. The subprocess creates a prerequisites marker before
  the ONNX/TensorRT work. After the model check: a prerequisite failure, or no
  usable report and no marker, stops the run; an ONNX/TensorRT failure, or no
  usable report with the marker (`model_check_crash`), goes to
  `tensorrt_error` and the kernelgen branch still runs. Either branch failing
  fails the run, the message reports both, and each keeps its files.
- The model-check subprocess talks to the workflow only through `report.json`
  and the prerequisites marker. Never through stdout, which the user's model
  owns; logs go to stderr.
- A `yasp-toolkit` exit code is not trusted: a node requires the files it
  asked for.
- Precision is applied once, at the ONNX export, to a copy, for `fp32` and
  `fp16` alike. The eager model in state is never cast. The TensorRT network
  is strongly typed.
- The engine is built for the user's inputs only; nothing is generalized.
- The user's inputs in state are never mutated: every node runs the model on
  copies.
- Benchmarks report a distribution (mean, median, p95 in ms for both sides;
  speedup as the ratio of medians), never one number. Outputs pair by position
  after pytree flattening, every pair must reach the threshold, and a
  non-finite metric fails. A success report must carry the benchmark.
- The TensorRT runner invariants in `architecture.md` (`benchmark`) prevent
  segfaults; keep them.
- Files for the user go only to this run's directory, created exclusively by
  the workflow before the model check. The workflow deletes nothing. List
  every such file in `artifacts.py`, and write each one as soon as the node
  has it.
- Customer-facing text lives only in `messages.py`, one table per check, keyed
  by stage. Stage names are one flat namespace; every `stage = "..."` needs
  an entry.

## Node conventions

- Every node wraps its body in one `try/except Exception` and returns
  `fail(logger, stage, exc)` on error. Exceptions never escape a node.
- Failure details travel in state (`error`, `error_detail`, or
  `tensorrt_error`, `tensorrt_error_detail`), never in a shared log buffer.
  Logs are for operators; state is for the workflow.
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
- TensorRT's real work stays in one module-level function per node
  (`tensorrt_check.build`, `benchmark_check.load_engine`) that an autouse
  fixture stubs; tests cover the node around it and the import failure. Verify
  the real paths with a GPU run before each merge.
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
- A container built from an environment's Dockerfile copies `src/` in; rebuild
  it after changing `src/` if you run the check in it.
- Branch off `main`, one PR per concern, commits that each do one thing.
- Before finishing: run the tests and report the result verbatim, verify the
  change does what was asked and nothing more, update `README.md`,
  `architecture.md`, or this file when the change alters the user contract,
  the data flow, or a rule, and report what was changed, what was not done,
  and anything that still needs the user's input.

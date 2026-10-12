# AGENTS.md

Decisions that govern this repository. `README.md` has the user contract and
the commands; `architecture.md` has the data flow. Keep this file to
decisions: do not add file listings, commands, or summaries here.

## Structure

- Two LangGraph graphs: the host graph validates config and environment, the
  container graph runs the model. The host never runs user code.
- The host must import without torch or numpy. Only the container graph's
  modules (`eager_check`, `pt2_check`, `onnx_check`, `onnxsim_check`,
  `tensorrt_check`, `benchmark_check`) import the model stack. A test
  enforces it.
- Kernelgen runs on the host, after the container, as `yasp-toolkit`
  subprocesses (`kernelgen compile`, then `eval compare`): the toolkit needs
  the user's Yasp API key, which never enters the container, and the host
  never imports yasp-agent. The `kernelgen` node requires the container's
  `model.pt2` and `model.engine` before it starts. The toolkit exits 0 after
  some early HTTP errors, so a node trusts the file it asked for, not the
  exit code.
- The kernelgen reference is produced in the container by `torch.export`
  directly (inference mode, Core ATen decompositions, `torch.export.save`),
  mirroring `yasp-toolkit kernelgen export`: the image has no yasp-agent and
  the host must not run user code. It is exported right after the eager pass
  and from the uncast model.
- The kernelgen evaluation runs on this machine: `eval compare` is routed to
  the evaluator registered from it by the `eval_routing` setting, and the same
  key is passed to `kernelgen compile --eval-routing`. The setting is
  required, so a config without it fails validation before any job is
  submitted; an unrouted job could land on any evaluator with the same GPU
  name. The kernel must pass
  the evaluator's correctness checks; its speedup is reported, not judged, and
  no expectation is derived from TensorRT versus the eager run.
- `config.yaml` has two kinds of fields. The five environment fields must
  equal a registry entry and select the image. Settings (`precision`,
  `metric`, `threshold`) are validated by the same schema but never take part
  in matching; changing a setting re-runs the checks in the same environment
  and overwrites the artifacts. `config.Config` is the environment schema plus
  the settings. A metric is accepted only once it is implemented; the
  threshold is the user's, never a constant in code.
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
  next to a fresh one. Each node writes its own file as soon as it has it, so
  a later failure keeps the earlier files. Intermediate models also stay in
  state in memory for the next node.
- Model-stack packages (`onnx`, `onnxsim`, and the exporter's lazy imports)
  are imported inside the node that needs them, never at module top, so a
  missing package in the image reaches the report as a stage instead of
  crashing the container before it can write the report.
- Models above the 2 GB protobuf limit are saved with external data and
  checked by path. The onnxsim step is unverified above that size.
- `yasp-agent[gaia]` comes from the yasp package index, not PyPI; the
  `README.md` install command names the index. Credentials are documented in
  `.env.example` only; nothing reads a `.env` file.

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

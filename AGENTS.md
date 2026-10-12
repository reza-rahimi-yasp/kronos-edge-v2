# AGENTS.md

Decisions that govern this repository. `README.md` has the user contract and
the commands; `architecture.md` has the data flow. Keep this file to
decisions: do not add file listings, commands, or summaries here.

## Structure

- Two LangGraph graphs: the host graph validates config and environment, the
  container graph runs the model. The host never runs user code.
- The host must import without torch or numpy. Only the container graph's
  modules (`eager_check`, `onnx_check`) import the model stack. A test
  enforces it.
- Customer-facing text lives only in `messages.py`, one table per check,
  keyed by stage. Stage names are one flat namespace across all checks;
  `messages.merge()` and a test reject duplicates, and every `stage = "..."`
  in a check must have an entry.
- A registry entry in `environments.yaml`, its Dockerfile under `docker/`,
  and the image labels state the same versions. Change all three together.
- The container talks to the host only through `report.json` in the mounted
  report directory. Never through stdout: the user's model owns stdout. Logs
  go to stderr, which the host inherits.
- The ONNX export is a check, not an output: it runs in memory and no `.onnx`
  file is written.

## Node conventions

- Every node wraps its body in one `try/except Exception` and returns
  `fail(logger, stage, exc)` on error. Exceptions never escape a node.
- Failure details travel in state (`error`, `error_detail`), never in a shared
  log buffer. Logs are for operators; state is for the workflow.
- Log through `kronos_edge.log.get_logger` and tag records with
  `extra={"stage": ...}`. A module that can run under `python -m` names its
  logger explicitly, because `__name__` becomes `__main__` there.
- Anything that shells out (`docker`, `nvidia-smi`) is a module-level function
  so tests can replace it with `monkeypatch`.

## Testing

- Tests come first: add the scenario to the parametrized table, see it fail,
  then implement.
- One test file per check. Fixtures are source strings written to
  `tmp_path`; nothing binary is committed. Tests never shell out: autouse
  fixtures replace the `docker` and `nvidia-smi` probes and run the container
  graph in-process on CPU.
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

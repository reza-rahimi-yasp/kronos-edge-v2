# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this repo is

A LangGraph state machine that validates a user's `config.yaml` against the
supported environments, then runs their PyTorch `model.py` with the inputs in
their `input.npz` in eager mode on the GPU, inside that environment's Docker
container. `README.md` has the
user-facing contract and how to run it. `architecture.md` has the state
diagram, the component list, the failure stages, and how to add a check.

## Repository map

| Path | What lives there |
|---|---|
| `src/kronos_edge/log.py` | Logging wrapper. `TRACE` level, `configure()`, `get_logger()`, and the `stage` record field. |
| `src/kronos_edge/state.py` | Shared `State`, `route_on_error()`, `fail()`. Must not import the other two modules. |
| `src/kronos_edge/config.py` | Pydantic schema for `config.yaml` and the registry; `parse_config()`, `load_environments()`. |
| `src/kronos_edge/environments.yaml` | Registry of supported environments. One entry today. Must agree with its Dockerfile. |
| `src/kronos_edge/env_check.py` | The environment check nodes, their `CUSTOMER_MESSAGES`, and the `docker` and `nvidia-smi` probes. |
| `docker/<name>/Dockerfile` | One per registry entry, pinning the entry's CUDA, TensorRT, and torch versions. `.dockerignore` keeps `.venv` out of the build context. |
| `src/kronos_edge/container.py` | The `run_in_container` host node and the `docker run` probe. |
| `src/kronos_edge/eager_check.py` | The eager check nodes, `InvalidUserInputs`, its `CUSTOMER_MESSAGES`, its sub-graph, and the `main()` that runs inside the container and writes the JSON report. |
| `src/kronos_edge/workflow.py` | Host graph assembly, `success` and `failure` nodes, `run_check()`, CLI `main()`. |
| `tests/conftest.py` | Shared fixtures. Autouse patches replace `env_check.image_exists` and `env_check.host_gpu`, so tests never shell out. |
| `tests/test_container.py`, `tests/test_env_check.py`, `tests/test_eager_check.py` | All tests. Fixtures are source strings written to `tmp_path`; nothing binary is committed. |
| `pyproject.toml` | Runtime deps, a `dev` dependency group with pytest, and `pythonpath = ["src"]` for pytest. No build system. |
| `requirements.txt` | Pinned runtime deps, kept in sync with `pyproject.toml`. |

## Running things

```bash
.venv/bin/pytest                                  # all tests
PYTHONPATH=src .venv/bin/python -m kronos_edge.workflow config.yaml model.py input.npz
docker build -f docker/rtx5090-cuda130/Dockerfile -t kronos-edge:rtx5090-cuda130 .
```

A gitignored `.venv` is expected at the repo root. Do not add lockfiles.

## Conventions in this code

- Every node wraps its body in one `try/except Exception` and returns
  `fail(logger, stage, exc)` on error. Exceptions never escape a node.
- Failure details travel in state (`error`, `error_detail`), never in a shared
  log buffer. Logs are for operators; state is for the workflow.
- Log through `kronos_edge.log.get_logger` and tag records with
  `extra={"stage": ...}`. A module that can run under `python -m` must name
  its logger explicitly, because `__name__` becomes `__main__` there.
- Customer-facing text lives only in `CUSTOMER_MESSAGES` tables, keyed by stage.
- A registry entry, its Dockerfile, and the image labels must state the same
  versions. Change all three together.
- Anything that shells out (`docker`, `nvidia-smi`) is a module-level function
  so tests can replace it with `monkeypatch`.
- The container talks to the host only through `report.json` in the mounted
  report directory, validated by `container.Report`. Never through stdout:
  the user's model owns stdout. Logs go to stderr, which the host inherits.
- Rebuild the image after changing anything under `src/`; it is copied in.
- Tests come first: add the scenario to the parametrized table, see it fail,
  then implement.
- Branch off `main`, one PR per concern, commits that each do one thing.

## Scope

- Work only inside this repository. Do not read, reference, or borrow code from
  any other project on this machine.
- Do not assume anything about the project beyond what is in this repo. If the
  codebase does not show it, treat it as unknown.

## Implementation rules

- Implement the minimum change that satisfies the user's request. No extra
  features, refactors, abstractions, or "while I'm here" cleanups.
- Do not add dependencies, tooling, config, or scaffolding unless the request
  requires it.
- Keep changes small and reviewable. Prefer editing existing files over
  creating new ones.
- Match the existing style and structure of the code you touch.

## When unsure

- If the request is ambiguous, or you are not sure which feature or behavior
  the user wants, ask a concise clarifying question before writing code.
- Do not guess at requirements. A wrong assumption costs more than a question.
- State any small, low-risk assumption you do make explicitly in your reply.

## Before finishing

- Run the tests and report the result verbatim.
- Verify the change does what was asked and nothing more.
- Update `README.md`, `architecture.md`, or this file when the change alters
  the user contract, a component, or a convention described there.
- Report exactly what was changed, what was not done, and anything that still
  needs the user's input.

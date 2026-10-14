"""Evaluator check: make sure an evaluator for the config's GPU and routing key is registered before kernelgen.

Runs on the host, between the container and kernelgen. A kernelgen job routed
by ``eval_routing`` to an evaluator that is not registered sits in the queue
indefinitely, so the host checks first and starts one when it has to. Two
nodes:

- ``ensure_evaluator``: lists the evaluators registered under the config's
  routing key with ``yasp-toolkit eval workers`` and looks for one advertising
  exactly the config's ``gpu``. When there is none, it starts ``yasp-eval
  register --tag routing=<key>`` on this machine (the host GPU already matched
  the config's, and the command detects the GPU and the CUDA build toolchain
  itself, so its name is not overridden) and polls the listing until the
  evaluator appears, for a bounded time. The started process is recorded in
  state as ``evaluator``; an evaluator that was already registered is not.
- ``release_evaluator``: stops the evaluator in state, if any. The graph runs
  it after every outcome of the kernelgen nodes, so a process this check
  started never outlives the check; an evaluator it did not start is never
  touched. A failure to stop is reported only when nothing failed before it.

Both commands come from the yasp packages (``yasp-agent`` for the toolkit,
``yasp-compile-eval`` for ``yasp-eval``) and read the user's Yasp API key
themselves, so they run as host subprocesses and the host imports neither.
The evaluator's output is captured to a temporary file: it must not reach
stdout, which carries the check's result, and it is the failure detail when
the process exits before registering.
"""

import os
import signal
import subprocess
import tempfile
import time

from kronos_edge.kernelgen_check import MISSING_TOKEN, TOOLKIT, UNAUTHORIZED, MissingCredentials, ToolkitFailed, run_logged
from kronos_edge.log import get_logger
from kronos_edge.messages import EVALUATOR as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)

EVALUATOR = "yasp-eval"
# How long a started evaluator may take to appear in the listing, and how often the listing is asked.
READY_TIMEOUT_S = 120.0
POLL_INTERVAL_S = 5.0
# A single inventory request must not consume the whole readiness window.
LISTING_TIMEOUT_S = 30.0
# How long a stopped evaluator may take to exit after the terminate signal before it is killed.
STOP_TIMEOUT_S = 10.0
# What `eval workers` prints instead of a table when nothing matches the filter.
NO_EVALUATORS = "No evaluators"
# The header of the table's first column; a listing without it is not a listing.
TABLE_HEADER = "GPU Name"

_active_evaluator = None


class Evaluator:
    """A ``yasp-eval register`` process this check started, with its output captured to a temporary file."""

    def __init__(self, process: subprocess.Popen, log):
        self.process, self.log = process, log

    def exit_code(self) -> int | None:
        return self.process.poll()

    def output(self) -> str:
        self.log.seek(0)
        return self.log.read()


def start_evaluator(args: list[str]) -> Evaluator:
    """Start yasp-eval with ``args``, its output captured; a missing command raises FileNotFoundError."""
    log = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            [EVALUATOR, *args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    return Evaluator(process, log)


def stop_evaluator(evaluator: Evaluator) -> None:
    """Terminate the process, kill it if it does not exit in time, and drop its output."""
    process = evaluator.process
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    finally:
        evaluator.log.close()


def listed(output: str, gpu: str, routing: str) -> bool:
    """Whether the toolkit's ``eval workers`` output has a row whose GPU Name and Routing cells are exactly these.

    The table's rows are ``│``-separated cells; a row whose Tags cell is wider than the terminal
    continues on lines whose leading cells are empty, so those are folded into the row above.
    An output that is neither the no-match line nor a table is a toolkit flow that stopped early.
    """
    if NO_EVALUATORS in output:
        return False
    if TABLE_HEADER not in output:
        raise ToolkitFailed(f"{TOOLKIT} exited 0 without listing evaluators\n{output}")
    rows: list[list[str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.strip("│").split("│")]
        if rows and not cells[0]:
            rows[-1] = [a + b for a, b in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    return any(row[:2] == [gpu, routing] for row in rows)


def _registered(stage: str, gpu: str, routing: str, timeout: float | None = None) -> bool:
    """Ask the toolkit for the evaluators under ``routing`` and look for one advertising ``gpu``."""
    _, output = run_logged(stage, ["eval", "workers", "--tag", f"routing={routing}"], timeout=timeout)
    return listed(output, gpu, routing)


def ensure_evaluator(state: State) -> State:
    stage = "evaluator_inventory"
    global _active_evaluator
    evaluator = None
    try:
        config = state["config"]
        gpu, routing = config.gpu, config.eval_routing
        try:
            found = _registered(stage, gpu, routing, timeout=LISTING_TIMEOUT_S)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        except MissingCredentials:
            stage = "kernelgen_credentials"
            raise
        if found:
            logger.info("an evaluator for %s with routing %s is registered", gpu, routing, extra={"stage": stage})
            return {}

        stage = "evaluator_start"
        args = ["register", "--tag", f"routing={routing}"]
        logger.info("no evaluator for %s with routing %s; running %s %s", gpu, routing, EVALUATOR, " ".join(args), extra={"stage": stage})
        try:
            evaluator = start_evaluator(args)
            _active_evaluator = evaluator
        except FileNotFoundError:
            stage = "evaluator_command"
            raise

        stage = "evaluator_ready"
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            code = evaluator.exit_code()
            if code is not None:
                output = evaluator.output()
                stage = "kernelgen_credentials" if MISSING_TOKEN in output or UNAUTHORIZED in output else "evaluator_start"
                raise RuntimeError(f"{EVALUATOR} exited {code} before registering\n{output.strip()}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no evaluator advertising {gpu!r} with routing {routing!r} was listed within {READY_TIMEOUT_S:.0f} s "
                    f"of starting {EVALUATOR}"
                )
            if _registered(stage, gpu, routing, timeout=min(LISTING_TIMEOUT_S, remaining)):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no evaluator advertising {gpu!r} with routing {routing!r} was listed within {READY_TIMEOUT_S:.0f} s "
                    f"of starting {EVALUATOR}"
                )
            time.sleep(POLL_INTERVAL_S)
        logger.info("the evaluator started on this machine is registered", extra={"stage": stage})
    except Exception as exc:
        update = fail(logger, stage, exc)
        if evaluator is not None:  # started, so release_evaluator must stop it
            update["evaluator"] = evaluator
        return update
    return {"evaluator": evaluator}


def release_evaluator(state: State) -> State:
    global _active_evaluator
    stage = "evaluator_release"
    evaluator = state.get("evaluator")
    if evaluator is None:
        return {}
    try:
        logger.info("stopping the evaluator this check started", extra={"stage": stage})
        stop_evaluator(evaluator)
        _active_evaluator = None
    except Exception as exc:
        _active_evaluator = None
        if state.get("error"):
            # The failure that brought the check here is the one to report; the leak is for the operator.
            logger.error("%s: %s", type(exc).__name__, exc, extra={"stage": stage})
            return {"evaluator": None}
        return {**fail(logger, stage, exc), "evaluator": None}
    return {"evaluator": None}


def release_active_evaluator() -> None:
    """Stop the evaluator outside the graph, preserving the original interruption."""
    global _active_evaluator
    evaluator = _active_evaluator
    if evaluator is None:
        return
    try:
        stop_evaluator(evaluator)
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc, extra={"stage": "evaluator_release"})
    finally:
        _active_evaluator = None

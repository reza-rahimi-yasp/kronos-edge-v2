"""The evaluator check: before kernelgen, the host makes sure an evaluator for the config's gpu and eval_routing is registered.

`yasp-toolkit eval workers` lists the evaluators and a Gaia worker container runs one, so both
are replaced by autouse fixtures (`toolkit_stub`, `evaluators`); the tests here cover the two
nodes around them, the listing parser against the toolkit's table, the worker lifecycle helpers
against recorded Docker and toolkit commands (nothing shells out, no key is minted, no image is
pulled), and that only the worker the check started is ever removed.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import CONFIG, evaluator_listing, fake_toolkit
from kronos_edge import evaluator_check, kernelgen_check
from kronos_edge.evaluator_check import DockerFailed
from kronos_edge.kernelgen_check import MissingCredentials, ToolkitFailed
from kronos_edge.workflow import main
from test_eager_check import CONV_RELU, TYPO, X, error_records, run, write_model, write_npz
from test_kernelgen_check import INVALID_TOKEN_RESPONSE, MISSING_TOKEN_PANEL, eval_fails, failing_toolkit, option

# Captured at import, before the autouse fixture replaces them with the recorder.
REAL_START = evaluator_check.start_evaluator
REAL_STOP = evaluator_check.stop_evaluator

GPU, ROUTING = CONFIG["gpu"], CONFIG["eval_routing"]

REPO = Path(__file__).resolve().parents[1]
IMAGE = "docker.yasp.ai/yasp-eval/nvidia:1.10.0"
# Shaped like the JWTs the toolkit's key commands print; never a real key.
REGISTRY_KEY = "eyJ0.registry-scope.sig1"
WORKER_KEY = "eyJ0.eval-register-scope.sig2"
RUNTIME_MISSING = "docker: Error response from daemon: unknown or invalid runtime name: nvidia.\n"


def listing_toolkit(calls, evaluators, before=(), after=None):
    """A toolkit whose `eval workers` lists `before` until the check starts an evaluator, then `after`.

    `after` defaults to the started evaluator itself, advertising the config's GPU under the routing
    key it was started with. Every command is appended to `calls`.
    """

    def toolkit(args, timeout=None):
        calls.append(args)
        if args[:2] != ["eval", "workers"]:
            return fake_toolkit(args)
        rows = before
        if evaluators.started:
            rows = after if after is not None else [(GPU, evaluators.started[-1].args[-1].removeprefix("routing="))]
        return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=evaluator_listing(*rows), stderr="")

    return toolkit


def commands(calls):
    return [args[:2] for args in calls]


def test_a_registered_evaluator_is_used_and_never_stopped(tmp_path, monkeypatch, env_files, good_model, evaluators):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators, before=[(GPU, ROUTING)]))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert commands(calls) == [["eval", "workers"], ["kernelgen", "compile"], ["eval", "compare"]]
    assert option(calls[0], "--tag") == f"routing={ROUTING}"
    assert evaluators.started == [] and evaluators.stopped == []
    assert state.get("evaluator") is None


def test_a_missing_evaluator_is_started_awaited_and_stopped_after_the_evaluation(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert [e.args for e in evaluators.started] == [["register", "--tag", f"routing={ROUTING}"]]
    # Listed once (nothing there), started, listed again until registered, then kernelgen and the evaluation.
    assert commands(calls) == [["eval", "workers"], ["eval", "workers"], ["kernelgen", "compile"], ["eval", "compare"]]
    assert evaluators.stopped == evaluators.started
    assert state.get("evaluator") is None
    assert error_records(caplog) == []


def test_the_evaluator_is_started_before_being_listed_and_stopped_after_the_evaluation(tmp_path, monkeypatch, env_files, good_model, evaluators):
    """Ordering: the second listing sees the started evaluator; the stop comes after `eval compare`."""
    events = []
    real_start, real_stop = evaluators.start, evaluators.stop

    def toolkit(args, timeout=None):
        events.append(args[:2])
        return listing_toolkit([], evaluators)(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)
    monkeypatch.setattr(evaluator_check, "start_evaluator", lambda args: (events.append(["start"]), real_start(args))[1])
    monkeypatch.setattr(evaluator_check, "stop_evaluator", lambda e: (events.append(["stop"]), real_stop(e))[1])

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert events == [["eval", "workers"], ["start"], ["eval", "workers"], ["kernelgen", "compile"], ["eval", "compare"], ["stop"]]


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([("NVIDIA GeForce RTX 4090", ROUTING)], id="other_gpu"),
        pytest.param([(GPU, "someone-else")], id="other_routing"),
        pytest.param([(GPU, "—")], id="no_routing"),
        pytest.param([(GPU.lower(), ROUTING)], id="gpu_differs_in_case"),
        pytest.param([("NVIDIA GeForce RTX 4090", ROUTING), (GPU, "someone-else")], id="each_half_matches"),
    ],
)
def test_matching_is_exact_on_gpu_and_routing(tmp_path, monkeypatch, env_files, good_model, evaluators, rows):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators, before=rows))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert len(evaluators.started) == 1
    assert evaluators.stopped == evaluators.started


def test_readiness_polls_until_the_evaluator_is_listed(tmp_path, monkeypatch, env_files, good_model, evaluators):
    calls = []
    listings = []

    def toolkit(args, timeout=None):
        calls.append(args)
        if args[:2] != ["eval", "workers"]:
            return fake_toolkit(args)
        listings.append(args)
        rows = [(GPU, ROUTING)] if len(listings) > 3 else []
        return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=evaluator_listing(*rows), stderr="")

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)
    monkeypatch.setattr(evaluator_check, "READY_TIMEOUT_S", 30)

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert len(listings) == 4
    assert commands(calls)[-2:] == [["kernelgen", "compile"], ["eval", "compare"]]


def test_an_evaluator_that_never_registers_is_stopped_and_kernelgen_is_skipped(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators, after=[]))
    monkeypatch.setattr(evaluator_check, "READY_TIMEOUT_S", 0)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_ready"
    assert GPU in state["customer_message"] and ROUTING in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_ready"]
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert ["kernelgen", "compile"] not in commands(calls)
    assert state.get("evaluator") is None


@pytest.mark.parametrize(
    "exit, stage, fragment",
    [
        pytest.param((125, RUNTIME_MISSING), "evaluator_start", "runtime name", id="container_could_not_start"),
        pytest.param((1, "Could not reach router: timeout\n"), "evaluator_start", "reach router", id="router_unreachable"),
        pytest.param((1, MISSING_TOKEN_PANEL), "kernelgen_credentials", "YASP_API_KEY", id="no_api_key"),
        pytest.param((0, ""), "evaluator_start", "exited 0", id="exited_cleanly"),
    ],
)
def test_an_evaluator_that_exits_before_registering_is_a_stage(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators, exit, stage, fragment):
    calls = []
    evaluators.exits = exit
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators, after=[]))

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert evaluators.started[0].name in state["customer_message"]  # the container whose output this is
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert ["kernelgen", "compile"] not in commands(calls)


@pytest.mark.parametrize(
    "failure, stage, fragment",
    [
        pytest.param(FileNotFoundError(2, "No such file or directory", "docker"), "evaluator_command", "Docker", id="docker_not_installed"),
        pytest.param(FileNotFoundError(2, "No such file or directory", "/var/run/docker.sock"), "evaluator_command", "Docker", id="no_docker_socket"),
        pytest.param(FileNotFoundError(2, "No such file or directory", "yasp-toolkit"), "kernelgen_toolkit", "yasp-agent", id="toolkit_gone_before_the_key_mint"),
        pytest.param(MissingCredentials("yasp-toolkit eval key found no valid Yasp API key"), "kernelgen_credentials", "YASP_API_KEY", id="key_mint_rejected"),
        pytest.param(ToolkitFailed("yasp-toolkit registry key exited 1 without printing a key"), "evaluator_start", "without printing a key", id="key_mint_failed"),
        pytest.param(DockerFailed("docker pull docker.yasp.ai/yasp-eval/nvidia:1.10.0 exited 1\nunauthorized"), "evaluator_start", "unauthorized", id="pull_failed"),
        pytest.param(subprocess.TimeoutExpired(["docker", "pull", IMAGE], 1800), "evaluator_start", "timed out", id="pull_hung"),
    ],
)
def test_a_startup_failure_before_the_worker_exists_is_a_stage_and_stops_nothing(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators, failure, stage, fragment):
    """What `start_evaluator` raises before it has a container: the stage follows the command that failed."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators))
    monkeypatch.setattr(evaluator_check, "start_evaluator", lambda args: (_ for _ in ()).throw(failure))

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluators.stopped == [] and commands(calls) == [["eval", "workers"]]
    assert state.get("evaluator") is None


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(
            lambda args, timeout=None: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-toolkit")),
            "kernelgen_toolkit", "yasp-agent", id="toolkit_not_installed",
        ),
        pytest.param(failing_toolkit(1, stderr=MISSING_TOKEN_PANEL), "kernelgen_credentials", "YASP_API_KEY", id="no_api_key"),
        pytest.param(failing_toolkit(0, stderr=INVALID_TOKEN_RESPONSE), "kernelgen_credentials", "not accepted", id="invalid_api_key"),
        pytest.param(failing_toolkit(1, stderr="Error: --tag expects KEY=VALUE\n"), "evaluator_inventory", "KEY=VALUE", id="exit_nonzero"),
        pytest.param(failing_toolkit(0, stderr="╭─ GET https://compile.yasp.ai/eval/evaluators ─╮\n│ HTTP 503 │\n╰──╯\n"), "evaluator_inventory", "503", id="exit_zero_without_a_listing"),
    ],
)
def test_listing_failures_are_stages_and_start_nothing(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators, toolkit, stage, fragment):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluators.started == [] and evaluators.stopped == []
    assert not (tmp_path / ".artifact" / "kernel.py").exists()


def compile_fails(args, timeout=None):
    if args[:2] == ["kernelgen", "compile"]:
        return subprocess.CompletedProcess(["yasp-toolkit", *args], 1, stdout="", stderr="╭─ Failure reason ─╮\n│ worker exploded │\n╰──╯\n")
    return fake_toolkit(args)


@pytest.mark.parametrize(
    "toolkit, stage",
    [
        pytest.param(compile_fails, "kernelgen_compile", id="kernelgen_fails"),
        pytest.param(eval_fails(1, "╭─ Failure reason ─╮\n│ evaluator crashed │\n╰──╯\n"), "kernelgen_eval", id="evaluation_fails"),
    ],
)
def test_a_started_evaluator_is_stopped_when_a_later_node_fails(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators, toolkit, stage):
    calls = []
    listing = listing_toolkit(calls, evaluators)
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: listing(args) if args[:2] == ["eval", "workers"] else toolkit(args))

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert state.get("evaluator") is None


def test_an_evaluator_the_check_did_not_start_is_left_alone_when_a_later_node_fails(tmp_path, monkeypatch, env_files, good_model, evaluators):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", compile_fails)

    state = run(env_files, *good_model)

    assert state["error"] == "kernelgen_compile"
    assert evaluators.started == [] and evaluators.stopped == []


def test_nothing_is_listed_or_started_when_the_container_fails(tmp_path, monkeypatch, env_files, evaluators):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators))

    state = run(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert state["error"] == "forward"
    assert calls == [] and evaluators.started == []


def test_a_stop_failure_is_a_stage_only_when_nothing_else_failed(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit([], evaluators))
    monkeypatch.setattr(evaluator_check, "stop_evaluator", lambda e: (_ for _ in ()).throw(PermissionError("not permitted")))

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_release"
    assert "not permitted" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_release"]
    assert (tmp_path / ".artifact" / "kernelgen_eval.json").is_file()  # the evaluation's files are kept


def test_a_stop_failure_does_not_hide_the_failure_that_came_first(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    listing = listing_toolkit([], evaluators)
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: listing(args) if args[:2] == ["eval", "workers"] else compile_fails(args))
    monkeypatch.setattr(evaluator_check, "stop_evaluator", lambda e: (_ for _ in ()).throw(PermissionError("not permitted")))

    state = run(env_files, *good_model)

    assert state["error"] == "kernelgen_compile"
    assert "worker exploded" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["kernelgen_compile", "evaluator_release"]


def test_the_inventory_listing_is_bounded_and_a_hung_one_is_a_stage(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    """A listing that never answers is killed at its timeout, which run_toolkit reports as TimeoutExpired."""
    timeouts = []

    def toolkit(args, timeout=None):
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired(["yasp-toolkit", *args], timeout)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_inventory"
    assert "timed out" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_inventory"]
    assert timeouts == [evaluator_check.LISTING_TIMEOUT_S]
    assert evaluators.started == [] and evaluators.stopped == []


def test_readiness_listings_are_bounded_by_the_time_left_to_the_deadline(tmp_path, monkeypatch, env_files, good_model, evaluators):
    timeouts = []

    def toolkit(args, timeout=None):
        if args[:2] != ["eval", "workers"]:
            return fake_toolkit(args)
        if evaluators.started:
            timeouts.append(timeout)
        rows = [(GPU, ROUTING)] if len(timeouts) == 3 else []
        return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=evaluator_listing(*rows), stderr="")

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)
    monkeypatch.setattr(evaluator_check, "READY_TIMEOUT_S", 30)

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert len(timeouts) == 3
    assert all(0 < t <= 30 for t in timeouts)
    assert timeouts == sorted(timeouts, reverse=True)  # each listing gets what is left, never more


def test_a_listing_that_hangs_while_waiting_for_readiness_stops_the_evaluator(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    calls = []
    listing = listing_toolkit(calls, evaluators)

    def toolkit(args, timeout=None):
        if args[:2] == ["eval", "workers"] and evaluators.started:
            raise subprocess.TimeoutExpired(["yasp-toolkit", *args], timeout)
        return listing(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_ready"
    assert "timed out" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_ready"]
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert ["kernelgen", "compile"] not in commands(calls)


def interrupted_toolkit(calls, evaluators, command, before=()):
    """`listing_toolkit`, except that `command` raises KeyboardInterrupt, as Ctrl-C does while a command runs.

    A listing raises only once an evaluator was started, so the interrupt lands in the readiness wait.
    """
    listing = listing_toolkit(calls, evaluators, before=before)

    def toolkit(args, timeout=None):
        if args[:2] == command and (command != ["eval", "workers"] or evaluators.started):
            raise KeyboardInterrupt
        return listing(args)

    return toolkit


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["eval", "workers"], id="while_waiting_for_readiness"),
        pytest.param(["kernelgen", "compile"], id="during_kernelgen"),
        pytest.param(["eval", "compare"], id="during_the_evaluation"),
    ],
)
def test_an_interrupt_stops_the_evaluator_the_check_started(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators, command):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", interrupted_toolkit([], evaluators, command))

    with pytest.raises(KeyboardInterrupt):
        run(env_files, *good_model)

    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert error_records(caplog) == []


def test_an_interrupt_leaves_an_evaluator_the_check_did_not_start_alone(tmp_path, monkeypatch, env_files, good_model, evaluators):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", interrupted_toolkit([], evaluators, ["kernelgen", "compile"], before=[(GPU, ROUTING)]))

    with pytest.raises(KeyboardInterrupt):
        run(env_files, *good_model)

    assert evaluators.started == [] and evaluators.stopped == []


def test_a_stop_failure_during_an_interrupt_is_logged_and_the_interrupt_is_kept(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", interrupted_toolkit([], evaluators, ["kernelgen", "compile"]))
    monkeypatch.setattr(evaluator_check, "stop_evaluator", lambda e: (_ for _ in ()).throw(PermissionError("not permitted")))

    with pytest.raises(KeyboardInterrupt):
        run(env_files, *good_model)

    assert [r.stage for r in error_records(caplog)] == ["evaluator_release"]


@pytest.fixture
def signal_handlers():
    """main() installs handlers for the termination signals; put the previous ones back after the test."""
    saved = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP], ids=["sigterm", "sighup"])
def test_main_stops_a_started_evaluator_when_the_process_is_signalled(tmp_path, monkeypatch, env_files, good_model, evaluators, signal_handlers, signum):
    """`kill`, a supervisor, or a closed terminal: the signal ends the check through the same cleanup as an interrupt."""
    listing = listing_toolkit([], evaluators)

    def toolkit(args, timeout=None):
        if args[:2] == ["kernelgen", "compile"]:
            os.kill(os.getpid(), signum)  # delivered to this thread before the stub returns
        return listing(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)
    config, environments = env_files
    model, npz = good_model

    with pytest.raises(SystemExit) as info:
        main([str(config), str(model), str(npz), "--environments", str(environments)])

    assert info.value.code == 128 + signum
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1


# `eval workers` as the toolkit prints it at 200 columns: a wide Tags column wraps a row onto a
# second line whose leading cells are empty.
LIVE_LISTING = """3 evaluators
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ GPU Name                                                 ┃ Routing            ┃ Version ┃ Tags                                                                                                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ NVIDIA RTX PRO 6000 Blackwell Server Edition             │ evit-rtx6000-sm120 │ 1.10.0  │ host: rtx6000-vm01                                                                                   │
│ NVIDIA GeForce RTX 3080 Ti                               │ —                  │ 1.10.0  │ cluster: lab, gpu: NVIDIA-GeForce-RTX-3080-Ti, host: nvidia-rnr-03, last_update: 2026-10-06, version: │
│                                                          │                    │         │ 1.10.0                                                                                               │
│ NVIDIA GeForce RTX 5090                                  │ reza-fractal       │ 1.10.0  │ host: fractal                                                                                        │
└──────────────────────────────────────────────────────────┴────────────────────┴─────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


@pytest.mark.parametrize(
    "gpu, routing, listed",
    [
        pytest.param("NVIDIA GeForce RTX 5090", "reza-fractal", True, id="listed"),
        pytest.param("NVIDIA RTX PRO 6000 Blackwell Server Edition", "evit-rtx6000-sm120", True, id="first_row"),
        pytest.param("NVIDIA GeForce RTX 3080 Ti", "—", True, id="wrapped_row"),
        pytest.param("NVIDIA GeForce RTX 3080 Ti", "1.10.0", False, id="continuation_is_not_a_row"),
        pytest.param("NVIDIA GeForce RTX 5090", "evit-rtx6000-sm120", False, id="routing_of_another_row"),
        pytest.param("GPU Name", "Routing", False, id="header"),
    ],
)
def test_listing_parser_matches_rows_of_the_toolkits_table(gpu, routing, listed):
    assert evaluator_check.listed(LIVE_LISTING, gpu, routing) is listed


def test_listing_parser_reports_no_match_and_no_table():
    assert evaluator_check.listed("No evaluators match.\n", GPU, ROUTING) is False
    assert evaluator_check.listed("No evaluators registered.\n", GPU, ROUTING) is False
    with pytest.raises(kernelgen_check.ToolkitFailed, match="without listing"):
        evaluator_check.listed("╭─ GET /evaluators ─╮\n│ HTTP 503 │\n╰──╯\n", GPU, ROUTING)


class FakeClient:
    """Stand in for the `docker start --attach` client: alive until `code` is set."""

    def __init__(self):
        self.pid, self.code, self.waits = 123, None, []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.code is None:
            raise subprocess.TimeoutExpired(["docker", "start", "--attach"], timeout)
        return self.code


@pytest.fixture
def docker(monkeypatch):
    """Record every Docker and toolkit command the worker helpers would run, without running any.

    Keys come back from the recorded toolkit as JWT-shaped strings that are never real; `calls` is
    the ordered list of (command args, keyword arguments) for `run_docker` and `attach_worker`.
    """
    calls = []
    client = FakeClient()

    def run_docker(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "inspect":  # nothing holds the task slot
            raise DockerFailed(f"docker {' '.join(args)} exited 1\nError response from daemon: No such container: {args[-1]}")
        return "container-id"

    def attach_worker(name, log):
        calls.append((["start", "--attach", name], {"log": log}))
        log.write("API Key OK\n")
        log.flush()
        return client

    monkeypatch.setattr(evaluator_check, "run_docker", run_docker)
    monkeypatch.setattr(evaluator_check, "attach_worker", attach_worker)
    monkeypatch.setattr(evaluator_check, "worker_key", lambda args: {"registry": REGISTRY_KEY, "eval": WORKER_KEY}[args[0]])
    monkeypatch.setattr(evaluator_check, "worker_tags", lambda key: ["1.9.0", "latest", "1.10.0", "1.11.0-rc1"])
    monkeypatch.setattr(evaluator_check, "docker_socket_gid", lambda: "998")
    return SimpleNamespace(calls=calls, client=client)


def docker_commands(calls):
    return [args[0] if not args[0].startswith("--") else args[2] for args, _ in calls]


def test_the_worker_is_created_as_the_documented_deployment_with_only_the_scoped_key(monkeypatch, docker, capfd):
    """The `docker create` mirrors the Gaia Docker guide's `docker run`, minus the persistent parts (restart policy,
    autoheal label, health check), with the registration key read from the client's environment, never an argument."""
    monkeypatch.setenv("YASP_API_KEY", "main-api-key")

    evaluator = REAL_START(["register", "--tag", f"routing={ROUTING}"])

    assert docker_commands(docker.calls) == ["login", "pull", "create", "start"]
    create, options = docker.calls[2]
    assert create[-5:] == [IMAGE, "yasp-eval", "register", "--tag", f"routing={ROUTING}"]
    assert option(create, "--name") == evaluator.name and evaluator.name.startswith("kronos-eval-")
    assert option(create, "--runtime") == "nvidia"
    assert option(create, "--group-add") == "998"
    assert option(create, "-v") == "/var/run/docker.sock:/var/run/docker.sock"
    env_flags = [create[i + 1] for i, flag in enumerate(create) if flag == "-e"]
    # The owner marker is set on the worker and forwarded into its task containers, where the stop reads it back.
    assert env_flags == [
        "NVIDIA_VISIBLE_DEVICES=all", "YASP_API_KEY", f"YASP_LAUNCH_WORKER_IMAGE={IMAGE}", "YASP_LAUNCH_DOCKER_GPUS=all",
        "YASP_LAUNCH_DOCKER_SHM_SIZE=8g", "YASP_LAUNCH_DOCKER_VOLUMES=yasp-dataset-cache:/mnt/dataset-cache",
        "YASP_DATASET_CACHE_DIR=/mnt/dataset-cache", f"KRONOS_EVALUATOR={evaluator.name}",
        "YASP_LAUNCH_DOCKER_FORWARD_ENV=YASP_DATASET_CACHE_DIR,KRONOS_EVALUATOR",
    ]
    assert not {"--restart", "--label", "--health-cmd", "--detach"} & set(create)
    assert options["env"]["YASP_API_KEY"] == WORKER_KEY and options["secret"] == WORKER_KEY
    assert docker.calls[3][0] == ["start", "--attach", evaluator.name]
    # The keys travel through stdin and the environment only; the main key goes nowhere.
    assert WORKER_KEY not in str([args for args, _ in docker.calls])
    assert REGISTRY_KEY not in str([args for args, _ in docker.calls])
    assert "main-api-key" not in str(docker.calls)
    assert "API Key OK" in evaluator.output() and evaluator.exit_code() is None
    assert capfd.readouterr().out == ""


def test_the_newest_release_is_pulled_with_a_temporary_login(monkeypatch, docker):
    """Keys are minted before the pull, the login lives in a temporary Docker config that is gone afterwards, and the
    pull has its own bound: a first pull of the worker image is not charged to the readiness window."""
    minted = []
    monkeypatch.setattr(evaluator_check, "worker_key", lambda args: (minted.append(args), {"registry": REGISTRY_KEY, "eval": WORKER_KEY}[args[0]])[1])

    assert evaluator_check.prepare_worker() == (IMAGE, WORKER_KEY)

    assert minted == [["registry", "key"], ["eval", "key"]]
    login, login_options = docker.calls[0]
    pull, pull_options = docker.calls[1]
    assert login[0] == "--config" and login[2:] == ["login", "docker.yasp.ai", "-u", "token", "--password-stdin"]
    assert login_options["input"] == REGISTRY_KEY and login_options["secret"] == REGISTRY_KEY
    assert pull[:2] == login[:2] and pull[2:] == ["pull", IMAGE]
    assert pull_options["timeout"] == evaluator_check.PULL_TIMEOUT_S
    assert not Path(login[1]).exists()


@pytest.mark.parametrize("tags", [[], None, ["latest", "dev"]], ids=["empty", "null", "no_release"])
def test_a_registry_without_a_release_fails_before_logging_in(monkeypatch, docker, tags):
    monkeypatch.setattr(evaluator_check, "worker_tags", lambda key: tags)
    with pytest.raises(DockerFailed, match="no release"):
        evaluator_check.prepare_worker()
    assert docker.calls == []


def test_a_worker_key_is_the_toolkits_output(monkeypatch):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: subprocess.CompletedProcess(args, 0, WORKER_KEY + "\n", ""))
    assert evaluator_check.worker_key(["eval", "key"]) == WORKER_KEY


@pytest.mark.parametrize(
    "returncode, stdout, stderr, error",
    [
        pytest.param(1, "secret.worker.key", "secret-error", ToolkitFailed, id="exit_nonzero"),
        pytest.param(0, "", "secret-error", ToolkitFailed, id="no_key_printed"),
        pytest.param(0, "Deprecated: secret-warning\n" + WORKER_KEY, "", ToolkitFailed, id="key_mixed_with_other_output"),
        pytest.param(1, "", MISSING_TOKEN_PANEL, MissingCredentials, id="no_api_key"),
        pytest.param(0, INVALID_TOKEN_RESPONSE, "", MissingCredentials, id="invalid_api_key"),
    ],
)
def test_key_command_failures_never_expose_their_output(monkeypatch, caplog, returncode, stdout, stderr, error):
    """A key command's output may be a key: it goes nowhere on failure, not into the error, the log, or the detail."""
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: subprocess.CompletedProcess(args, returncode, stdout, stderr))

    with pytest.raises(error) as info:
        evaluator_check.worker_key(["eval", "key"])

    assert "yasp-toolkit eval key" in str(info.value)
    assert "secret" not in str(info.value) and WORKER_KEY not in str(info.value)
    assert "secret" not in caplog.text and WORKER_KEY not in caplog.text


def test_a_hung_key_command_is_bounded(monkeypatch):
    timeouts = []

    def toolkit(args, timeout=None):
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired(["yasp-toolkit", *args], timeout)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)
    with pytest.raises(subprocess.TimeoutExpired):
        evaluator_check.worker_key(["registry", "key"])
    assert timeouts == [evaluator_check.LISTING_TIMEOUT_S]


def test_docker_failures_name_the_command_and_redact_the_secret(monkeypatch):
    """`run_docker` is the one place a key is handed to Docker; its error carries the daemon's message with
    the key blanked, so a failure that echoed it could not reach the customer message or the log."""
    runs = []

    def run(cmd, **kwargs):
        runs.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 1, "", f"Error response from daemon: login attempt with {REGISTRY_KEY} failed\n")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(DockerFailed) as info:
        evaluator_check.run_docker(["login", "docker.yasp.ai", "-u", "token", "--password-stdin"], input=REGISTRY_KEY, secret=REGISTRY_KEY)

    assert str(info.value).startswith("docker login docker.yasp.ai -u token --password-stdin exited 1\n")
    assert REGISTRY_KEY not in str(info.value) and "login attempt with " in str(info.value)
    assert runs[0][0][0] == "docker" and runs[0][1]["input"] == REGISTRY_KEY
    assert runs[0][1]["timeout"] == evaluator_check.DOCKER_TIMEOUT_S and runs[0][1]["capture_output"]


def test_docker_output_is_returned_stripped(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "abc123\n", ""))
    assert evaluator_check.run_docker(["create", "x"]) == "abc123"


@pytest.mark.parametrize(
    "failure, during",
    [
        pytest.param(DockerFailed("docker create exited 125\n" + RUNTIME_MISSING), "create", id="create_fails"),
        pytest.param(KeyboardInterrupt(), "create", id="create_interrupted"),
        pytest.param(OSError("spawn failed"), "start", id="attach_fails"),
        pytest.param(KeyboardInterrupt(), "start", id="attach_interrupted"),
    ],
)
def test_a_start_that_fails_or_is_interrupted_after_create_removes_its_container(monkeypatch, docker, caplog, failure, during):
    """Once `docker create` has been asked, the container may exist whatever the client reported, so it is removed
    by name on every failure after that point, and the failure itself is what propagates."""
    real_run_docker = evaluator_check.run_docker

    def run_docker(args, **kwargs):
        if args[0] == during:
            docker.calls.append((args, kwargs))
            raise failure
        return real_run_docker(args, **kwargs)

    monkeypatch.setattr(evaluator_check, "run_docker", run_docker)
    if during == "start":
        monkeypatch.setattr(evaluator_check, "attach_worker", lambda name, log: (_ for _ in ()).throw(failure))

    with pytest.raises(type(failure)) as info:
        REAL_START(["register"])

    assert info.value is failure
    name = option([args for args, _ in docker.calls if args[0] == "create"][0], "--name")
    assert docker.calls[-1][0] == ["rm", "--force", name]
    assert error_records(caplog) == []


def test_a_removal_failure_after_a_failed_start_is_logged_and_the_start_failure_is_kept(monkeypatch, docker, caplog):
    failure = OSError("spawn failed")
    monkeypatch.setattr(evaluator_check, "attach_worker", lambda name, log: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(evaluator_check, "remove_worker", lambda name: (_ for _ in ()).throw(DockerFailed(f"docker rm --force {name} exited 1\ndaemon gone")))

    with pytest.raises(OSError) as info:
        REAL_START(["register"])

    assert info.value is failure
    assert [(r.stage, "daemon gone" in r.getMessage()) for r in error_records(caplog)] == [("evaluator_release", True)]


def test_a_failure_before_create_removes_nothing(monkeypatch, docker):
    monkeypatch.setattr(evaluator_check, "worker_key", lambda args: (_ for _ in ()).throw(MissingCredentials("no key")))
    with pytest.raises(MissingCredentials):
        REAL_START(["register"])
    assert docker.calls == []


TASK_SLOT = "yasp-launcher-slot"
OTHER_WORKER = "gaia-worker-persistent"


def task_env(owner):
    """The environment of a task container the launcher started: the image's plus what the worker forwarded."""
    env = ["PATH=/opt/conda/bin:/usr/bin", "YASP_DATASET_CACHE_DIR=/mnt/dataset-cache"]
    return env + ([f"KRONOS_EVALUATOR={owner}"] if owner else [])


@pytest.fixture
def daemon(monkeypatch, docker):
    """The containers of a fake daemon, name -> environment, behind the recorded `inspect`, `stop`, and `rm`.

    Besides this check's worker, a host may hold someone else's persistent worker and the launcher's task
    slot, which every worker on the machine shares. `inspect` answers for a name in the table and fails as
    Docker does for one that is not; `stop` ends the attached client; `rm` drops the name.
    """
    containers = {OTHER_WORKER: ["YASP_API_KEY=theirs", "YASP_LAUNCH_DOCKER_FORWARD_ENV=YASP_DATASET_CACHE_DIR"]}
    recorded = evaluator_check.run_docker

    def run_docker(args, **kwargs):
        name = args[-1]
        if args[0] == "inspect":
            docker.calls.append((args, kwargs))
            if name not in containers:
                raise DockerFailed(f"docker {' '.join(args)} exited 1\nError response from daemon: No such container: {name}")
            return json.dumps(containers[name])
        if args[0] == "stop":
            docker.client.code = 143
        elif args[0] == "rm":
            containers.pop(name, None)
        return recorded(args, **kwargs)

    monkeypatch.setattr(evaluator_check, "run_docker", run_docker)
    return containers


def named(calls, name):
    return [args[0] for args, _ in calls if args[-1] == name]


def test_stop_stops_the_worker_gracefully_then_removes_it_and_reaps_the_client(docker, daemon):
    """`docker stop` lets the worker cancel its task and tear down the task container; the removal follows."""
    evaluator = REAL_START(["register"])

    REAL_STOP(evaluator)

    after_start = docker.calls[4:]
    assert [args[0] for args, _ in after_start] == ["stop", "rm", "inspect"]
    stop, options = after_start[0]
    assert stop == ["stop", "--time", "30", evaluator.name]
    assert options["timeout"] > evaluator_check.STOP_GRACE_S
    assert after_start[1][0] == ["rm", "--force", evaluator.name]
    assert after_start[2][0][-1] == TASK_SLOT
    assert docker.client.waits == [evaluator_check.STOP_TIMEOUT_S]
    assert evaluator.log.closed


@pytest.mark.parametrize(
    "owner, removed",
    [
        pytest.param("this", True, id="this_workers_task_still_running"),
        pytest.param("kronos-eval-" + "0" * 32, False, id="another_checks_task"),
        pytest.param(None, False, id="task_without_a_marker"),
    ],
)
def test_stop_removes_the_task_slot_only_when_this_workers_task_holds_it(docker, daemon, caplog, owner, removed):
    """The slot is shared: after the worker is gone, its own task is removed, anyone else's is left alone."""
    evaluator = REAL_START(["register"])
    daemon[TASK_SLOT] = task_env(evaluator.name if owner == "this" else owner)

    REAL_STOP(evaluator)

    assert (TASK_SLOT not in daemon) is removed
    assert named(docker.calls, TASK_SLOT) == (["inspect", "rm"] if removed else ["inspect"])
    assert docker.calls[-1][0] == (["rm", "--force", TASK_SLOT] if removed else ["inspect", "--type", "container", "--format", "{{json .Config.Env}}", TASK_SLOT])
    assert OTHER_WORKER in daemon and named(docker.calls, OTHER_WORKER) == []
    assert evaluator.name not in daemon
    assert [r.stage for r in caplog.records if r.levelname == "WARNING"] == (["evaluator_release"] if removed else [])


def test_stop_leaves_a_free_slot_and_an_unrelated_worker_alone(docker, daemon):
    evaluator = REAL_START(["register"])

    REAL_STOP(evaluator)

    assert named(docker.calls, TASK_SLOT) == ["inspect"]
    assert OTHER_WORKER in daemon and named(docker.calls, OTHER_WORKER) == []
    assert evaluator.name not in daemon


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(DockerFailed("docker stop --time 30 kronos-eval-x exited 1\ndaemon gone"), id="stop_fails"),
        pytest.param(subprocess.TimeoutExpired(["docker", "stop"], 90), id="stop_hangs"),
    ],
)
def test_a_graceful_stop_that_fails_is_logged_and_the_worker_is_still_removed(monkeypatch, docker, daemon, caplog, failure):
    """The removal is the contract; a stop the daemon refuses or that hangs is logged and the force path follows."""
    with_daemon = evaluator_check.run_docker

    def run_docker(args, **kwargs):
        if args[0] == "stop":
            docker.calls.append((args, kwargs))
            raise failure
        return with_daemon(args, **kwargs)

    monkeypatch.setattr(evaluator_check, "run_docker", run_docker)
    evaluator = REAL_START(["register"])
    daemon[TASK_SLOT] = task_env(evaluator.name)
    docker.client.code = 137

    REAL_STOP(evaluator)

    assert [args[0] for args, _ in docker.calls[4:]] == ["stop", "rm", "inspect", "rm"]
    assert evaluator.name not in daemon and TASK_SLOT not in daemon
    assert [r.stage for r in caplog.records if r.levelname == "WARNING"] == ["evaluator_release", "evaluator_release"]
    assert error_records(caplog) == []


def test_stop_kills_an_attached_client_that_does_not_exit(monkeypatch, docker):
    killed = []
    evaluator = REAL_START(["register"])

    def killpg(pgid, signum):
        killed.append((pgid, signum))
        docker.client.code = -9

    monkeypatch.setattr(os, "killpg", killpg)

    REAL_STOP(evaluator)

    assert killed == [(docker.client.pid, signal.SIGKILL)]
    assert docker.client.waits == [evaluator_check.STOP_TIMEOUT_S, evaluator_check.STOP_TIMEOUT_S]
    assert evaluator.log.closed


def test_stop_tolerates_a_container_that_is_already_gone(monkeypatch, docker):
    """`docker stop` and `rm --force` of a missing container exit 1; the goal, no container, is met."""
    evaluator = REAL_START(["register"])
    docker.client.code = 0
    monkeypatch.setattr(evaluator_check, "run_docker", lambda args, **kwargs: (_ for _ in ()).throw(DockerFailed(f"docker {' '.join(args)} exited 1\nError response from daemon: No such container: {args[-1]}")))

    REAL_STOP(evaluator)

    assert evaluator.log.closed


def test_a_stop_whose_removal_fails_still_reaps_the_client_and_reports_the_container(monkeypatch, docker):
    evaluator = REAL_START(["register"])
    docker.client.code = 0
    monkeypatch.setattr(evaluator_check, "run_docker", lambda args, **kwargs: (_ for _ in ()).throw(DockerFailed(f"docker {' '.join(args)} exited 1\ndaemon gone")))

    with pytest.raises(DockerFailed, match=f"rm --force {evaluator.name}"):
        REAL_STOP(evaluator)

    assert docker.client.waits == [evaluator_check.STOP_TIMEOUT_S]
    assert evaluator.log.closed


def test_the_host_depends_on_the_toolkit_and_no_standalone_evaluator_package():
    """The evaluator is the `yasp-eval` executable inside the official worker image, never a host package."""
    for name in ("pyproject.toml", "requirements.txt"):
        text = (REPO / name).read_text()
        assert "yasp-agent[gaia]" in text
        assert "yasp-compile-eval" not in text
    assert evaluator_check.EVALUATOR == "yasp-eval"

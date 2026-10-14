"""The evaluator check: before kernelgen, the host makes sure an evaluator for the config's gpu and eval_routing is registered.

`yasp-toolkit eval workers` lists the evaluators and `yasp-eval register` starts one, so both are
replaced by autouse fixtures (`toolkit_stub`, `evaluators`); the tests here cover the two nodes
around them, the listing parser against the toolkit's table, the real process helpers against
shell scripts, and that only an evaluator the check started is ever stopped.
"""

import os
import signal
import subprocess
import time

import pytest

from conftest import CONFIG, evaluator_listing, fake_toolkit
from kronos_edge import evaluator_check, kernelgen_check
from kronos_edge.workflow import main
from test_eager_check import CONV_RELU, TYPO, X, error_records, run, write_model, write_npz
from test_kernelgen_check import INVALID_TOKEN_RESPONSE, MISSING_TOKEN_PANEL, eval_fails, failing_toolkit, option

# Captured at import, before the autouse fixture replaces them with the recorder.
REAL_START = evaluator_check.start_evaluator
REAL_STOP = evaluator_check.stop_evaluator

GPU, ROUTING = CONFIG["gpu"], CONFIG["eval_routing"]

NVCC_MISSING = "Build toolchain incomplete — cannot compile candidate kernels:\n  - missing nvcc (CUDA toolkit)\n"


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
        pytest.param((1, NVCC_MISSING), "evaluator_start", "nvcc", id="toolchain_incomplete"),
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
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluators.stopped == evaluators.started and len(evaluators.started) == 1
    assert ["kernelgen", "compile"] not in commands(calls)


def test_a_missing_yasp_eval_command_is_a_stage(tmp_path, monkeypatch, caplog, env_files, good_model, evaluators):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, evaluators))
    monkeypatch.setattr(
        evaluator_check, "start_evaluator",
        lambda args: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-eval")),
    )

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_command"
    assert "yasp-compile-eval" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_command"]
    assert evaluators.stopped == [] and commands(calls) == [["eval", "workers"]]


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


def write_script(tmp_path, body):
    script = tmp_path / "fake-yasp-eval"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


def test_real_start_and_stop_run_the_command_capture_its_output_and_terminate_it(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator_check, "EVALUATOR", str(write_script(tmp_path, "echo registering $1 $2 $3\necho oops >&2\nsleep 30")))

    evaluator = REAL_START(["register", "--tag", "routing=x"])
    try:
        assert evaluator.exit_code() is None
        deadline = time.monotonic() + 10
        while "oops" not in evaluator.output() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "registering register --tag routing=x" in evaluator.output()
        assert "oops" in evaluator.output()  # stderr is captured with stdout
        REAL_STOP(evaluator)
        assert evaluator.exit_code() is not None
        assert evaluator.process.returncode < 0  # ended by our signal, not on its own
    finally:
        if evaluator.exit_code() is None:
            evaluator.process.kill()


def test_real_stop_kills_a_command_that_ignores_the_terminate_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator_check, "EVALUATOR", str(write_script(tmp_path, "trap '' TERM\nwhile true; do sleep 1; done")))
    monkeypatch.setattr(evaluator_check, "STOP_TIMEOUT_S", 0.5)

    evaluator = REAL_START(["register"])
    try:
        REAL_STOP(evaluator)
        assert evaluator.exit_code() is not None
    finally:
        if evaluator.exit_code() is None:
            evaluator.process.kill()


def test_real_start_reports_a_missing_command(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator_check, "EVALUATOR", str(tmp_path / "no-such-yasp-eval"))

    with pytest.raises(FileNotFoundError, match="no-such-yasp-eval"):
        REAL_START(["register"])


def test_real_start_does_not_write_to_this_processs_stdout(tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(evaluator_check, "EVALUATOR", str(write_script(tmp_path, "echo noise\nsleep 30")))

    evaluator = REAL_START(["register"])
    REAL_STOP(evaluator)

    assert capfd.readouterr().out == ""

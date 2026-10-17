"""The evaluator check: before kernelgen, the host confirms a Gaia worker for the config's gpu and eval_routing is registered.

The worker is the user's persistent deployment (the Gaia Docker guide plus a routing tag); the
check only lists the registered evaluators with `yasp-toolkit eval workers` and never starts,
stops, or removes a worker. The toolkit is replaced by the autouse `toolkit_stub`; the tests
cover the node around it and the listing parser against the toolkit's table.
"""

import subprocess

import pytest

from conftest import CONFIG, evaluator_listing, fake_toolkit
from kronos_edge import evaluator_check, kernelgen_check
from test_eager_check import TYPO, X, error_records, run, write_model, write_npz
from test_kernelgen_check import failing_toolkit, option

GPU, ROUTING = CONFIG["gpu"], CONFIG["eval_routing"]


def listing_toolkit(calls, rows):
    """A toolkit whose `eval workers` lists `rows`; every command is appended to `calls`."""

    def toolkit(args, timeout=None):
        calls.append(args)
        if args[:2] != ["eval", "workers"]:
            return fake_toolkit(args)
        return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=evaluator_listing(*rows), stderr="")

    return toolkit


def commands(calls):
    return [args[:2] for args in calls]


def test_a_registered_evaluator_lets_kernelgen_run(tmp_path, monkeypatch, env_files, good_model):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, [(GPU, ROUTING)]))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert commands(calls) == [["kernelgen", "export"], ["eval", "workers"], ["kernelgen", "compile"], ["eval", "inspect"]]
    assert option(calls[1], "--tag") == f"routing={ROUTING}"


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([], id="nothing_registered"),
        pytest.param([("NVIDIA GeForce RTX 4090", ROUTING)], id="other_gpu"),
        pytest.param([(GPU, "someone-else")], id="other_routing"),
        pytest.param([(GPU, "—")], id="no_routing"),
        pytest.param([(GPU.lower(), ROUTING)], id="gpu_differs_in_case"),
        pytest.param([("NVIDIA GeForce RTX 4090", ROUTING), (GPU, "someone-else")], id="each_half_matches"),
    ],
)
def test_without_an_exact_match_the_check_stops_before_kernelgen(tmp_path, monkeypatch, caplog, env_files, good_model, rows):
    """Matching is exact on gpu and routing; a missing worker is the user's to deploy, so nothing is submitted."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, rows))

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "evaluator_missing"
    assert f"routing={ROUTING}" in state["customer_message"]
    assert "README.md" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["evaluator_missing"]
    assert commands(calls) == [["kernelgen", "export"], ["eval", "workers"]]
    assert not (tmp_path / ".artifact" / "kernel.py").exists()


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(
            lambda args, timeout=None: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-toolkit")),
            "kernelgen_toolkit", "yasp-agent", id="toolkit_not_installed",
        ),
        pytest.param(failing_toolkit(1, stderr="Error: --tag expects KEY=VALUE\n"), "evaluator_inventory", "KEY=VALUE", id="exit_nonzero"),
        pytest.param(failing_toolkit(0, stderr="╭─ GET https://compile.yasp.ai/eval/evaluators ─╮\n│ HTTP 503 │\n╰──╯\n"), "evaluator_inventory", "503", id="exit_zero_without_a_listing"),
        pytest.param(
            lambda args, timeout=None: (_ for _ in ()).throw(subprocess.TimeoutExpired(["yasp-toolkit", *args], timeout)),
            "evaluator_inventory", "timed out", id="hung_listing",
        ),
    ],
)
def test_listing_failures_are_stages_and_submit_nothing(tmp_path, monkeypatch, caplog, env_files, good_model, toolkit, stage, fragment):
    calls = []

    def recording(args, timeout=None):
        calls.append((args, timeout))
        if args[:2] == ["kernelgen", "export"]:  # the step before the listing is not under test here
            return fake_toolkit(args)
        return toolkit(args, timeout=timeout)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert [(args[:2], timeout) for args, timeout in calls] == [(["kernelgen", "export"], None), (["eval", "workers"], evaluator_check.LISTING_TIMEOUT_S)]
    assert not (tmp_path / ".artifact" / "kernel.py").exists()


def test_nothing_is_listed_when_the_model_check_fails(tmp_path, monkeypatch, env_files):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", listing_toolkit(calls, [(GPU, ROUTING)]))

    state = run(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert state["error"] == "forward"
    assert calls == []


def test_the_check_manages_no_worker_lifecycle():
    """The worker is deployed once by the user; the check has nothing that creates or removes one."""
    for name in ("start_evaluator", "stop_evaluator", "release_evaluator", "release_active_evaluator", "ensure_evaluator"):
        assert not hasattr(evaluator_check, name), name


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

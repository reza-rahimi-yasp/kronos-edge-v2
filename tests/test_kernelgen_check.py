"""The kernelgen check: on the host, after the container, `yasp-toolkit` optimizes the model and scores the kernel on this machine.

yasp-toolkit needs a Yasp API key and the hosted service, so `kernelgen_check.run_toolkit` is
replaced by an autouse fixture; the tests here cover the two nodes around it, the real
subprocess helper against a missing command, and the config setting that pins the evaluator.
"""

import json
import subprocess
from pathlib import Path

import pytest

from conftest import CONFIG, FAKE_TASK_LINE, PASSING_RESULT, fake_toolkit, write_yaml
from kronos_edge import artifacts, container, kernelgen_check
from kronos_edge.workflow import run_check
from test_container import BENCHMARK
from test_eager_check import CONV_RELU, TYPO, X, error_records, run, write_model, write_npz

# Captured at import, before the autouse fixture replaces it with the stub.
REAL_RUN_TOOLKIT = kernelgen_check.run_toolkit

MISSING_TOKEN_PANEL = (
    "╭─ Missing authentication token ─╮\n"
    "│ No token found in the OS keyring (and `YASP_API_KEY` is unset). │\n"
    "╰─────────────────────────────────╯\n"
)

INVALID_TOKEN_RESPONSE = (
    "POST https://compile.yasp.ai/api/token?audience=kernelgen-api HTTP 401\n"
    '{"code": "ERROR_UNAUTHORIZED", "message": "Invalid authentication credentials."}'
)


def recording_toolkit(calls, tmp_path):
    """The stub, plus a record of each command and of what the artifact directory held when it ran."""

    def record(args):
        calls.append((args, sorted(f.name for f in (tmp_path / ".artifact").iterdir())))
        return fake_toolkit(args)

    return record


def failing_toolkit(code, stderr="", stdout=""):
    return lambda args: subprocess.CompletedProcess(["yasp-toolkit", *args], code, stdout=stdout, stderr=stderr)


def run_with_config(tmp_path, env_files, **settings):
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, **settings})
    return run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)


def option(args, name):
    return args[args.index(name) + 1]


def test_kernelgen_runs_after_the_container_and_evaluation_after_kernelgen(tmp_path, monkeypatch, env_files, good_model):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert [args[:2] for args, _ in calls] == [["kernelgen", "compile"], ["eval", "compare"]]
    compile_args, seen_before_compile = calls[0]
    eval_args, seen_before_eval = calls[1]
    # kernelgen starts only once the container has produced the reference and the TensorRT engine.
    assert {"model.pt2", "model.engine", "model.onnx", "model_simplified.onnx"} <= set(seen_before_compile)
    assert "kernel.py" not in seen_before_compile
    # The evaluation scores the kernel kernelgen wrote, against the same reference.
    assert "kernel.py" in seen_before_eval and "kernelgen_report.json" in seen_before_eval
    assert option(eval_args, "--implementation") == option(compile_args, "--output") == str(tmp_path / ".artifact" / "kernel.py")
    assert option(eval_args, "--reference") == option(compile_args, "--reference") == str(tmp_path / ".artifact" / "model.pt2")
    assert option(compile_args, "--source") == option(eval_args, "--source") == str(good_model[0])
    assert option(compile_args, "--gpu-name") == option(eval_args, "--gpu-name") == CONFIG["gpu"]
    assert option(compile_args, "--report") == str(tmp_path / ".artifact" / "kernelgen_report.json")
    assert option(eval_args, "--output") == str(tmp_path / ".artifact" / "kernelgen_eval.json")
    assert "--no-progress" in compile_args and "--no-progress" in eval_args
    assert state["kernel_path"] == str(tmp_path / ".artifact" / "kernel.py")


def test_success_records_the_speedup_and_names_it(tmp_path, caplog, env_files, good_model, capsys):
    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert state["kernelgen_speedup"] == 2.0
    assert state["kernelgen_result"] == PASSING_RESULT
    assert "kernelgen optimization" in state["customer_message"]
    assert "kernelgen evaluation (speedup 2.00x) succeeded" in state["customer_message"]
    assert sorted(f.name for f in (tmp_path / ".artifact").iterdir()) == [
        "benchmark.json", "kernel.py", "kernelgen_eval.json", "kernelgen_report.json",
        "model.engine", "model.onnx", "model.pt2", "model_simplified.onnx",
    ]
    assert error_records(caplog) == []
    assert capsys.readouterr().out == ""  # the toolkit's output goes to the log, never to stdout


def test_eval_routing_pins_this_machines_evaluator_for_both_commands(tmp_path, monkeypatch, env_files):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run_with_config(tmp_path, env_files, eval_routing="reza-fractal")

    assert state["outcome"] == "success"
    (compile_args, _), (eval_args, _) = calls
    assert option(compile_args, "--eval-routing") == "reza-fractal"
    assert option(eval_args, "--routing") == "reza-fractal"


def test_a_config_without_eval_routing_is_rejected_before_any_toolkit_job(tmp_path, caplog, monkeypatch, env_files):
    """Without the key, any evaluator advertising the GPU could take the job; the check must not submit one."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {k: v for k, v in CONFIG.items() if k != "eval_routing"})

    state = run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)

    assert state["outcome"] == "failure"
    assert state["error"] == "validate_config"
    assert "eval_routing" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["validate_config"]
    assert calls == []
    assert not (tmp_path / ".artifact").exists()


@pytest.mark.parametrize(
    "data, fragment",
    [
        pytest.param({**CONFIG, "eval_routing": ""}, "eval_routing", id="empty"),
        pytest.param({**CONFIG, "eval_routing": 5}, "eval_routing", id="not_a_string"),
    ],
)
def test_invalid_eval_routing_is_a_schema_violation(tmp_path, env_files, good_model, data, fragment):
    _, environments = env_files
    config = write_yaml(tmp_path / "bad.yaml", data)

    state = run_check(config, *good_model, environments_path=environments)

    assert state["error"] == "validate_config"
    assert fragment in state["customer_message"]


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(
            lambda args: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-toolkit")),
            "kernelgen_toolkit", "yasp-agent", id="toolkit_not_installed",
        ),
        pytest.param(failing_toolkit(1, stderr=MISSING_TOKEN_PANEL), "kernelgen_credentials", "YASP_API_KEY", id="no_api_key"),
        pytest.param(failing_toolkit(0, stderr=INVALID_TOKEN_RESPONSE), "kernelgen_credentials", "not accepted", id="invalid_api_key"),
        pytest.param(
            failing_toolkit(1, stdout=FAKE_TASK_LINE.replace("completed", "failed"), stderr="╭─ Failure reason ─╮\n│ worker exploded │\n╰──╯\n"),
            "kernelgen_compile", "worker exploded", id="job_failed",
        ),
        pytest.param(failing_toolkit(0, stderr="╭─ POST https://compile.yasp.ai/kernelgen/v1/kernels ─╮\n│ HTTP 503 │\n╰──╯\n"), "kernelgen_compile", "503", id="exit_zero_without_a_kernel"),
    ],
)
def test_kernelgen_failures_are_stages_and_skip_the_evaluation(tmp_path, caplog, monkeypatch, env_files, toolkit, stage, fragment):
    evaluations = []

    def dispatch(args):
        if args[:2] == ["eval", "compare"]:
            evaluations.append(args)
        return toolkit(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", dispatch)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert evaluations == []
    assert "kernelgen_eval.json" not in [f.name for f in (tmp_path / ".artifact").iterdir()]


def test_real_run_toolkit_reports_a_missing_command_as_a_stage(tmp_path, monkeypatch, caplog, env_files):
    """The real subprocess helper, against a command that does not exist, must become the toolkit stage."""
    monkeypatch.setattr(kernelgen_check, "run_toolkit", REAL_RUN_TOOLKIT)
    monkeypatch.setattr(kernelgen_check, "TOOLKIT", str(tmp_path / "no-such-yasp-toolkit"))

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == "kernelgen_toolkit"
    assert "no-such-yasp-toolkit" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["kernelgen_toolkit"]


def test_real_run_toolkit_captures_output_and_exit_code(tmp_path, monkeypatch):
    script = tmp_path / "fake-toolkit"
    script.write_text("#!/bin/sh\necho out-$1\necho err-$2 >&2\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.setattr(kernelgen_check, "TOOLKIT", str(script))

    result = REAL_RUN_TOOLKIT(["kernelgen", "compile"])

    assert result.returncode == 3
    assert result.stdout == "out-kernelgen\n"
    assert result.stderr == "err-compile\n"


def failing_result(**overrides):
    checks = {**PASSING_RESULT["correctness_checks"], "all_checks_passed": False, **overrides}
    return {**PASSING_RESULT, "correctness_checks": checks}


def eval_writes(payload):
    """A toolkit whose `eval compare` writes `payload` (a string is written verbatim) instead of a passing result."""

    def toolkit(args):
        if args[:2] == ["eval", "compare"]:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            Path(option(args, "--output")).write_text(text)
            return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=FAKE_TASK_LINE, stderr="")
        return fake_toolkit(args)

    return toolkit


def eval_fails(code, stderr):
    def toolkit(args):
        if args[:2] == ["eval", "compare"]:
            return subprocess.CompletedProcess(["yasp-toolkit", *args], code, stdout="", stderr=stderr)
        return fake_toolkit(args)

    return toolkit


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(eval_fails(1, "╭─ Failure reason ─╮\n│ evaluator crashed │\n╰──╯\n"), "kernelgen_eval", "evaluator crashed", id="eval_failed"),
        pytest.param(eval_fails(0, "╭─ POST https://compile.yasp.ai/eval/v1/evals ─╮\n│ HTTP 503 │\n╰──╯\n"), "kernelgen_eval", "503", id="eval_exit_zero_without_results"),
        pytest.param(eval_fails(1, MISSING_TOKEN_PANEL), "kernelgen_credentials", "YASP_API_KEY", id="no_api_key"),
        pytest.param(eval_fails(0, INVALID_TOKEN_RESPONSE), "kernelgen_credentials", "not accepted", id="invalid_api_key"),
        pytest.param(eval_writes("not json"), "kernelgen_eval_results", "kernelgen_eval.json", id="results_not_json"),
        pytest.param(eval_writes({"kernel_1": PASSING_RESULT}), "kernelgen_eval_results", "opt_1", id="results_without_opt_1"),
        pytest.param(eval_writes({"opt_1": {k: v for k, v in PASSING_RESULT.items() if k != "speedup"}}), "kernelgen_eval_results", "speedup", id="results_without_speedup"),
        pytest.param(eval_writes({"opt_1": {"speedup": 1.5, "stage": "build"}}), "kernelgen_eval_incorrect", "correctness_checks", id="results_without_checks"),
        pytest.param(eval_writes({"opt_1": failing_result(all_output_values_match=False)}), "kernelgen_eval_incorrect", "all_output_values_match", id="wrong_values"),
        pytest.param(eval_writes({"opt_1": failing_result(all_shapes_match=False)}), "kernelgen_eval_incorrect", "all_shapes_match", id="wrong_shapes"),
    ],
)
def test_evaluation_failures_are_stages(tmp_path, caplog, monkeypatch, env_files, toolkit, stage, fragment):
    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert "kernelgen_speedup" not in state
    assert (tmp_path / ".artifact" / "kernel.py").is_file()  # the kernel from the earlier node is kept


def test_kernelgen_is_skipped_when_the_container_fails(tmp_path, monkeypatch, env_files):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert state["error"] == "forward"
    assert calls == []


@pytest.mark.parametrize(
    "produced, stage, missing",
    [
        pytest.param([], "kernelgen_reference", "model.pt2", id="no_reference"),
        pytest.param(["model.pt2"], "kernelgen_engine", "model.engine", id="no_engine"),
    ],
)
def test_kernelgen_requires_the_reference_and_the_tensorrt_output(tmp_path, caplog, monkeypatch, env_files, good_model, produced, stage, missing):
    """A container that reports success but left no reference or engine behind cannot be optimized."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    def fake_container(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies):
        for name in produced:
            (Path(artifacts_dir) / name).write_bytes(b"x")
        (Path(report_dir) / container.REPORT_FILE).write_text(json.dumps({"error": None, "benchmark": BENCHMARK}))
        return 0

    monkeypatch.setattr(container, "run_container", fake_container)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert missing in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert calls == []


def test_kernelgen_files_are_managed_artifacts():
    assert {artifacts.PT2_FILE, artifacts.KERNEL_FILE, artifacts.KERNELGEN_REPORT_FILE, artifacts.KERNELGEN_EVAL_FILE} <= set(artifacts.MANAGED_FILES)
    assert kernelgen_check.KERNEL_FILE == artifacts.KERNEL_FILE

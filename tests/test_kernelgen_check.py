"""The kernelgen checks: after the model check, `yasp-toolkit kernelgen export` writes the reference, then
`yasp-toolkit kernelgen compile` generates the kernel and `eval inspect` shows its report.

yasp-toolkit needs a Yasp API key and the hosted service, so `kernelgen_check.run_toolkit` is
replaced by an autouse fixture; the tests here cover the nodes around it, the real subprocess
helper against a missing command, and the config setting that pins the evaluator.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import CONFIG, FAKE_INSPECT, FAKE_REFERENCE, FAKE_TASK_LINE, fake_toolkit, run_directory, write_yaml
from kronos_edge import artifacts, kernelgen_check
from kronos_edge.workflow import run_check
from test_eager_check import CONV_RELU, TYPO, X, error_records, run, write_model, write_npz

# Captured at import, before the autouse fixture replaces it with the stub.
REAL_RUN_TOOLKIT = kernelgen_check.run_toolkit

def recording_toolkit(calls, tmp_path):
    """The stub, plus a record of each command and of what the artifact directory held when it ran."""

    def record(args, timeout=None):
        calls.append((args, sorted(f.name for f in run_directory(tmp_path / ".artifact").iterdir())))
        return fake_toolkit(args)

    return record


def failing_toolkit(code, stderr="", stdout=""):
    return lambda args, timeout=None: subprocess.CompletedProcess(["yasp-toolkit", *args], code, stdout=stdout, stderr=stderr)


def run_with_config(tmp_path, env_files, **settings):
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, **settings})
    return run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)


def option(args, name):
    return args[args.index(name) + 1]


def test_kernelgen_runs_after_the_export_and_its_report_is_inspected(tmp_path, monkeypatch, env_files, good_model):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    # The documented order: export, the evaluator listing (test_evaluator_check), generate, inspect.
    assert [args[:2] for args, _ in calls] == [["kernelgen", "export"], ["eval", "workers"], ["kernelgen", "compile"], ["eval", "inspect"]]
    compile_args, seen_before_compile = calls[2]
    inspect_args, seen_before_inspect = calls[3]
    # kernelgen starts only once the export has written the reference and the model check the TensorRT engine.
    assert {"model.pt2", "model.engine", "model.onnx", "model_simplified.onnx"} <= set(seen_before_compile)
    assert "kernel.py" not in seen_before_compile
    art = run_directory(tmp_path / ".artifact")
    assert option(compile_args, "--reference") == str(art / "model.pt2")
    assert option(compile_args, "--gpu-name") == CONFIG["gpu"]
    assert option(compile_args, "--output") == str(art / "kernel.py")
    assert option(compile_args, "--report") == str(art / "kernelgen_report.json")
    assert "--no-progress" in compile_args
    # The documented starting workflow: the reference alone, no model source uploaded to the service.
    assert "--source" not in compile_args
    # The report the compile wrote is what gets inspected.
    assert inspect_args == ["eval", "inspect", "--file", str(art / "kernelgen_report.json")]
    assert {"kernel.py", "kernelgen_report.json"} <= set(seen_before_inspect)
    assert state["kernel_path"] == str(art / "kernel.py")
    assert f"the kernel is {art / 'kernel.py'}." in state["customer_message"]


def test_success_shows_the_inspected_report(tmp_path, caplog, env_files, good_model, capsys):
    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    assert state["kernelgen_report"] == FAKE_INSPECT.strip()
    assert "kernel.py" in state["customer_message"]
    assert FAKE_INSPECT.strip() in state["customer_message"]
    assert sorted(f.name for f in run_directory(tmp_path / ".artifact").iterdir()) == [
        "benchmark.json", "kernel.py", "kernelgen_report.json",
        "model.engine", "model.onnx", "model.pt2", "model_simplified.onnx",
    ]
    assert error_records(caplog) == []
    assert capsys.readouterr().out == ""  # the toolkit's output goes to the log and the state, never to stdout


def test_eval_routing_pins_the_evaluator_for_the_listing_and_the_compile(tmp_path, monkeypatch, env_files):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run_with_config(tmp_path, env_files, eval_routing="reza-fractal")

    assert state["outcome"] == "success"
    _, (workers_args, _), (compile_args, _), _ = calls
    assert option(workers_args, "--tag") == "routing=reza-fractal"
    assert option(compile_args, "--eval-routing") == "reza-fractal"


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


def compile_writes(*names):
    """A compile that exits 0 but writes only the files in `names` (of kernel.py and the report)."""

    def toolkit(args, timeout=None):
        if args[:2] == ["kernelgen", "compile"]:
            if "kernel.py" in names:
                Path(option(args, "--output")).write_text("kernel")
            if "kernelgen_report.json" in names:
                Path(option(args, "--report")).write_text("{}")
            return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=FAKE_TASK_LINE, stderr="")
        return fake_toolkit(args)

    return toolkit


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(
            lambda args, timeout=None: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-toolkit")),
            "kernelgen_toolkit", "yasp-agent", id="toolkit_not_installed",
        ),
        pytest.param(
            failing_toolkit(1, stdout=FAKE_TASK_LINE.replace("completed", "failed"), stderr="╭─ Failure reason ─╮\n│ worker exploded │\n╰──╯\n"),
            "kernelgen_compile", "worker exploded", id="job_failed",
        ),
        pytest.param(
            failing_toolkit(1, stderr="No token found in the OS keyring (and `YASP_API_KEY` is unset).\n"),
            "kernelgen_compile", "YASP_API_KEY", id="no_api_key_is_shown_as_the_toolkit_said_it",
        ),
        pytest.param(failing_toolkit(0, stderr="╭─ POST https://compile.yasp.ai/kernelgen/v1/kernels ─╮\n│ HTTP 503 │\n╰──╯\n"), "kernelgen_compile", "503", id="exit_zero_without_a_kernel"),
        pytest.param(compile_writes("kernelgen_report.json"), "kernelgen_compile", "kernel.py", id="report_without_a_kernel"),
        pytest.param(compile_writes("kernel.py"), "kernelgen_compile", "kernelgen_report.json", id="kernel_without_a_report"),
    ],
)
def test_kernelgen_failures_are_stages_and_skip_the_inspection(tmp_path, caplog, monkeypatch, env_files, toolkit, stage, fragment):
    inspections = []

    def dispatch(args, timeout=None):
        if args[:2] in (["kernelgen", "export"], ["eval", "workers"]):  # the steps before kernelgen are not under test here
            return fake_toolkit(args)
        if args[:2] == ["eval", "inspect"]:
            inspections.append(args)
        return toolkit(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", dispatch)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert inspections == []


@pytest.mark.parametrize(
    "code, stderr",
    [
        pytest.param(1, "unrecognized report\n", id="inspect_failed"),
        pytest.param(2, "No such option: --file\n", id="inspect_usage_error"),
    ],
)
def test_an_inspection_failure_is_a_stage_and_keeps_the_kernel(tmp_path, caplog, monkeypatch, env_files, code, stderr):
    def toolkit(args, timeout=None):
        if args[:2] == ["eval", "inspect"]:
            return subprocess.CompletedProcess(["yasp-toolkit", *args], code, stdout="", stderr=stderr)
        return fake_toolkit(args)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", toolkit)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == "kernelgen_inspect"
    assert stderr.strip() in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["kernelgen_inspect"]
    art = run_directory(tmp_path / ".artifact")
    assert (art / "kernel.py").is_file() and (art / "kernelgen_report.json").is_file()


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


def test_real_run_toolkit_kills_a_command_that_outlives_its_timeout(tmp_path, monkeypatch):
    script = tmp_path / "fake-toolkit"
    script.write_text("#!/bin/sh\nexec sleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(kernelgen_check, "TOOLKIT", str(script))
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        REAL_RUN_TOOLKIT(["eval", "workers"], timeout=0.2)

    assert time.monotonic() - started < 5  # killed at the timeout, not awaited


CREDENTIALS = ("YASP_API_KEY", "YASP_API_TOKEN", "REGISTRY_KEY")


def test_the_real_export_hands_the_model_no_yasp_credentials(tmp_path, monkeypatch):
    """`kernelgen export` imports the user's model.py in the toolkit's process, so the real helper runs it without the
    Yasp credentials and with the keyring disabled; the export itself needs no key."""
    for name in CREDENTIALS:
        monkeypatch.setenv(name, "fake-secret-sentinel")
    script = tmp_path / "fake-toolkit"
    script.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        "args = sys.argv[1:]\n"
        "runpy.run_path(args[args.index('--input') + 1])\n"
        "open(args[args.index('--output') + 1], 'wb').write(b'reference')\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(kernelgen_check, "TOOLKIT", str(script))
    monkeypatch.setattr(kernelgen_check, "run_toolkit", REAL_RUN_TOOLKIT)
    seen = tmp_path / "seen.json"
    model = tmp_path / "model.py"
    model.write_text(
        "import json, os\n"
        f"open({str(seen)!r}, 'w').write(json.dumps({{k: os.environ.get(k) for k in {[*CREDENTIALS, 'PYTHON_KEYRING_BACKEND']!r}}}))\n"
    )
    artifacts_dir = tmp_path / ".artifact"
    artifacts_dir.mkdir()

    state = kernelgen_check.export_reference({"run_path": str(artifacts_dir), "model_path": str(model)})

    assert state == {}
    assert (artifacts_dir / "model.pt2").read_bytes() == b"reference"
    assert json.loads(seen.read_text()) == {**dict.fromkeys(CREDENTIALS), "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}


@pytest.mark.parametrize(
    "args, authenticated",
    [
        (["kernelgen", "export", "--input", "model.py"], False),
        (["eval", "workers", "--tag", "routing=r"], True),
        (["kernelgen", "compile", "--reference", "model.pt2"], True),
        (["eval", "inspect", "--file", "kernelgen_report.json"], True),
    ],
)
def test_only_the_service_commands_get_the_yasp_credentials(tmp_path, monkeypatch, args, authenticated):
    for name in CREDENTIALS:
        monkeypatch.setenv(name, f"fake-{name}")
    script = tmp_path / "fake-toolkit"
    script.write_text("#!/bin/sh\nenv\n")
    script.chmod(0o755)
    monkeypatch.setattr(kernelgen_check, "TOOLKIT", str(script))

    env = dict(line.split("=", 1) for line in REAL_RUN_TOOLKIT(args).stdout.splitlines() if "=" in line)

    assert {k: env.get(k) for k in CREDENTIALS} == {k: f"fake-{k}" if authenticated else None for k in CREDENTIALS}
    assert env["COLUMNS"] == "200"


def test_kernelgen_is_skipped_when_the_model_check_fails(tmp_path, monkeypatch, env_files):
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
def test_kernelgen_requires_the_reference_and_the_tensorrt_output(tmp_path, caplog, monkeypatch, produced, stage, missing):
    """Nothing is submitted to kernelgen without the export's reference and the model check's engine."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: calls.append(args))
    art = tmp_path / ".artifact"
    art.mkdir()
    for name in produced:
        (art / name).write_bytes(b"x")

    update = kernelgen_check.kernelgen({"run_path": str(art), "config": None})

    assert update["error"] == stage
    assert missing in update["error_detail"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert calls == []


# ---------- The reference export: the documented `yasp-toolkit kernelgen export` on model.py, as a subprocess, output captured.


def test_export_runs_after_the_model_check_on_the_users_model(tmp_path, monkeypatch, env_files, good_model):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run(env_files, *good_model)

    assert state["outcome"] == "success"
    export_args, present = calls[0]
    art = run_directory(tmp_path / ".artifact")
    assert export_args == ["kernelgen", "export", "--input", str(good_model[0].resolve()), "--output", str(art / "model.pt2")]
    assert {"model.engine", "benchmark.json"} <= set(present)  # the model check ran first
    assert "model.pt2" not in present
    assert (art / "model.pt2").read_bytes() == FAKE_REFERENCE


def test_the_export_is_not_told_where_to_run(tmp_path, monkeypatch, env_files, good_model):
    """The toolkit traces the reference however its own torch is set up: no device, CPU-only, or GPU-only setting."""
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    run(env_files, *good_model)

    export_args, _ = calls[0]
    assert [arg for arg in export_args if arg.startswith("--")] == ["--input", "--output"]


def exporting(code, output, writes=False):
    def toolkit(args, timeout=None):
        if args[:2] != ["kernelgen", "export"]:
            return fake_toolkit(args)
        if writes:
            Path(option(args, "--output")).write_bytes(b"x")
        return subprocess.CompletedProcess(["yasp-toolkit", *args], code, stdout=output, stderr="")

    return toolkit


@pytest.mark.parametrize(
    "toolkit, stage, fragment",
    [
        pytest.param(
            lambda args, timeout=None: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "yasp-toolkit")),
            "kernelgen_toolkit", "yasp-agent", id="toolkit_not_installed",
        ),
        pytest.param(
            exporting(1, "torch._dynamo.exc.UserError: data-dependent branch\n"),
            "export_reference", "data-dependent", id="export_failed",
        ),
        pytest.param(
            exporting(1, "ModuleNotFoundError: No module named 'timm'\n"),
            "export_reference", "timm", id="model_import_fails_in_the_toolkits_environment",
        ),
        pytest.param(exporting(0, "  wrote      nothing\n"), "export_reference", "model.pt2", id="exit_zero_without_a_reference"),
    ],
)
def test_export_failures_are_stages_and_stop_before_any_other_toolkit_call(tmp_path, caplog, monkeypatch, env_files, good_model, capsys, toolkit, stage, fragment):
    calls = []

    def recording(args, timeout=None):
        calls.append(args)
        return toolkit(args, timeout=timeout)

    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording)

    state = run(env_files, *good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert [args[:2] for args in calls] == [["kernelgen", "export"]]
    assert capsys.readouterr().out == ""  # the export's output, the user's prints included, never reaches stdout


def test_export_is_skipped_when_the_model_check_fails(tmp_path, monkeypatch, env_files):
    calls = []
    monkeypatch.setattr(kernelgen_check, "run_toolkit", recording_toolkit(calls, tmp_path))

    state = run(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert state["error"] == "forward"
    assert calls == []


def test_kernelgen_files_are_managed_artifacts():
    assert {artifacts.PT2_FILE, artifacts.KERNEL_FILE, artifacts.KERNELGEN_REPORT_FILE} <= set(artifacts.MANAGED_FILES)
    assert not hasattr(artifacts, "KERNELGEN_EVAL_FILE")
    assert kernelgen_check.KERNEL_FILE == artifacts.KERNEL_FILE


def test_no_undocumented_result_parsing_or_evaluation_remains():
    """The outcome is read through the documented `eval inspect`, not private report fields or a second paid job."""
    for name in ("evaluate_kernelgen", "RESULT_KEY", "MissingCredentials", "MISSING_TOKEN", "UNAUTHORIZED"):
        assert not hasattr(kernelgen_check, name), name

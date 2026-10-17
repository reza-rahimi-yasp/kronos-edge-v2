"""The model check: the eager graph runs as a subprocess of this interpreter and reports through report.json.

An autouse fixture runs that graph in-process on CPU instead of spawning it; the tests here cover the
node around it, the report protocol, the command, and the real subprocess against the user's stdout.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import GOOD_MODEL
from kronos_edge import artifacts, eager_check, model_check, onnxsim_check

BENCHMARK = {
    "eager_ms": {"mean": 0.05, "median": 0.046, "p95": 0.06},
    "tensorrt_ms": {"mean": 0.016, "median": 0.015, "p95": 0.02},
    "speedup": 3.07,
    "metric": "cosine",
    "threshold": 0.99,
    "values": [0.9998],
}
from kronos_edge.workflow import run_check

TYPO_MODEL = GOOD_MODEL.replace("return self.conv(x)", "return self.convv(x)")

NOISY_MODEL = GOOD_MODEL.replace(
    "import torch\n",
    "import atexit\nimport sys\nimport torch\n\n"
    "sys.stdout.write('loading without newline')\n"
    "atexit.register(lambda: print('bye from atexit'))\n",
).replace("        return self.conv(x)", "        print('forward called', end='')\n        return self.conv(x)")

SRC = Path(__file__).resolve().parents[1] / "src"


def error_records(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR" and r.name.startswith("kronos_edge")]


def run(env_files, good_model):
    config, environments = env_files
    return run_check(config, *good_model, environments_path=environments)


def fake_process(monkeypatch, report=None, code=0):
    """Replace the subprocess with something that writes `report` (if given) and exits with `code`."""

    def fake(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        if report is not None:
            (Path(report_dir) / model_check.REPORT_FILE).write_text(report)
        return code

    monkeypatch.setattr(model_check, "run_process", fake)


def test_build_command_runs_the_eager_check_in_this_interpreter(tmp_path):
    model, npz, out, art = tmp_path / "m.py", tmp_path / "i.npz", tmp_path / "out", tmp_path / ".artifact"

    cmd = model_check.build_command(model, npz, out, art, "fp16", 0.99, "cuda", "DEBUG")

    assert cmd == [
        sys.executable, "-m", "kronos_edge.eager_check", str(model.resolve()), str(npz.resolve()),
        "--report", str(out.resolve() / "report.json"), "--artifacts", str(art.resolve()),
        "--precision", "fp16", "--threshold", "0.99", "--device", "cuda", "--log-level", "DEBUG",
    ]
    assert "docker" not in cmd and "--gpus" not in cmd


def test_the_subprocess_imports_this_package(monkeypatch):
    """The child must find the same kronos_edge whatever the working directory, so its source directory leads PYTHONPATH."""
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")

    env = model_check.process_environment()

    assert env["PYTHONPATH"].split(os.pathsep) == [str(SRC), "/somewhere/else"]
    monkeypatch.delenv("PYTHONPATH")
    assert model_check.process_environment()["PYTHONPATH"] == str(SRC)


def test_rerun_clears_managed_files_but_keeps_the_users_own(monkeypatch, tmp_path, env_files, good_model):
    """A second run in the same directory must not leave the first run's outputs next to the new ones."""
    config, environments = env_files
    model, npz = good_model
    art = tmp_path / ".artifact"
    first = run_check(config, model, npz, environments_path=environments, artifacts_path=art)
    assert first["outcome"] == "success"
    assert sorted(f.name for f in art.iterdir()) == [
        "benchmark.json", "kernel.py", "kernelgen_report.json",
        "model.engine", "model.onnx", "model.pt2", "model_simplified.onnx",
    ]
    (art / "notes.txt").write_text("mine")
    (art / "model.onnx.data").write_bytes(b"stale companion")

    def simplify_fails(model):
        raise RuntimeError("second run cannot simplify")

    monkeypatch.setattr(onnxsim_check, "simplify", simplify_fails)

    second = run_check(config, model, npz, environments_path=environments, artifacts_path=art)

    assert second["error"] == "onnx_simplify"
    assert sorted(f.name for f in art.iterdir()) == ["model.onnx", "notes.txt"]


def test_a_run_in_an_artifact_directory_from_an_older_version_clears_its_evaluation_report(monkeypatch, tmp_path, env_files, good_model):
    """The retired kernelgen_eval.json of an earlier run must not sit next to this run's files, even when this run fails."""
    config, environments = env_files
    art = tmp_path / ".artifact"
    art.mkdir()
    (art / "kernelgen_eval.json").write_text('{"opt_1": {"speedup": 2.0}}')
    (art / "notes.txt").write_text("mine")
    fake_process(monkeypatch, None, 1)

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["error"] == "model_check_run"
    assert sorted(f.name for f in art.iterdir()) == ["notes.txt"]


def test_managed_files_are_cleared_even_if_the_subprocess_fails_to_start(monkeypatch, tmp_path, env_files, good_model):
    config, environments = env_files
    art = tmp_path / ".artifact"
    art.mkdir()
    for name in artifacts.MANAGED_FILES:
        (art / name).write_bytes(b"stale")
    fake_process(monkeypatch, None, 1)

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["error"] == "model_check_run"
    assert list(art.iterdir()) == []


@pytest.mark.parametrize(
    "blocked, stage",
    [
        pytest.param("onnxscript", "onnx_exporter", id="exporter_missing"),
        pytest.param("onnxsim", "onnx_simplifier", id="simplifier_missing"),
        pytest.param("tensorrt", "tensorrt_import", id="tensorrt_missing"),
    ],
)
def test_missing_package_reaches_the_report(tmp_path, good_model, artifacts, blocked, stage):
    """Model-stack imports happen inside the nodes, so a missing package is a stage, not a startup crash."""
    model, npz = good_model
    report_path = tmp_path / "report.json"
    code = (
        f"import sys; sys.modules[{blocked!r}] = None\n"
        "from kronos_edge.eager_check import main\n"
        f"sys.exit(main([{str(model)!r}, {str(npz)!r}, '--report', {str(report_path)!r}, "
        f"'--artifacts', {str(artifacts)!r}, '--precision', 'fp32', '--threshold', '0.99', '--device', 'cpu']))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["error"] == stage
    assert blocked in report["error_detail"]


def test_artifact_directory_is_created_before_the_subprocess_runs(monkeypatch, tmp_path, env_files, good_model):
    seen = {}

    def fake(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        seen["exists"] = Path(artifacts_dir).is_dir()
        for name in ("model.engine",):  # what the kernelgen step waits for besides the export's reference
            (Path(artifacts_dir) / name).write_bytes(b"x")
        (Path(report_dir) / model_check.REPORT_FILE).write_text(json.dumps({"error": None, "benchmark": BENCHMARK}))
        return 0

    monkeypatch.setattr(model_check, "run_process", fake)
    config, environments = env_files
    art = tmp_path / "nested" / ".artifact"

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["outcome"] == "success"
    assert seen["exists"] and art.is_dir()


def test_a_subprocess_that_cannot_start_is_reported(monkeypatch, caplog, env_files, good_model):
    def boom(*args):
        raise OSError("cannot spawn: boom")

    monkeypatch.setattr(model_check, "run_process", boom)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "model_check_run"
    assert "support" in state["customer_message"]
    assert "boom" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["model_check_run"]


@pytest.mark.parametrize(
    "report, code",
    [
        pytest.param(None, 0, id="no_report_file"),
        pytest.param("not json", 0, id="not_json"),
        pytest.param("null", 0, id="null"),
        pytest.param("[]", 0, id="list"),
        pytest.param("{}", 0, id="missing_error_key"),
        pytest.param('{"error": null, "extra": 1}', 0, id="unknown_key"),
        pytest.param('{"error": "bogus_stage", "error_detail": "x"}', 1, id="unknown_stage"),
        pytest.param('{"error": 5}', 1, id="wrong_type"),
        pytest.param('{"error": null}', 137, id="success_report_but_killed"),
        pytest.param('{"error": "forward", "error_detail": "x"}', 0, id="failure_report_but_exit_zero"),
        pytest.param('{"error": "onnx_capture", "error_detail": "x"}', 0, id="onnx_failure_report_but_exit_zero"),
        pytest.param('{"error": null, "benchmark": {"speedup": "fast"}}', 0, id="malformed_benchmark"),
        pytest.param('{"error": null, "benchmark": null}', 0, id="success_without_benchmark"),
        pytest.param(json.dumps({"error": None, "benchmark": {**BENCHMARK, "values": [float("nan")]}}), 0, id="nan_score_in_report"),
    ],
)
def test_invalid_reports_are_model_check_failures(monkeypatch, caplog, env_files, good_model, report, code):
    fake_process(monkeypatch, report, code)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "model_check_run"
    assert [r.stage for r in error_records(caplog)] == ["model_check_run"]


def test_exit_code_mismatch_names_the_code(monkeypatch, env_files, good_model):
    fake_process(monkeypatch, '{"error": null}', 137)

    state = run(env_files, good_model)

    assert "137" in state["customer_message"]


def test_inner_failure_passes_through_without_a_host_error(monkeypatch, caplog, env_files, good_model):
    fake_process(monkeypatch, json.dumps({"error": "forward", "error_detail": "AttributeError: inner"}), 1)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "forward"
    assert "cannot be run through eager mode" in state["customer_message"]
    assert "AttributeError: inner" in state["customer_message"]
    assert error_records(caplog) == []


def test_model_check_runs_on_cuda(monkeypatch, env_files, good_model):
    seen = {}

    def record(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        seen.update(device=device, precision=precision, threshold=threshold)
        state = eager_check.run(model_path, npz_path, artifacts_dir, precision=precision, threshold=threshold, device="cpu")
        (Path(report_dir) / model_check.REPORT_FILE).write_text(eager_check.report(state))
        return 0

    monkeypatch.setattr(model_check, "run_process", record)

    state = run(env_files, good_model)

    assert state["outcome"] == "success"
    assert seen == {"device": "cuda", "precision": "fp32", "threshold": 0.99}


@pytest.mark.parametrize(
    "source, code, error",
    [
        pytest.param(GOOD_MODEL, 0, None, id="success"),
        pytest.param(TYPO_MODEL, 1, "forward", id="failure"),
    ],
)
def test_eager_main_writes_json_report(tmp_path, good_model, artifacts, source, code, error):
    _, npz = good_model
    model = tmp_path / "model.py"
    model.write_text(source)
    report_path = tmp_path / "report.json"

    exit_code = eager_check.main([str(model), str(npz), "--report", str(report_path), "--artifacts", str(artifacts), "--precision", "fp32", "--threshold", "0.99", "--device", "cpu"])

    report = json.loads(report_path.read_text())
    assert exit_code == code
    assert report["error"] == error
    assert ("convv" in report["error_detail"]) == (error is not None)


def test_user_prints_do_not_corrupt_the_report(tmp_path, good_model, artifacts):
    """Real subprocess, like the model check: user code printing on stdout must not affect the report.

    TensorRT is stubbed inside the subprocess the way the autouse fixture stubs it here.
    """
    _, npz = good_model
    model = tmp_path / "model.py"
    model.write_text(NOISY_MODEL)
    report_path = tmp_path / "report.json"
    code = (
        "import sys\n"
        "from kronos_edge import tensorrt_check\n"
        "tensorrt_check.build = lambda onnx_path, inputs: b'fake engine'\n"
        "from kronos_edge import benchmark_check\n"
        "from conftest import FakeEngineRunner\n"
        "benchmark_check.load_engine = FakeEngineRunner\n"
        "from kronos_edge.eager_check import main\n"
        f"sys.exit(main([{str(model)!r}, {str(npz)!r}, '--report', {str(report_path)!r}, "
        f"'--artifacts', {str(artifacts)!r}, '--precision', 'fp32', '--threshold', '0.99', '--device', 'cpu']))\n"
    )
    tests_dir = str(Path(__file__).resolve().parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), tests_dir])}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "loading without newline" in result.stdout and "bye from atexit" in result.stdout
    report = json.loads(report_path.read_text())
    assert (report["error"], report["error_detail"]) == (None, "")
    assert model_check.Report.model_validate(report).benchmark is not None
    assert "[torch.onnx]" not in result.stdout
    assert sorted(f.name for f in artifacts.iterdir()) == ["benchmark.json", "model.engine", "model.onnx", "model_simplified.onnx"]


def test_the_real_subprocess_runs_the_eager_check_and_writes_the_report(tmp_path, good_model, artifacts, capsys):
    """`run_process` as the workflow calls it, with stdout and stderr inherited: the eager graph runs in a child of
    this interpreter from any working directory and leaves a report the workflow can validate."""
    model, npz = good_model
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.chdir(elsewhere)
    os.environ.pop("PYTHONPATH", None)

    code = REAL_RUN_PROCESS(model, npz, tmp_path, artifacts, "fp32", 0.99, "cpu", "ERROR")

    report = model_check.Report.model_validate_json((tmp_path / model_check.REPORT_FILE).read_text())
    assert code == (0 if report.error is None else 1)
    assert (report.error is None) == (report.benchmark is not None)
    assert report.error in (None, "tensorrt_import", "benchmark_runtime")  # this environment may lack TensorRT
    assert "model.onnx" in {f.name for f in artifacts.iterdir()}  # the child ran this package's nodes
    assert capsys.readouterr().out == ""  # nothing of the child's stdout is routed through the workflow


# Captured at import, before the autouse fixture replaces it.
REAL_RUN_PROCESS = model_check.run_process


def test_cuda_unavailable_is_its_own_stage(monkeypatch, good_model, artifacts):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = eager_check.run(*good_model, artifacts, device="cuda")

    assert state["error"] == "select_device"
    assert "cuda" in state["error_detail"]

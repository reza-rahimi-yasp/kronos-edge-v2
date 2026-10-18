"""The model check: the eager graph runs as a subprocess of this interpreter and reports through report.json.

An autouse fixture runs that graph in-process on CPU instead of spawning it; the tests here cover the
node around it, the report protocol, the command, and the real subprocess against the user's stdout.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from conftest import GOOD_MODEL, run_directory
from kronos_edge import artifacts, eager_check, model_check, onnxsim_check

BENCHMARK = {
    "eager_ms": {"mean": 0.05, "median": 0.046, "p95": 0.06},
    "tensorrt_ms": {"mean": 0.016, "median": 0.015, "p95": 0.02},
    "speedup": 3.07,
    "metric": "cosine",
    "threshold": 0.99,
    "values": [0.9998],
}
from kronos_edge.workflow import main, run_check

# The credentials the README and .env.example have the user export: the API key, its deprecated alias, the registry key.
CREDENTIALS = ("YASP_API_KEY", "YASP_API_TOKEN", "REGISTRY_KEY")

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


ALL_FILES = [
    "benchmark.json", "kernel.py", "kernelgen_report.json",
    "model.engine", "model.onnx", "model.pt2", "model_simplified.onnx",
]


def clock(monkeypatch, *times):
    """Each run directory created from here on is named for the next of `times` (UTC)."""
    starts = iter(datetime(2026, 10, 17, *t, tzinfo=timezone.utc) for t in times)
    monkeypatch.setattr(artifacts, "utc_now", lambda: next(starts))


def test_successive_runs_keep_the_earlier_run_directories(monkeypatch, tmp_path, env_files, good_model):
    """A second run writes to a new directory; the first run's files and the user's own stay exactly as they were."""
    config, environments = env_files
    model, npz = good_model
    clock(monkeypatch, (14, 30, 5), (14, 31, 0))
    first = run_check(config, model, npz, environments_path=environments)
    assert first["outcome"] == "success"
    art = tmp_path / ".artifact"
    first_run = art / "20261017T143005Z"
    assert sorted(f.name for f in first_run.iterdir()) == ALL_FILES
    before = {f.name: f.read_bytes() for f in first_run.iterdir()}
    (art / "notes.txt").write_text("mine")

    def simplify_fails(model):
        raise RuntimeError("second run cannot simplify")

    monkeypatch.setattr(onnxsim_check, "simplify", simplify_fails)

    second = run_check(config, model, npz, environments_path=environments)

    assert second["tensorrt_error"] == "onnx_simplify"
    assert sorted(f.name for f in art.iterdir()) == ["20261017T143005Z", "20261017T143100Z", "notes.txt"]
    assert {f.name: f.read_bytes() for f in first_run.iterdir()} == before
    # The failed run keeps what it produced before the failure and the kernelgen branch's files, and nothing of the first run.
    assert sorted(f.name for f in (art / "20261017T143100Z").iterdir()) == [
        "kernel.py", "kernelgen_report.json", "model.onnx", "model.pt2",
    ]


def test_two_runs_in_the_same_second_do_not_share_a_directory(monkeypatch, tmp_path, env_files, good_model):
    config, environments = env_files
    clock(monkeypatch, (14, 30, 5), (14, 30, 5))

    first = run_check(config, *good_model, environments_path=environments)
    second = run_check(config, *good_model, environments_path=environments)

    art = tmp_path / ".artifact"
    assert first["kernel_path"] == str((art / "20261017T143005Z" / "kernel.py").resolve())
    assert second["kernel_path"] == str((art / "20261017T143005Z-2" / "kernel.py").resolve())
    for run in ("20261017T143005Z", "20261017T143005Z-2"):
        assert sorted(f.name for f in (art / run).iterdir()) == ALL_FILES


def test_a_run_leaves_files_in_the_artifact_root_alone(monkeypatch, tmp_path, env_files, good_model):
    """Files an older version wrote straight into the root, and the user's own, are neither read nor removed."""
    config, environments = env_files
    art = tmp_path / ".artifact"
    art.mkdir()
    legacy = {name: b"older version" for name in (*artifacts.MANAGED_FILES, "kernelgen_eval.json", "notes.txt")}
    for name, data in legacy.items():
        (art / name).write_bytes(data)
    fake_process(monkeypatch, None, 1)

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["error"] == "model_check_run"
    assert {f.name: f.read_bytes() for f in art.iterdir() if f.is_file()} == legacy
    assert list(run_directory(art).iterdir()) == []


def test_a_custom_artifact_root_gets_the_run_directory(monkeypatch, tmp_path, env_files, good_model, capsys):
    """`--artifacts path/to/root` puts the run in path/to/root/<timestamp>, and the success message names it."""
    config, environments = env_files
    model, npz = good_model
    clock(monkeypatch, (14, 30, 5))
    root = tmp_path / "path" / "to" / "root"

    code = main([str(config), str(model), str(npz), "--environments", str(environments), "--artifacts", str(root)])

    assert code == 0
    run = root / "20261017T143005Z"
    assert sorted(f.name for f in run.iterdir()) == ALL_FILES
    assert f"the kernel is {(run / 'kernel.py').resolve()}." in capsys.readouterr().out
    assert not (tmp_path / ".artifact").exists()


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


def test_the_run_directory_is_created_before_the_subprocess_runs_and_handed_to_it(monkeypatch, tmp_path, env_files, good_model):
    seen = {}

    def fake(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        seen["dir"] = Path(artifacts_dir)
        seen["exists"] = Path(artifacts_dir).is_dir()
        (Path(report_dir) / model_check.REPORT_FILE).write_text(json.dumps({"error": None, "benchmark": BENCHMARK}))
        return 0

    monkeypatch.setattr(model_check, "run_process", fake)
    config, environments = env_files
    art = tmp_path / "nested" / ".artifact"

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["outcome"] == "success"
    assert seen["exists"] and seen["dir"] == run_directory(art)
    assert state["run_path"] == str(seen["dir"])


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


def test_an_inner_tensorrt_branch_failure_is_kept_apart_and_kernelgen_runs(monkeypatch, caplog, env_files, good_model):
    fake_process(monkeypatch, json.dumps({"error": "onnx_capture", "error_detail": "GuardOnDataDependentSymNode: inner"}), 1)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["tensorrt_error"] == "onnx_capture"
    assert state["tensorrt_error_detail"] == "GuardOnDataDependentSymNode: inner"
    assert state.get("error") is None
    assert "could not be traced by torch.export" in state["customer_message"]
    assert "the kernel is" in state["customer_message"]
    assert error_records(caplog) == []


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


def test_the_real_subprocess_hands_the_model_no_yasp_credentials(tmp_path, good_model, artifacts, monkeypatch, capsys):
    """The user's model runs in the child: the Yasp API key, its alias, and the registry key the README has the user
    export must not reach it, nor the OS keyring through Python's keyring package. The report still comes back."""
    _, npz = good_model
    for name in CREDENTIALS:
        monkeypatch.setenv(name, "fake-secret-sentinel")
    seen = tmp_path / "seen.json"
    model = tmp_path / "model.py"
    model.write_text(GOOD_MODEL.replace(
        "import torch\n",
        "import json\nimport os\nimport torch\n\n"
        f"open({str(seen)!r}, 'w').write(json.dumps({{k: os.environ.get(k) for k in {[*CREDENTIALS, 'PYTHON_KEYRING_BACKEND']!r}}}))\n",
        1,
    ))

    code = REAL_RUN_PROCESS(model, npz, tmp_path, artifacts, "fp32", 0.99, "cpu", "ERROR")

    assert json.loads(seen.read_text()) == {**dict.fromkeys(CREDENTIALS), "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    report = model_check.Report.model_validate_json((tmp_path / model_check.REPORT_FILE).read_text())
    assert code == (0 if report.error is None else 1)
    assert "fake-secret-sentinel" not in report.error_detail
    assert capsys.readouterr().out == ""


def test_the_subprocess_environment_drops_only_the_yasp_credentials(monkeypatch):
    for name in CREDENTIALS:
        monkeypatch.setenv(name, "fake-secret-sentinel")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = model_check.process_environment()

    assert not set(CREDENTIALS) & set(env)
    assert "fake-secret-sentinel" not in env.values()
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["PATH"] == os.environ["PATH"]


# Captured at import, before the autouse fixture replaces it.
REAL_RUN_PROCESS = model_check.run_process


def test_cuda_unavailable_is_its_own_stage(monkeypatch, good_model, artifacts):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = eager_check.run(*good_model, artifacts, device="cuda")

    assert state["error"] == "select_device"
    assert "cuda" in state["error_detail"]

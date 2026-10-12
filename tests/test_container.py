import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import GOOD_MODEL
from kronos_edge import artifacts, container, eager_check, onnxsim_check
from kronos_edge.workflow import run_check

TYPO_MODEL = GOOD_MODEL.replace("return self.conv(x)", "return self.convv(x)")

NOISY_MODEL = GOOD_MODEL.replace(
    "import torch\n",
    "import atexit\nimport sys\nimport torch\n\n"
    "sys.stdout.write('loading without newline')\n"
    "atexit.register(lambda: print('bye from atexit'))\n",
).replace("        return self.conv(x)", "        print('forward called', end='')\n        return self.conv(x)")


def error_records(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR" and r.name.startswith("kronos_edge")]


def run(env_files, good_model):
    config, environments = env_files
    return run_check(config, *good_model, environments_path=environments)


def fake_container(monkeypatch, report=None, code=0):
    """Replace docker run with something that writes `report` (if given) and exits with `code`."""

    def fake(image, model_path, npz_path, report_dir, artifacts_dir, device, log_level):
        if report is not None:
            (Path(report_dir) / container.REPORT_FILE).write_text(report)
        return code

    monkeypatch.setattr(container, "run_container", fake)


def test_build_command_mounts_files_and_requests_gpu(tmp_path):
    model, npz, out, artifacts = tmp_path / "m.py", tmp_path / "i.npz", tmp_path / "out", tmp_path / ".artifact"

    cmd = container.build_command("kronos-edge:x", model, npz, out, artifacts, "cuda", "DEBUG")

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert f"{model.resolve()}:/work/model.py:ro" in cmd
    assert f"{npz.resolve()}:/work/input.npz:ro" in cmd
    assert f"{out.resolve()}:/work/out" in cmd
    assert f"{artifacts.resolve()}:/work/artifact" in cmd
    assert cmd[cmd.index("kronos-edge:x") + 1:] == [
        "python", "-m", "kronos_edge.eager_check", "/work/model.py", "/work/input.npz",
        "--report", "/work/out/report.json", "--artifacts", "/work/artifact",
        "--device", "cuda", "--log-level", "DEBUG",
    ]


def test_rerun_clears_managed_files_but_keeps_the_users_own(monkeypatch, tmp_path, env_files, good_model):
    """A second run in the same directory must not leave the first run's outputs next to the new ones."""
    config, environments = env_files
    model, npz = good_model
    art = tmp_path / ".artifact"
    first = run_check(config, model, npz, environments_path=environments, artifacts_path=art)
    assert first["outcome"] == "success"
    assert sorted(f.name for f in art.iterdir()) == ["model.onnx", "model_simplified.onnx"]
    (art / "notes.txt").write_text("mine")
    (art / "model.onnx.data").write_bytes(b"stale companion")

    def simplify_fails(model):
        raise RuntimeError("second run cannot simplify")

    monkeypatch.setattr(onnxsim_check, "simplify", simplify_fails)

    second = run_check(config, model, npz, environments_path=environments, artifacts_path=art)

    assert second["error"] == "onnx_simplify"
    assert sorted(f.name for f in art.iterdir()) == ["model.onnx", "notes.txt"]


def test_managed_files_are_cleared_even_if_the_container_fails_to_start(monkeypatch, tmp_path, env_files, good_model):
    config, environments = env_files
    art = tmp_path / ".artifact"
    art.mkdir()
    for name in artifacts.MANAGED_FILES:
        (art / name).write_bytes(b"stale")
    fake_container(monkeypatch, None, 1)

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["error"] == "container_run"
    assert list(art.iterdir()) == []


@pytest.mark.parametrize(
    "blocked, stage",
    [
        pytest.param("onnxscript", "onnx_exporter", id="exporter_missing"),
        pytest.param("onnxsim", "onnx_simplifier", id="simplifier_missing"),
    ],
)
def test_missing_dependency_reaches_the_report(tmp_path, good_model, artifacts, blocked, stage):
    """Dependency imports happen inside the nodes, so a missing package is a stage, not a startup crash."""
    model, npz = good_model
    report_path = tmp_path / "report.json"
    code = (
        f"import sys; sys.modules[{blocked!r}] = None\n"
        "from kronos_edge.eager_check import main\n"
        f"sys.exit(main([{str(model)!r}, {str(npz)!r}, '--report', {str(report_path)!r}, "
        f"'--artifacts', {str(artifacts)!r}, '--device', 'cpu']))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["error"] == stage
    assert blocked in report["error_detail"]


def test_artifact_directory_is_created_before_the_container_runs(monkeypatch, tmp_path, env_files, good_model):
    seen = {}

    def fake(image, model_path, npz_path, report_dir, artifacts_dir, device, log_level):
        seen["exists"] = Path(artifacts_dir).is_dir()
        (Path(report_dir) / container.REPORT_FILE).write_text('{"error": null}')
        return 0

    monkeypatch.setattr(container, "run_container", fake)
    config, environments = env_files
    artifacts = tmp_path / "nested" / ".artifact"

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=artifacts)

    assert state["outcome"] == "success"
    assert seen["exists"] and artifacts.is_dir()


def test_docker_failure_is_reported(monkeypatch, caplog, env_files, good_model):
    def boom(*args):
        raise RuntimeError("docker: boom")

    monkeypatch.setattr(container, "run_container", boom)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "container_run"
    assert "support" in state["customer_message"]
    assert "boom" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["container_run"]


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
    ],
)
def test_invalid_reports_are_container_failures(monkeypatch, caplog, env_files, good_model, report, code):
    fake_container(monkeypatch, report, code)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "container_run"
    assert [r.stage for r in error_records(caplog)] == ["container_run"]


def test_exit_code_mismatch_names_the_code(monkeypatch, env_files, good_model):
    fake_container(monkeypatch, '{"error": null}', 137)

    state = run(env_files, good_model)

    assert "137" in state["customer_message"]


def test_inner_failure_passes_through_without_a_host_error(monkeypatch, caplog, env_files, good_model):
    fake_container(monkeypatch, json.dumps({"error": "forward", "error_detail": "AttributeError: inner"}), 1)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "forward"
    assert "cannot be run through eager mode" in state["customer_message"]
    assert "AttributeError: inner" in state["customer_message"]
    assert error_records(caplog) == []


def test_container_runs_on_cuda(monkeypatch, env_files, good_model):
    seen = {}

    def record(image, model_path, npz_path, report_dir, artifacts_dir, device, log_level):
        seen.update(image=image, device=device)
        state = eager_check.run(model_path, npz_path, artifacts_dir, device="cpu")
        (Path(report_dir) / container.REPORT_FILE).write_text(eager_check.report(state))
        return 0

    monkeypatch.setattr(container, "run_container", record)

    state = run(env_files, good_model)

    assert state["outcome"] == "success"
    assert seen == {"image": "kronos-edge:rtx5090-cuda130", "device": "cuda"}


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

    exit_code = eager_check.main([str(model), str(npz), "--report", str(report_path), "--artifacts", str(artifacts), "--device", "cpu"])

    report = json.loads(report_path.read_text())
    assert exit_code == code
    assert report["error"] == error
    assert ("convv" in report["error_detail"]) == (error is not None)


def test_user_prints_do_not_corrupt_the_report(tmp_path, good_model, artifacts):
    """Real subprocess, like the container: user code printing on stdout must not affect the report."""
    _, npz = good_model
    model = tmp_path / "model.py"
    model.write_text(NOISY_MODEL)
    report_path = tmp_path / "report.json"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    result = subprocess.run(
        [sys.executable, "-m", "kronos_edge.eager_check", str(model), str(npz),
         "--report", str(report_path), "--artifacts", str(artifacts), "--device", "cpu"],
        env=env, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "loading without newline" in result.stdout and "bye from atexit" in result.stdout
    assert json.loads(report_path.read_text()) == {"error": None, "error_detail": ""}
    assert "[torch.onnx]" not in result.stdout
    assert sorted(f.name for f in artifacts.iterdir()) == ["model.onnx", "model_simplified.onnx"]


def test_cuda_unavailable_is_its_own_stage(monkeypatch, good_model, artifacts):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = eager_check.run(*good_model, artifacts, device="cuda")

    assert state["error"] == "select_device"
    assert "cuda" in state["error_detail"]

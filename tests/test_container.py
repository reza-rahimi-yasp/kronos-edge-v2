import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import CONFIG, GOOD_MODEL, fake_export, write_yaml
from kronos_edge import artifacts, container, eager_check, onnxsim_check

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


def error_records(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR" and r.name.startswith("kronos_edge")]


def run(env_files, good_model):
    config, environments = env_files
    return run_check(config, *good_model, environments_path=environments)


def fake_container(monkeypatch, report=None, code=0):
    """Replace docker run with something that writes `report` (if given) and exits with `code`."""

    def fake(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies):
        if report is not None:
            (Path(report_dir) / container.REPORT_FILE).write_text(report)
        return code

    monkeypatch.setattr(container, "run_container", fake)


def test_build_command_mounts_files_and_requests_gpu(tmp_path):
    model, npz, out, artifacts = tmp_path / "m.py", tmp_path / "i.npz", tmp_path / "out", tmp_path / ".artifact"

    cmd = container.build_command(
        "kronos-edge:x", model, npz, out, artifacts, "fp16", 0.99, "cuda", "DEBUG", ["timm", "einops==0.8.0"]
    )

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert f"{model.resolve()}:/work/model.py:ro" in cmd
    assert f"{npz.resolve()}:/work/input.npz:ro" in cmd
    assert f"{out.resolve()}:/work/out" in cmd
    assert f"{artifacts.resolve()}:/work/artifact" in cmd
    assert cmd[cmd.index("kronos-edge:x") + 1:] == [
        "python", "-m", "kronos_edge.eager_check", "/work/model.py", "/work/input.npz",
        "--report", "/work/out/report.json", "--artifacts", "/work/artifact",
        "--precision", "fp16", "--threshold", "0.99", "--device", "cuda", "--log-level", "DEBUG",
        "--dependency", "timm", "--dependency", "einops==0.8.0",
    ]


def test_no_dependencies_adds_no_flags(tmp_path):
    cmd = container.build_command("kronos-edge:x", tmp_path / "m.py", tmp_path / "i.npz", tmp_path, tmp_path, "fp32", 0.99, "cuda", "INFO", [])

    assert "--dependency" not in cmd


def test_config_dependencies_are_installed_in_the_container(tmp_path, env_files, good_model, pip_stub):
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, "dependencies": ["timm", "einops==0.8.0"]})

    state = run_check(config, *good_model, environments_path=environments)

    assert state["outcome"] == "success"
    assert pip_stub == [["timm", "einops==0.8.0"]]


def test_without_dependencies_nothing_is_installed(env_files, good_model, pip_stub):
    state = run(env_files, good_model)

    assert state["outcome"] == "success"
    assert pip_stub == []


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
    fake_container(monkeypatch, None, 1)

    state = run_check(config, *good_model, environments_path=environments, artifacts_path=art)

    assert state["error"] == "container_run"
    assert sorted(f.name for f in art.iterdir()) == ["notes.txt"]


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
        pytest.param("tensorrt", "tensorrt_import", id="tensorrt_missing"),
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
        f"'--artifacts', {str(artifacts)!r}, '--precision', 'fp32', '--threshold', '0.99', '--device', 'cpu']))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["error"] == stage
    assert blocked in report["error_detail"]


def test_artifact_directory_is_created_before_the_container_runs(monkeypatch, tmp_path, env_files, good_model):
    seen = {}

    def fake(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies):
        seen["exists"] = Path(artifacts_dir).is_dir()
        for name in ("model.engine",):  # what the kernelgen step waits for besides the export's reference
            (Path(artifacts_dir) / name).write_bytes(b"x")
        (Path(report_dir) / container.REPORT_FILE).write_text(json.dumps({"error": None, "benchmark": BENCHMARK}))
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
        pytest.param('{"error": null, "benchmark": {"speedup": "fast"}}', 0, id="malformed_benchmark"),
        pytest.param('{"error": null, "benchmark": null}', 0, id="success_without_benchmark"),
        pytest.param(json.dumps({"error": None, "benchmark": {**BENCHMARK, "values": [float("nan")]}}), 0, id="nan_score_in_report"),
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

    def record(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies):
        seen.update(image=image, device=device, precision=precision, threshold=threshold)
        state = eager_check.run(model_path, npz_path, artifacts_dir, precision=precision, threshold=threshold, device="cpu")
        (Path(report_dir) / container.REPORT_FILE).write_text(eager_check.report(state))
        return 0

    monkeypatch.setattr(container, "run_container", record)

    state = run(env_files, good_model)

    assert state["outcome"] == "success"
    assert seen == {"image": "kronos-edge:rtx5090-cuda130", "device": "cuda", "precision": "fp32", "threshold": 0.99}


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
    """Real subprocess, like the container: user code printing on stdout must not affect the report.

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
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), tests_dir])}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "loading without newline" in result.stdout and "bye from atexit" in result.stdout
    report = json.loads(report_path.read_text())
    assert (report["error"], report["error_detail"]) == (None, "")
    assert container.Report.model_validate(report).benchmark is not None
    assert "[torch.onnx]" not in result.stdout
    assert sorted(f.name for f in artifacts.iterdir()) == ["benchmark.json", "model.engine", "model.onnx", "model_simplified.onnx"]


def test_cuda_unavailable_is_its_own_stage(monkeypatch, good_model, artifacts):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = eager_check.run(*good_model, artifacts, device="cuda")

    assert state["error"] == "select_device"
    assert "cuda" in state["error_detail"]


# ---------- The export container: the documented `yasp-toolkit kernelgen export` on model.py, on the CPU, without a key.


def test_export_command_runs_the_documented_exporter_without_gpu_or_credentials(tmp_path):
    model, art = tmp_path / "m.py", tmp_path / ".artifact"

    cmd = container.build_export_command(model, art, ["timm", "einops==0.8.0"])

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert f"{model.resolve()}:/work/model.py:ro" in cmd
    assert f"{art.resolve()}:/work/artifact" in cmd
    assert container.EXPORT_IMAGE in cmd
    assert "--gpus" not in cmd
    assert not any(part in ("-e", "--env", "--env-file") or "YASP" in part for part in cmd)  # no credential enters
    script = cmd[cmd.index("-c") + 1]
    assert "yasp-toolkit kernelgen export --input /work/model.py --output /work/artifact/model.pt2" in script
    # The requirements reach the script as arguments, never spliced into its text.
    assert cmd[cmd.index("-c") + 2:] == ["kronos-export", "timm", "einops==0.8.0"]
    assert "timm" not in script and "einops" not in script


def test_export_without_dependencies_passes_none(tmp_path):
    cmd = container.build_export_command(tmp_path / "m.py", tmp_path / ".artifact", [])

    assert cmd[-1] == "kronos-export"


def test_export_runs_after_the_container_on_the_users_model_with_the_config_dependencies(monkeypatch, tmp_path, env_files, good_model):
    seen = []

    def recording(model_path, artifacts_dir, dependencies):
        seen.append((str(model_path), sorted(f.name for f in Path(artifacts_dir).iterdir()), list(dependencies)))
        return fake_export(model_path, artifacts_dir, dependencies)

    monkeypatch.setattr(container, "run_export", recording)
    _, environments = env_files
    config = write_yaml(tmp_path / "deps.yaml", {**CONFIG, "dependencies": ["einops==0.8.0"]})

    state = run_check(config, *good_model, environments_path=environments)

    assert state["outcome"] == "success"
    ((model_path, present, dependencies),) = seen
    assert model_path == str(good_model[0])
    assert {"model.engine", "benchmark.json"} <= set(present)  # the model-check container ran first
    assert "model.pt2" not in present
    assert dependencies == ["einops==0.8.0"]


def exporting(code, output, writes=False):
    def run_export(model_path, artifacts_dir, dependencies):
        if writes:
            (Path(artifacts_dir) / "model.pt2").write_bytes(b"x")
        return subprocess.CompletedProcess(["docker", "run"], code, stdout=output, stderr="")

    return run_export


@pytest.mark.parametrize(
    "run_export, stage, fragment",
    [
        pytest.param(
            lambda *args: (_ for _ in ()).throw(FileNotFoundError(2, "No such file or directory", "docker")),
            "export_run", "docker", id="docker_missing",
        ),
        pytest.param(
            exporting(container.DEPENDENCIES_FAILED, "ERROR: No matching distribution found for nosuchpkg\n"),
            "export_dependencies", "nosuchpkg", id="dependency_not_installable",
        ),
        pytest.param(
            exporting(1, "torch._dynamo.exc.UserError: data-dependent branch\n"),
            "export_reference", "data-dependent", id="export_failed",
        ),
        pytest.param(exporting(0, "  wrote      nothing\n"), "export_reference", "model.pt2", id="exit_zero_without_a_reference"),
    ],
)
def test_export_failures_are_stages_and_stop_before_any_toolkit_call(monkeypatch, caplog, env_files, good_model, capsys, run_export, stage, fragment):
    from kronos_edge import kernelgen_check

    calls = []
    monkeypatch.setattr(container, "run_export", run_export)
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: calls.append(args))

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert calls == []
    assert capsys.readouterr().out == ""  # the export's output, the user's prints included, never reaches stdout


def test_export_is_skipped_when_the_container_fails(monkeypatch, env_files, good_model):
    exports = []
    monkeypatch.setattr(container, "run_export", lambda *args: exports.append(args))
    fake_container(monkeypatch, json.dumps({"error": "forward", "error_detail": "boom", "benchmark": None}), 1)

    state = run(env_files, good_model)

    assert state["error"] == "forward"
    assert exports == []

"""The preflight: before anything runs, the Python environment the check was started in must be the matched registry entry.

The probe that imports torch and tensorrt runs as a subprocess of this interpreter; an autouse fixture
replaces it with the registry entry's versions and GPU. The tests here cover the node around it, the
version comparison, the real probe in this environment, and the Dockerfile that documents the entry.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from conftest import CONFIG, ENVIRONMENT, PROBE, REGISTRY, write_yaml
from kronos_edge import kernelgen_check, model_check, preflight
from kronos_edge.workflow import run_check

ROOT = Path(__file__).resolve().parents[1]


def error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith("kronos_edge")]


def run(env_files, good_model):
    config, environments = env_files
    return run_check(config, *good_model, environments_path=environments)


def recording_run_process(seen):
    def record(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        seen.append("model_check")
        return 1

    return record


def test_an_environment_that_is_the_registry_entry_passes(env_files, good_model):
    state = run(env_files, good_model)

    assert state["outcome"] == "success"


def test_preflight_runs_after_the_match_and_before_the_model_check(monkeypatch, tmp_path, env_files, good_model):
    seen = []
    monkeypatch.setattr(preflight, "run_probe", lambda: seen.append("probe") or dict(PROBE))
    monkeypatch.setattr(model_check, "run_process", recording_run_process(seen))
    _, environments = env_files
    mismatch = write_yaml(tmp_path / "mismatch.yaml", {**CONFIG, "torch": "9.9.9"})

    state = run_check(mismatch, *good_model, environments_path=environments)

    assert state["error"] == "match_environment"
    assert seen == []  # no registry entry, nothing to compare the environment with

    state = run(env_files, good_model)

    assert state["error"] == "model_check_run"
    assert seen == ["probe", "model_check"]


@pytest.mark.parametrize(
    "probe, fragments",
    [
        pytest.param({**PROBE, "torch": "2.9.1+cu130"}, ["torch: requested 2.14.1, found 2.9.1+cu130"], id="torch_version"),
        pytest.param({**PROBE, "torch": None, "cuda": None, "gpu": None}, ["torch: requested 2.14.1, not installed"], id="torch_missing"),
        pytest.param({**PROBE, "cuda": "12.8"}, ["cuda: requested 13.0, found 12.8"], id="cuda_version"),
        pytest.param({**PROBE, "cuda": None}, ["cuda: requested 13.0, found a torch without CUDA"], id="torch_without_cuda"),
        pytest.param({**PROBE, "tensorrt": "10.13.0.35"}, ["tensorrt: requested 11.4.0, found 10.13.0.35"], id="tensorrt_version"),
        pytest.param({**PROBE, "tensorrt": None}, ["tensorrt: requested 11.4.0, not installed"], id="tensorrt_missing"),
        pytest.param({**PROBE, "gpu": None}, ["gpu: requested NVIDIA GeForce RTX 5090, none available to torch"], id="no_gpu"),
        pytest.param({**PROBE, "gpu": "NVIDIA GeForce RTX 4090"}, ["gpu: requested NVIDIA GeForce RTX 5090, found NVIDIA GeForce RTX 4090"], id="other_gpu"),
        pytest.param(
            {"torch": "2.9.1+cpu", "cuda": None, "tensorrt": None, "gpu": None},
            ["torch: requested 2.14.1, found 2.9.1+cpu", "cuda: requested 13.0", "tensorrt: requested 11.4.0, not installed", "gpu: requested"],
            id="every_difference_is_listed_at_once",
        ),
    ],
)
def test_a_differing_environment_stops_before_the_model_check(monkeypatch, caplog, env_files, good_model, probe, fragments):
    seen = []
    monkeypatch.setattr(preflight, "run_probe", lambda: probe)
    monkeypatch.setattr(model_check, "run_process", recording_run_process(seen))

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "environment_mismatch"
    assert "does not match" in state["customer_message"]
    for fragment in fragments:
        assert fragment in state["customer_message"]
    assert REGISTRY["environments"][0]["dockerfile"] in state["customer_message"]  # the recipe for the environment
    assert "driver" not in state["customer_message"]  # the driver is not checked against this machine
    assert [r.stage for r in error_records(caplog)] == ["environment_mismatch"]
    assert seen == []
    assert not Path(".artifact").exists()


@pytest.mark.parametrize(
    "installed, required, matches",
    [
        pytest.param("2.14.1+cu130", "2.14.1", True, id="torch_build_label"),
        pytest.param("2.14.1", "2.14.1", True, id="exact"),
        pytest.param("11.4.0.106", "11.4.0", True, id="tensorrt_build_number"),
        pytest.param("13.0", "13.0", True, id="cuda"),
        pytest.param("2.14.10", "2.14.1", False, id="component_not_prefix"),
        pytest.param("2.14", "2.14.1", False, id="shorter"),
        pytest.param("2.14.2+cu130", "2.14.1", False, id="patch_differs"),
        pytest.param("11.4.0.106", "11.4.1", False, id="tensorrt_patch_differs"),
        pytest.param(None, "2.14.1", False, id="missing"),
    ],
)
def test_installed_versions_match_the_registry_by_component(installed, required, matches):
    assert preflight.matches(installed, required) is matches


def test_a_probe_that_cannot_run_is_its_own_stage(monkeypatch, caplog, env_files, good_model):
    def broken():
        raise RuntimeError("probe exited 1\nSegmentation fault")

    monkeypatch.setattr(preflight, "run_probe", broken)

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "preflight_probe"
    assert "support" in state["customer_message"]
    assert "Segmentation fault" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["preflight_probe"]


def test_a_missing_toolkit_stops_before_the_model_check(monkeypatch, caplog, env_files, good_model):
    seen = []
    monkeypatch.setattr(preflight, "toolkit_on_path", lambda: None)
    monkeypatch.setattr(model_check, "run_process", recording_run_process(seen))
    monkeypatch.setattr(kernelgen_check, "run_toolkit", lambda args, timeout=None: seen.append(args))

    state = run(env_files, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "kernelgen_toolkit"
    assert "yasp-agent" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["kernelgen_toolkit"]
    assert seen == []


def test_the_real_toolkit_lookup_is_a_path_search(monkeypatch, tmp_path):
    script = tmp_path / "yasp-toolkit"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert REAL_TOOLKIT_ON_PATH() == str(script)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert REAL_TOOLKIT_ON_PATH() is None


# Captured at import, before the autouse fixtures replace them.
REAL_TOOLKIT_ON_PATH = preflight.toolkit_on_path
REAL_RUN_PROBE = preflight.run_probe


def test_the_real_probe_reports_this_environment():
    """In-process: torch as it is here; tensorrt and a GPU only if this environment has them."""
    report = preflight.probe()

    assert set(report) == {"torch", "cuda", "tensorrt", "gpu"}
    assert report["torch"] == torch.__version__
    assert report["cuda"] == torch.version.cuda
    assert report["gpu"] == (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    assert isinstance(report["tensorrt"], (str, type(None)))


def test_the_real_probe_runs_as_a_subprocess_of_this_interpreter():
    """The workflow never imports torch: the probe runs under `python -m` and reports JSON on its stdout."""
    assert REAL_RUN_PROBE() == preflight.probe()


def test_a_probe_that_prints_no_report_is_an_error(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="boom"))

    with pytest.raises(RuntimeError, match="boom"):
        REAL_RUN_PROBE()


def test_a_probe_whose_output_is_not_a_report_is_an_error(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="hello from the model\n", stderr=""))

    with pytest.raises(RuntimeError, match="hello from the model"):
        REAL_RUN_PROBE()


def test_the_probe_command_is_this_interpreter():
    assert preflight.probe_command()[:3] == [sys.executable, "-m", "kronos_edge.preflight"]


# ---------- The Dockerfiles are the recipes for the registry's environments, validated, never run by the check.


def dockerfile_labels(path: Path) -> dict[str, str]:
    return dict(re.findall(r'ai\.yasp\.kronos-edge\.(\w+)="([^"]*)"', path.read_text()))


@pytest.mark.parametrize("entry", yaml.safe_load((ROOT / "src/kronos_edge/environments.yaml").read_text())["environments"], ids=lambda e: e["name"])
def test_each_registry_entry_has_a_dockerfile_stating_the_same_versions(entry):
    dockerfile = ROOT / entry["dockerfile"]

    assert dockerfile.is_file()
    assert dockerfile_labels(dockerfile) == {field: entry[field] for field in ENVIRONMENT}


def test_the_check_runs_no_docker_command():
    """The Dockerfiles are fixtures: nothing under src/ invokes docker or checks for an image."""
    sources = "\n".join(p.read_text() for p in (ROOT / "src/kronos_edge").glob("*.py"))

    assert '"docker"' not in sources
    assert "image_exists" not in sources

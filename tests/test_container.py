import json

import pytest
import torch

from conftest import GOOD_MODEL
from kronos_edge import container, eager_check
from kronos_edge.workflow import run_check

TYPO_MODEL = GOOD_MODEL.replace("return self.conv(x)", "return self.convv(x)")


def error_records(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR" and r.name.startswith("kronos_edge")]


def test_build_command_mounts_files_and_requests_gpu(tmp_path):
    model, npz = tmp_path / "m.py", tmp_path / "i.npz"

    cmd = container.build_command("kronos-edge:x", model, npz, "cuda", "DEBUG")

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert f"{model.resolve()}:/work/model.py:ro" in cmd
    assert f"{npz.resolve()}:/work/input.npz:ro" in cmd
    assert cmd[cmd.index("kronos-edge:x") + 1:] == [
        "python", "-m", "kronos_edge.eager_check", "/work/model.py", "/work/input.npz",
        "--device", "cuda", "--log-level", "DEBUG",
    ]


def test_docker_failure_is_reported(monkeypatch, caplog, env_files, good_model):
    def boom(*args):
        raise RuntimeError("docker: boom")

    monkeypatch.setattr(container, "run_container", boom)
    config, environments = env_files

    state = run_check(config, *good_model, environments_path=environments)

    assert state["outcome"] == "failure"
    assert state["error"] == "container_run"
    assert "support" in state["customer_message"]
    assert "boom" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["container_run"]


def test_non_json_output_is_reported(monkeypatch, caplog, env_files, good_model):
    monkeypatch.setattr(container, "run_container", lambda *args: "not json")
    config, environments = env_files

    state = run_check(config, *good_model, environments_path=environments)

    assert state["error"] == "container_run"
    assert [r.stage for r in error_records(caplog)] == ["container_run"]


def test_inner_failure_passes_through_without_a_host_error(monkeypatch, caplog, env_files, good_model):
    report = json.dumps({"error": "forward", "error_detail": "AttributeError: inner"})
    monkeypatch.setattr(container, "run_container", lambda *args: report)
    config, environments = env_files

    state = run_check(config, *good_model, environments_path=environments)

    assert state["outcome"] == "failure"
    assert state["error"] == "forward"
    assert "cannot be run through eager mode" in state["customer_message"]
    assert "AttributeError: inner" in state["customer_message"]
    assert error_records(caplog) == []


def test_container_runs_on_cuda(monkeypatch, env_files, good_model):
    seen = {}

    def record(image, model_path, npz_path, device, log_level):
        seen.update(image=image, device=device)
        return eager_check.report(eager_check.run(model_path, npz_path, device="cpu"))

    monkeypatch.setattr(container, "run_container", record)
    config, environments = env_files

    state = run_check(config, *good_model, environments_path=environments)

    assert state["outcome"] == "success"
    assert seen == {"image": "kronos-edge:rtx5090-cuda130", "device": "cuda"}


@pytest.mark.parametrize(
    "source, code, error",
    [
        pytest.param(GOOD_MODEL, 0, None, id="success"),
        pytest.param(TYPO_MODEL, 1, "forward", id="failure"),
    ],
)
def test_eager_main_prints_json_report(tmp_path, good_model, capsys, source, code, error):
    _, npz = good_model
    model = tmp_path / "model.py"
    model.write_text(source)

    exit_code = eager_check.main([str(model), str(npz), "--device", "cpu"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == code
    assert report["error"] == error
    assert ("convv" in report["error_detail"]) == (error is not None)


def test_cuda_unavailable_is_its_own_stage(monkeypatch, good_model):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = eager_check.run(*good_model, device="cuda")

    assert state["error"] == "select_device"
    assert "cuda" in state["error_detail"]

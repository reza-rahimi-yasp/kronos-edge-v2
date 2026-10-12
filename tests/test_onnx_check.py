"""The ONNX export check: it runs after a successful eager pass, in memory, and names the cause when it fails."""

from pathlib import Path

import pytest
import torch

from kronos_edge import container, onnx_check
from test_eager_check import CONV_RELU, X, error_records, run, write_model, write_npz

# Runs in eager mode but torch.export cannot trace the data-dependent branch.
DATA_DEPENDENT = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return -x


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""

# Runs in eager mode and traces, but the operator has no ONNX translation.
UNSUPPORTED_OP = DATA_DEPENDENT.replace(
    "        if x.sum() > 0:\n            return x * 2\n        return -x", "        return torch.special.airy_ai(x)"
)

# Mutates a parameter in place in forward: fine under no_grad, an error if gradients are enabled.
INPLACE_PARAM = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 3, 8, 8))

    def forward(self, x):
        self.weight.clamp_(min=0)
        return x * self.weight


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""


def onnx_files(*dirs):
    return sorted(f for d in dirs for f in Path(d).rglob("*.onnx"))


@pytest.mark.parametrize(
    "source, stage, fragment, cause",
    [
        pytest.param(DATA_DEPENDENT, "onnx_capture", "could not be traced", "data-dependent", id="data_dependent"),
        pytest.param(UNSUPPORTED_OP, "onnx_convert", "no ONNX translation", "airy_ai", id="unsupported_op"),
    ],
)
def test_failure_names_the_stage_and_the_cause(tmp_path, caplog, env_files, source, stage, fragment, cause):
    state = run(env_files, write_model(tmp_path, source), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert cause in state["customer_message"]
    assert "\x1b[" not in state["customer_message"]
    assert "submit a PR" not in state["customer_message"]
    records = error_records(caplog)
    assert len(records) == 1
    assert records[0].stage == stage


def test_export_runs_without_gradients_like_the_eager_pass(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, INPLACE_PARAM), write_npz(tmp_path, [X]))

    assert state.get("error") is None
    assert state["outcome"] == "success"


def test_missing_exporter_is_a_support_failure(tmp_path, monkeypatch, env_files):
    def no_exporter(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'onnxscript'")

    monkeypatch.setattr(onnx_check, "export", no_exporter)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == "onnx_exporter"
    assert "support" in state["customer_message"]
    assert "onnxscript" in state["customer_message"]


def test_export_is_run_in_memory_and_writes_no_onnx_file(tmp_path, monkeypatch, env_files):
    monkeypatch.chdir(tmp_path)
    calls = []
    real_export = onnx_check.export

    def recording_export(model, inputs):
        program = real_export(model, inputs)
        calls.append((model, inputs, program))
        return program

    monkeypatch.setattr(onnx_check, "export", recording_export)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert len(calls) == 1
    model, inputs, program = calls[0]
    assert isinstance(model, torch.nn.Module)
    assert len(inputs) == 1 and tuple(inputs[0].shape) == X.shape
    assert isinstance(program, torch.onnx.ONNXProgram)
    assert onnx_files(tmp_path) == []


def test_report_directory_holds_only_the_report(tmp_path, monkeypatch, env_files):
    """The export must not leave an .onnx file next to report.json for the host to clean up."""
    seen = {}
    real_run_container = container.run_container

    def spying_run_container(image, model_path, npz_path, report_dir, device, log_level):
        code = real_run_container(image, model_path, npz_path, report_dir, device, log_level)
        seen["files"] = sorted(f.name for f in Path(report_dir).iterdir())
        return code

    monkeypatch.setattr(container, "run_container", spying_run_container)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert seen["files"] == [container.REPORT_FILE]


def test_failed_export_writes_no_onnx_file(tmp_path, monkeypatch, env_files):
    monkeypatch.chdir(tmp_path)

    state = run(env_files, write_model(tmp_path, UNSUPPORTED_OP), write_npz(tmp_path, [X]))

    assert state["error"] == "onnx_convert"
    assert onnx_files(tmp_path) == []

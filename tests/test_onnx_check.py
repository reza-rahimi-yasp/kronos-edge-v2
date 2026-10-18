"""The ONNX export check: it runs after a successful eager pass, in memory, and names the cause when it fails."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from conftest import run_directory
from kronos_edge import eager_check, model_check, onnx_check
from kronos_edge.workflow import run_check
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


def test_export_is_written_to_the_artifact_directory(tmp_path, monkeypatch, env_files):
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
    assert (run_directory(tmp_path / ".artifact") / "model.onnx").is_file()
    onnx.checker.check_model(onnx.load(run_directory(tmp_path / ".artifact") / "model.onnx"))


def test_report_directory_holds_only_the_report(tmp_path, monkeypatch, env_files):
    """The export must not leave an .onnx file next to report.json for the host to clean up."""
    seen = {}
    real_run_process = model_check.run_process

    def spying_run_process(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        code = real_run_process(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level)
        seen["files"] = sorted(f.name for f in Path(report_dir).iterdir())
        return code

    monkeypatch.setattr(model_check, "run_process", spying_run_process)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert seen["files"] == [model_check.REPORT_FILE]


def test_failed_export_writes_no_onnx_file(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, UNSUPPORTED_OP), write_npz(tmp_path, [X]))

    assert state["error"] == "onnx_convert"
    assert onnx_files(tmp_path) == []


def test_unwritable_artifact_directory_is_its_own_stage(tmp_path, env_files):
    config, environments = env_files
    (tmp_path / "blocked").write_text("not a directory")

    state = run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]),
                      environments_path=environments, artifacts_path=tmp_path / "blocked")

    assert state["outcome"] == "failure"
    assert state["error"] == "model_check_run"
    assert "blocked" in state["customer_message"]


def test_failing_model_proto_is_a_stage_not_an_escape(monkeypatch, caplog, good_model, artifacts):
    """model_proto materializes the IR; if that raises after the file is saved, it must still be a stage."""

    class SavesButCannotMaterialize:
        def save(self, path):
            Path(path).write_bytes(b"saved")

        @property
        def model_proto(self):
            raise MemoryError("cannot materialize the proto")

    monkeypatch.setattr(onnx_check, "export", lambda model, inputs: SavesButCannotMaterialize())

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "onnx_save"
    assert "materialize" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["onnx_save"]


def test_fp16_casts_a_copy_for_export_and_keeps_the_eager_model_fp32(monkeypatch, good_model, artifacts):
    seen = {}
    real_export = onnx_check.export

    def recording_export(model, inputs):
        seen["param_dtype"] = next(model.parameters()).dtype
        seen["input_dtypes"] = [t.dtype for t in inputs]
        return real_export(model, inputs)

    monkeypatch.setattr(onnx_check, "export", recording_export)

    state = eager_check.run(*good_model, artifacts, precision="fp16", device="cpu")

    assert state.get("error") is None
    assert seen == {"param_dtype": torch.float16, "input_dtypes": [torch.float16]}
    assert next(state["model"].parameters()).dtype == torch.float32
    assert [t.dtype for t in state["user_inputs"]] == [torch.float32]
    exported = onnx.load(artifacts / "model.onnx")
    assert exported.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


HALF_MODEL = CONV_RELU.replace("self.relu = torch.nn.ReLU()", "self.relu = torch.nn.ReLU()\n        self.half()").replace(
    "return [torch.randn(1, 3, 8, 8)]", "return [torch.randn(1, 3, 8, 8).half()]"
)


def test_fp32_casts_a_half_model_up_and_keeps_the_eager_model_half(monkeypatch, tmp_path, artifacts):
    """Selecting fp32 must apply fp32, not merely leave whatever the model already is."""
    model = write_model(tmp_path, HALF_MODEL)
    npz = write_npz(tmp_path, [X.astype(np.float16)])
    seen = {}
    real_export = onnx_check.export

    def recording_export(m, inputs):
        seen["param_dtype"] = next(m.parameters()).dtype
        seen["input_dtypes"] = [t.dtype for t in inputs]
        return real_export(m, inputs)

    monkeypatch.setattr(onnx_check, "export", recording_export)

    state = eager_check.run(model, npz, artifacts, precision="fp32", device="cpu")

    assert state.get("error") is None
    assert seen == {"param_dtype": torch.float32, "input_dtypes": [torch.float32]}
    assert next(state["model"].parameters()).dtype == torch.float16
    assert [t.dtype for t in state["user_inputs"]] == [torch.float16]
    assert onnx.load(artifacts / "model.onnx").graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_no_copy_when_the_model_already_has_the_precision(monkeypatch, good_model, artifacts):
    seen = {}
    real_export = onnx_check.export
    monkeypatch.setattr(onnx_check, "export", lambda m, inputs: seen.setdefault("model", m) and real_export(m, inputs))

    state = eager_check.run(*good_model, artifacts, precision="fp32", device="cpu")

    assert state.get("error") is None
    assert seen["model"] is state["model"]


def test_fp16_leaves_integer_inputs_alone(monkeypatch, tmp_path, artifacts):
    model = tmp_path / "model.py"
    model.write_text(
        "import torch\n\n\nclass Model(torch.nn.Module):\n"
        "    def forward(self, x, idx):\n        return x[idx]\n\n\n"
        "def get_inputs():\n    return [torch.randn(4, 3), torch.tensor([0, 2])]\n"
    )
    npz = tmp_path / "input.npz"
    np.savez(npz, np.zeros((4, 3), dtype=np.float32), np.array([0, 2], dtype=np.int64))
    seen = {}
    real_export = onnx_check.export
    monkeypatch.setattr(onnx_check, "export", lambda m, inputs: seen.setdefault("dtypes", [t.dtype for t in inputs]) and real_export(m, inputs))

    state = eager_check.run(model, npz, artifacts, precision="fp16", device="cpu")

    assert state.get("error") is None
    assert seen["dtypes"] == [torch.float16, torch.int64]


def test_cast_failure_is_its_own_stage(monkeypatch, caplog, good_model, artifacts):
    def refuse_half(model, dtype):
        raise RuntimeError("this module cannot be cast to half")

    monkeypatch.setattr(onnx_check, "cast_copy", refuse_half)

    state = eager_check.run(*good_model, artifacts, precision="fp16", device="cpu")

    assert state["error"] == "onnx_precision"
    assert "cannot be cast" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["onnx_precision"]


def test_export_kept_in_state_for_the_next_node(good_model, artifacts):
    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert isinstance(state["onnx_model"], onnx.ModelProto)

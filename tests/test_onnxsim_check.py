"""The ONNX simplification check: onnxsim runs on the exported model and the result lands in the artifact directory."""

import onnx
import pytest

from conftest import run_directory
from kronos_edge import eager_check, onnxsim_check
from test_eager_check import CONV_RELU, X, error_records, run, write_model, write_npz


def test_simplified_model_is_valid_and_written(caplog, good_model, artifacts):
    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    simplified = state["simplified_onnx"]
    assert isinstance(simplified, onnx.ModelProto)
    onnx.checker.check_model(simplified)
    exported = state["onnx_model"]
    assert [i.name for i in simplified.graph.input] == [i.name for i in exported.graph.input]
    assert [o.name for o in simplified.graph.output] == [o.name for o in exported.graph.output]
    on_disk = onnx.load(artifacts / "model_simplified.onnx")
    assert on_disk.SerializeToString() == simplified.SerializeToString()
    assert error_records(caplog) == []


def test_success_message_names_simplification(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert "ONNX simplification" in state["customer_message"]


@pytest.mark.parametrize(
    "simplify, stage, fragment",
    [
        pytest.param(lambda model: (_ for _ in ()).throw(RuntimeError("onnxsim exploded")), "onnx_simplify", "exploded", id="simplify_raises"),
        pytest.param(lambda model: (model, False), "onnx_simplified_invalid", "check", id="simplify_check_fails"),
        pytest.param(lambda model: (onnx.ModelProto(), True), "onnx_simplified_invalid", "simplified", id="simplified_model_invalid"),
        pytest.param(lambda model: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'onnxsim'")), "onnx_simplifier", "support", id="onnxsim_missing"),
    ],
)
def test_failure_stages(tmp_path, caplog, monkeypatch, env_files, simplify, stage, fragment):
    monkeypatch.setattr(onnxsim_check, "simplify", simplify)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["tensorrt_error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    # The ONNX/TensorRT branch stops at the simplification; the kernelgen branch still runs.
    assert sorted(f.name for f in run_directory(tmp_path / ".artifact").iterdir()) == [
        "kernel.py", "kernelgen_guidance.txt", "kernelgen_report.json", "model.onnx", "model.pt2",
    ]


def test_large_simplified_model_is_saved_with_external_data(monkeypatch, good_model, artifacts):
    """Above the protobuf limit the weights go to a companion file; the checker runs by path."""
    monkeypatch.setattr(onnxsim_check, "EXTERNAL_DATA_THRESHOLD", 0)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert sorted(f.name for f in artifacts.iterdir()) == [
        "benchmark.json", "model.engine", "model.onnx", "model_simplified.onnx", "model_simplified.onnx.data"
    ]
    in_state = state["simplified_onnx"]
    assert in_state.graph.initializer and all(t.raw_data for t in in_state.graph.initializer)
    onnx.checker.check_model(in_state)  # the in-memory handoff still carries its weights
    reloaded = onnx.load(artifacts / "model_simplified.onnx")  # pulls the weights back in from the companion
    assert reloaded.SerializeToString() == in_state.SerializeToString()

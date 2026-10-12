"""The ONNX simplification check: onnxsim runs on the exported model and the result lands in the artifact directory."""

import onnx
import pytest

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
    ],
)
def test_failure_stages(tmp_path, caplog, monkeypatch, env_files, simplify, stage, fragment):
    monkeypatch.setattr(onnxsim_check, "simplify", simplify)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert (tmp_path / ".artifact" / "model.onnx").is_file()
    assert not (tmp_path / ".artifact" / "model_simplified.onnx").exists()

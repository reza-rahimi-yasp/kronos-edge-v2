"""The pt2 export check: right after the eager pass, the model is traced with torch.export and saved as the kernelgen reference."""

import torch

from kronos_edge import eager_check, onnx_check, pt2_check
from test_eager_check import CONV_RELU, X, error_records, run, write_model, write_npz
from test_onnx_check import DATA_DEPENDENT


def test_pt2_is_written_right_after_the_eager_run_and_before_onnx(tmp_path, monkeypatch, env_files):
    order = []
    real_pt2_export, real_onnx_export = pt2_check.export, onnx_check.export

    def pt2_export(model, inputs):
        order.append("pt2")
        return real_pt2_export(model, inputs)

    def onnx_export(model, inputs):
        order.append("onnx")
        assert (tmp_path / ".artifact" / "model.pt2").is_file(), "the reference must exist before the ONNX export starts"
        return real_onnx_export(model, inputs)

    monkeypatch.setattr(pt2_check, "export", pt2_export)
    monkeypatch.setattr(onnx_check, "export", onnx_export)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert order == ["pt2", "onnx"]


def test_pt2_failure_stops_before_the_onnx_export(tmp_path, monkeypatch, caplog, env_files):
    def refuse(model, inputs):
        raise RuntimeError("torch.export refused this model")

    monkeypatch.setattr(pt2_check, "export", refuse)
    monkeypatch.setattr(onnx_check, "export", lambda model, inputs: (_ for _ in ()).throw(AssertionError("onnx export must not run")))

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == "pt2_capture"
    assert "refused" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["pt2_capture"]
    assert list((tmp_path / ".artifact").iterdir()) == []


def test_exported_program_is_loadable_and_matches_the_eager_model(good_model, artifacts):
    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert isinstance(state["exported_program"], torch.export.ExportedProgram)
    loaded = torch.export.load(str(artifacts / "model.pt2"))
    x = state["user_inputs"][0]
    with torch.no_grad():
        expected = state["model"](x)
    torch.testing.assert_close(loaded.module()(x), expected)


def test_pt2_keeps_the_eager_precision(good_model, artifacts):
    """Precision is applied at the ONNX export only; the kernelgen reference is the model as written."""
    state = eager_check.run(*good_model, artifacts, precision="fp16", device="cpu")

    assert state.get("error") is None
    program = torch.export.load(str(artifacts / "model.pt2"))
    assert next(iter(program.state_dict.values())).dtype == torch.float32
    assert program.example_inputs[0][0].dtype == torch.float32


def test_untraceable_model_is_named_with_its_cause(tmp_path, caplog, env_files):
    state = run(env_files, write_model(tmp_path, DATA_DEPENDENT), write_npz(tmp_path, [X]))

    assert state["outcome"] == "failure"
    assert state["error"] == "pt2_capture"
    assert "could not be traced" in state["customer_message"]
    assert "data-dependent" in state["customer_message"]
    assert "\x1b[" not in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["pt2_capture"]


def test_unwritable_pt2_is_the_save_stage(monkeypatch, caplog, good_model, artifacts):
    def cannot_save(program, path):
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr(pt2_check, "save", cannot_save)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "pt2_save"
    assert "model.pt2" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["pt2_save"]

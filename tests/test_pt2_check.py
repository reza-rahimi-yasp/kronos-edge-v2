"""The pt2 export check: right after the eager pass, the model is traced with torch.export and saved as the kernelgen reference."""

import json
import zipfile
from pathlib import Path

import torch

from kronos_edge import eager_check, onnx_check, pt2_check
from test_eager_check import CONV_RELU, X, error_records, run, write_model, write_npz
from test_onnx_check import DATA_DEPENDENT


def test_pt2_is_written_right_after_the_eager_run_and_before_onnx(tmp_path, monkeypatch, env_files, official_exporter):
    order = []
    real_pt2_export, real_onnx_export = official_exporter.export, onnx_check.export

    def pt2_export(model, inputs):
        order.append("pt2")
        return real_pt2_export(model, inputs)

    def onnx_export(model, inputs):
        order.append("onnx")
        assert (tmp_path / ".artifact" / "model.pt2").is_file(), "the reference must exist before the ONNX export starts"
        return real_onnx_export(model, inputs)

    monkeypatch.setattr(official_exporter, "export", pt2_export)
    monkeypatch.setattr(onnx_check, "export", onnx_export)

    state = run(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]))

    assert state["outcome"] == "success"
    assert order == ["pt2", "onnx"]


def test_pt2_failure_stops_before_the_onnx_export(tmp_path, monkeypatch, caplog, env_files, official_exporter):
    def refuse(model, inputs):
        raise RuntimeError("torch.export refused this model")

    monkeypatch.setattr(official_exporter, "export", refuse)
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


def test_unwritable_pt2_is_the_save_stage(monkeypatch, caplog, good_model, artifacts, official_exporter):
    def cannot_save(program, path):
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr(official_exporter, "save", cannot_save)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "pt2_save"
    assert "model.pt2" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["pt2_save"]


def tensors_of(model, inputs):
    return [*model.parameters(), *model.buffers(), *inputs]


def test_reference_is_produced_by_the_official_exporter_on_cpu_copies(monkeypatch, good_model, artifacts, official_exporter):
    """The reference is yasp's own export (`yasp.compile.common.exporter`, what `yasp-toolkit kernelgen export`
    runs), on independent CPU copies: a GPU-traced reference failed to load on the kernelgen service as it does on
    a torch built without CUDA, and the eager model and the user's inputs in state keep their objects, storage,
    values, dtypes, and mode."""
    received = {}
    real_export, real_save = official_exporter.export, official_exporter.save

    def export(model, inputs, *args, **kwargs):
        received["model"], received["inputs"] = model, inputs
        received["program"] = real_export(model, inputs, *args, **kwargs)
        return received["program"]

    def save(program, path):
        received["saved"] = (program, Path(path))
        real_save(program, path)

    monkeypatch.setattr(official_exporter, "export", export)
    monkeypatch.setattr(official_exporter, "save", save)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert received["saved"] == (received["program"], artifacts / "model.pt2")
    assert state["exported_program"] is received["program"]
    model, inputs = received["model"], received["inputs"]
    assert model is not state["model"]
    assert all(t.device.type == "cpu" for t in tensors_of(model, inputs))
    assert model.training is state["model"].training is False
    originals = tensors_of(state["model"], state["user_inputs"])
    copies = tensors_of(model, inputs)
    assert len(copies) == len(originals)
    for copied, original in zip(copies, originals, strict=True):
        assert copied is not original
        assert copied.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        assert (copied.dtype, copied.shape) == (original.dtype, original.shape)
        torch.testing.assert_close(copied, original)


def test_a_trace_that_writes_to_its_model_or_inputs_cannot_reach_the_eager_ones(monkeypatch, good_model, artifacts, official_exporter):
    real_export = official_exporter.export

    def export(model, inputs):
        with torch.no_grad():
            for t in tensors_of(model, inputs):
                t.fill_(7.0)
        return real_export(model, inputs)

    monkeypatch.setattr(official_exporter, "export", export)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert not any(bool((t == 7.0).all()) for t in tensors_of(state["model"], state["user_inputs"]))


def test_cpu_copies_keep_every_dtype(artifacts):
    """Copying is not casting: half weights and integer buffers and inputs reach the trace as they are."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float16))
            self.register_buffer("shift", torch.tensor([1, 2, 3], dtype=torch.int64))

        def forward(self, x, steps):
            return x * self.scale + (self.shift + steps).to(x.dtype)

    model = Model().eval()
    inputs = (torch.ones(2, 3, dtype=torch.float16), torch.zeros(3, dtype=torch.int64))

    copied_model, copied_inputs = pt2_check.cpu_copies(model, inputs)

    for copied, original in zip(tensors_of(copied_model, copied_inputs), tensors_of(model, inputs), strict=True):
        assert copied.dtype == original.dtype
        assert copied.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    state = pt2_check.export_pt2({"model": model, "user_inputs": inputs, "artifacts_dir": artifacts})
    assert state.get("error") is None
    program = torch.export.load(str(artifacts / "model.pt2"))
    assert program.state_dict["scale"].dtype == torch.float16
    assert program.state_dict["shift"].dtype == torch.int64
    assert [t.dtype for t in program.example_inputs[0]] == [torch.float16, torch.int64]


def archive_devices(path):
    """Every device recorded in the archive's JSON: the graph's values and the weights."""
    devices = set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("device"), dict):
                devices.add(node["device"]["type"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                walk(json.loads(archive.read(name)))
    return devices


def test_saved_reference_records_only_cpu_devices(good_model, artifacts):
    """The archive bakes in the trace device; a cpu archive is the one shown to load on a torch built without CUDA."""
    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state.get("error") is None
    assert archive_devices(artifacts / "model.pt2") == {"cpu"}


def test_missing_exporter_package_is_the_pt2_exporter_stage(monkeypatch, caplog, good_model, artifacts):
    """yasp-compile-common is imported in the node, so an image without it reports a stage instead of crashing."""
    def missing():
        raise ImportError("No module named 'yasp'")

    monkeypatch.setattr(pt2_check, "import_exporter", missing)
    monkeypatch.setattr(onnx_check, "export", lambda model, inputs: (_ for _ in ()).throw(AssertionError("onnx export must not run")))

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "pt2_exporter"
    assert "No module named 'yasp'" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["pt2_exporter"]
    assert list(artifacts.iterdir()) == []

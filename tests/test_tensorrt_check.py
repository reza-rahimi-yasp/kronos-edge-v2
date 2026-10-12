"""The TensorRT engine check: builds from the simplified ONNX at the configured precision into the artifact directory.

TensorRT exists only in the image, so `tensorrt_check.build` is stubbed by an autouse fixture;
the tests here cover the node around it and the real import failure.
"""

import os

import onnx
import pytest

from conftest import CONFIG, FAKE_ENGINE, write_yaml
from kronos_edge import eager_check, tensorrt_check
from kronos_edge.workflow import run_check
from test_eager_check import CONV_RELU, X, error_records, write_model, write_npz


def run_with_precision(tmp_path, env_files, precision):
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, "precision": precision})
    return run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_engine_is_built_from_the_simplified_model_and_named_in_the_message(tmp_path, monkeypatch, env_files, precision):
    """The precision is applied at the ONNX export; the builder reads it from the file (see test_onnx_check)."""
    seen = {}

    def recording_build(onnx_path, inputs):
        seen["onnx_path"] = onnx_path
        seen["inputs"] = inputs
        return FAKE_ENGINE

    monkeypatch.setattr(tensorrt_check, "build", recording_build)

    state = run_with_precision(tmp_path, env_files, precision)

    assert state["outcome"] == "success"
    assert f"TensorRT engine build ({precision})" in state["customer_message"]
    assert seen["onnx_path"].name == "model_simplified.onnx" and seen["onnx_path"].is_file()
    assert [tuple(t.shape) for t in seen["inputs"]] == [X.shape]  # the user's inputs, for shape-input profiles
    assert (tmp_path / ".artifact" / "model.engine").read_bytes() == FAKE_ENGINE


def test_engine_bytes_stay_in_state_for_the_next_node(good_model, artifacts):
    state = eager_check.run(*good_model, artifacts, precision="fp32", device="cpu")

    assert state.get("error") is None
    assert state["engine"] == FAKE_ENGINE
    assert (artifacts / "model.engine").read_bytes() == FAKE_ENGINE


def test_changing_precision_overwrites_the_engine(tmp_path, monkeypatch, env_files):
    """The builder sees the ONNX of the new precision; the fake echoes the input dtype it was given."""
    monkeypatch.setattr(
        tensorrt_check, "build",
        lambda onnx_path, inputs: onnx.TensorProto.DataType.Name(onnx.load(onnx_path).graph.input[0].type.tensor_type.elem_type).encode(),
    )
    engine = tmp_path / ".artifact" / "model.engine"

    assert run_with_precision(tmp_path, env_files, "fp32")["outcome"] == "success"
    assert engine.read_bytes() == b"FLOAT"
    assert run_with_precision(tmp_path, env_files, "fp16")["outcome"] == "success"
    assert engine.read_bytes() == b"FLOAT16"


@pytest.mark.parametrize(
    "build, stage, fragment",
    [
        pytest.param(
            lambda p, i: (_ for _ in ()).throw(tensorrt_check.TensorRTParseError("No importer registered for op: Foo")),
            "tensorrt_parse", "op: Foo", id="parser_rejects_an_operator",
        ),
        pytest.param(
            lambda p, i: (_ for _ in ()).throw(tensorrt_check.TensorRTBuildError("out of memory while tactic timing")),
            "tensorrt_build", "out of memory", id="builder_returns_no_engine",
        ),
        pytest.param(
            lambda p, i: (_ for _ in ()).throw(RuntimeError("CUDA driver mismatch")),
            "tensorrt_build", "driver mismatch", id="unexpected_exception",
        ),
    ],
)
def test_failure_stages(tmp_path, caplog, monkeypatch, env_files, build, stage, fragment):
    monkeypatch.setattr(tensorrt_check, "build", build)

    state = run_with_precision(tmp_path, env_files, "fp32")

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == [stage]
    assert sorted(f.name for f in (tmp_path / ".artifact").iterdir()) == ["model.onnx", "model_simplified.onnx"]  # no engine, no benchmark


def test_missing_tensorrt_is_a_support_failure(monkeypatch, caplog, good_model, artifacts, tensorrt_stub):
    """The venv has no tensorrt; the real build() must turn that into a stage, not a crash."""
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("tensorrt is installed here; the import stage cannot be exercised")
    monkeypatch.setattr(tensorrt_check, "build", tensorrt_stub)

    state = eager_check.run(*good_model, artifacts, precision="fp32", device="cpu")

    assert state["error"] == "tensorrt_import"
    assert "tensorrt" in state["error_detail"]
    assert "support" in tensorrt_check.CUSTOMER_MESSAGES["tensorrt_import"]
    assert [r.stage for r in error_records(caplog)] == ["tensorrt_import"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only directory")
def test_unwritable_artifact_directory_is_the_save_stage(monkeypatch, good_model, artifacts):
    real_build = tensorrt_check.build

    def build_then_lock(onnx_path, inputs):
        engine = real_build(onnx_path, inputs)
        artifacts.chmod(0o500)
        return engine

    monkeypatch.setattr(tensorrt_check, "build", build_then_lock)
    try:
        state = eager_check.run(*good_model, artifacts, precision="fp32", device="cpu")
    finally:
        artifacts.chmod(0o700)

    assert state["error"] == "engine_save"
    assert "model.engine" in state["error_detail"]

import logging

import numpy as np
import pytest

from kronos_edge.eager_check import main, route_on_error, run_check

CONV_RELU = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""

TYPO = CONV_RELU.replace("self.relu(self.conv(x))", "self.relo(self.conv(x))")
FWD_ARGS_ALIAS = CONV_RELU.replace("def get_inputs", "def get_fwd_args")
NO_MODEL_CLASS = CONV_RELU.replace("class Model(", "class Net(")
NO_GET_INPUTS = CONV_RELU.split("def get_inputs")[0]
GET_INPUTS_RAISES = CONV_RELU.replace(
    "return [torch.randn(1, 3, 8, 8)]", 'raise RuntimeError("boom")'
)

MODEL_NOT_MODULE = """
import torch


class Model:
    pass


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""

INIT_INPUTS = """
import torch


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.conv(x)


def get_init_inputs():
    return [3, 4]


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""
INIT_INPUTS_MISSING = INIT_INPUTS.replace("def get_init_inputs():\n    return [3, 4]\n", "")

TWO_INPUTS = """
import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.randn(2, 3), torch.randn(2, 3)]
"""

SYNTAX_ERROR = "def (:\n"

X = np.zeros((1, 3, 8, 8), dtype=np.float32)
XY = [np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]


def write_model(tmp_path, source):
    path = tmp_path / "model.py"
    if source is not None:
        path.write_text(source)
    return path


def write_npz(tmp_path, arrays):
    path = tmp_path / "input.npz"
    if isinstance(arrays, dict):
        np.savez(path, **arrays)
    elif arrays is not None:
        np.savez(path, *arrays)
    return path


def error_records(state):
    return [r for r in state["collector"].records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize(
    "source, arrays",
    [
        pytest.param(CONV_RELU, [X], id="conv_relu"),
        pytest.param(FWD_ARGS_ALIAS, [X], id="get_fwd_args_alias"),
        pytest.param(INIT_INPUTS, [X], id="get_init_inputs"),
        pytest.param(TWO_INPUTS, XY, id="two_inputs"),
    ],
)
def test_success(tmp_path, source, arrays):
    state = run_check(write_model(tmp_path, source), write_npz(tmp_path, arrays))

    assert state["outcome"] == "success"
    assert state.get("error") is None
    assert "succeeded" in state["customer_message"]
    assert error_records(state) == []


@pytest.mark.parametrize(
    "source, arrays, stage, fragment",
    [
        pytest.param(TYPO, [X], "forward", "cannot be run through eager mode", id="forward_typo"),
        pytest.param(None, [X], "import_module", "could not be loaded", id="model_file_missing"),
        pytest.param(SYNTAX_ERROR, [X], "import_module", "could not be loaded", id="syntax_error"),
        pytest.param(NO_MODEL_CLASS, [X], "find_model_class", "does not define a 'Model' class", id="no_model_class"),
        pytest.param(MODEL_NOT_MODULE, [X], "find_model_class", "does not define a 'Model' class", id="model_not_module"),
        pytest.param(NO_GET_INPUTS, [X], "find_get_inputs", "get_inputs() or get_fwd_args()", id="no_get_inputs"),
        pytest.param(INIT_INPUTS_MISSING, [X], "init_model", "get_init_inputs", id="init_inputs_missing"),
        pytest.param(GET_INPUTS_RAISES, [X], "get_inputs", "boom", id="get_inputs_raises"),
        pytest.param(CONV_RELU, None, "load_npz", "could not be read", id="npz_file_missing"),
        pytest.param(TWO_INPUTS, [X], "validate_inputs", "arr_1", id="npz_wrong_count"),
        pytest.param(CONV_RELU, {"x": X}, "validate_inputs", "arr_0", id="npz_wrong_key"),
        pytest.param(CONV_RELU, [np.zeros((1, 3, 9, 9), dtype=np.float32)], "validate_inputs", "shape", id="shape_mismatch"),
        pytest.param(CONV_RELU, [np.zeros((1, 3, 8, 8), dtype=np.float64)], "validate_inputs", "dtype", id="dtype_mismatch"),
    ],
)
def test_failure(tmp_path, source, arrays, stage, fragment):
    state = run_check(write_model(tmp_path, source), write_npz(tmp_path, arrays))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    records = error_records(state)
    assert len(records) == 1
    assert records[0].stage == stage


def test_forward_failure_includes_exception_text(tmp_path):
    state = run_check(write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert "relo" in state["customer_message"]


def test_init_inputs_missing_names_required_arguments(tmp_path):
    state = run_check(write_model(tmp_path, INIT_INPUTS_MISSING), write_npz(tmp_path, [X]))

    assert "in_channels" in state["customer_message"]
    assert "out_channels" in state["customer_message"]


def test_route_on_error():
    assert route_on_error({"error": "forward"}) == "failure"
    assert route_on_error({}) == "ok"


def test_main_success_exit_code(tmp_path, capsys):
    code = main([str(write_model(tmp_path, CONV_RELU)), str(write_npz(tmp_path, [X]))])

    assert code == 0
    assert "succeeded" in capsys.readouterr().out


def test_main_failure_exit_code(tmp_path, capsys):
    code = main([str(write_model(tmp_path, TYPO)), str(write_npz(tmp_path, [X]))])

    assert code == 1
    assert "cannot be run through eager mode" in capsys.readouterr().out

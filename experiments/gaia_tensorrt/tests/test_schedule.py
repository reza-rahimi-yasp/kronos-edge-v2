"""The model-specific op schedule reproduces kernel.py's forward and validates shapes without torch."""


# The exporter modules are plain scripts, not a package: make them importable from this test alone.
# (No conftest.py here: pytest imports conftest modules by that bare name, and a second one would
# shadow the product's tests/conftest.py, which its tests import from directly.)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))

import json
from pathlib import Path

import pytest

import schedule

EXPERIMENT = Path(__file__).resolve().parents[1]
SIGNATURE = json.loads((EXPERIMENT / "exporter" / "model_signature.json").read_text())


@pytest.fixture(scope="module")
def ops():
    return schedule.build(SIGNATURE)


def test_parameters_consumed_in_order_and_head_unused(ops):
    used = [i for op in ops for i in schedule.parameter_indices(op)]
    assert sorted(used) == list(range(159)) and len(used) == 159  # the classification head is never reached
    assert len(SIGNATURE["parameters"]) == 167
    assert all(p["fqn"].startswith("m.head.") for p in SIGNATURE["parameters"][159:])


def test_batch_norms_consumed_in_order(ops):
    bns = [a["bn"] for op in ops for a in schedule.affines(op) if a["mode"] == 2]
    assert bns == list(range(34))
    assert len(SIGNATURE["buffers"]) == 35 * 3
    assert SIGNATURE["buffers"][34 * 3]["fqn"].startswith("m.head.")


def test_op_counts(ops):
    kinds = {}
    for op in ops:
        kinds[op["kind"]] = kinds.get(op["kind"], 0) + 1
    # stem: conv, dw(emit), conv+epilogue; 14 MBConv blocks: conv, dw, conv+epilogue; 7 LiteMLA blocks:
    # conv, dw(5x5), group conv, attn_prep, 2 bmm, attn_final, conv+epilogue.
    assert kinds == {"conv": 2 + 2 * 14 + 3 * 7, "dw": 1 + 14 + 7, "epilogue": 1 + 14 + 7, "attn_prep": 7, "bmm": 14, "attn_final": 7}
    # Cross-check with the decomposed PT2 graph: every conv and depthwise conv is one aten.convolution there.
    counts = SIGNATURE["pt2_op_counts"]
    assert kinds["conv"] + kinds["dw"] == counts["aten.convolution.default"]
    assert kinds["bmm"] == counts["aten.bmm.default"]
    assert sum(1 for op in ops for a in schedule.affines(op) if a["mode"] == 2) == counts["aten._native_batch_norm_legit_no_training.default"]


def test_dw_configs_are_the_ones_gaia_compiled(ops):
    configs = {(op["k"], op["s"]) for op in ops if op["kind"] == "dw"}
    assert configs == {(3, 1), (3, 2), (5, 1)}
    emitting = [op for op in ops if op["kind"] == "dw" and op["emit"]]
    assert len(emitting) == 1 and emitting[0]["s"] == 1 and emitting[0]["pre"] == 1


def test_shapes_of_the_measured_input(ops):
    shapes = schedule.infer_shapes(ops, SIGNATURE, SIGNATURE["input"]["shape"])
    assert shapes[ops[-1]["out"]] == SIGNATURE["output"]["shape"]
    assert shapes["x"] == [4, 3, 1088, 1920]
    prep = [op for op in ops if op["kind"] == "attn_prep"][0]
    # the first attention block runs at stride 16 with 384 = 48 * 8 channels: 8 heads of q, k, v, each 16 wide
    assert shapes[prep["out"][0]] == [4 * 2 * 8, 16, 68 * 120] and shapes[prep["out"][2]] == [64, 68 * 120, 17]


@pytest.mark.parametrize(
    "input_shape, message",
    [
        pytest.param([1, 3, 10, 10], "multiple of 4", id="epilogue_hw_not_multiple_of_4"),
        pytest.param([4, 4, 1088, 1920], "channels", id="wrong_input_channels"),
        pytest.param([4, 3, 1088], "4-D", id="wrong_rank"),
    ],
)
def test_unsupported_inputs_are_rejected(ops, input_shape, message):
    with pytest.raises(schedule.Unsupported, match=message):
        schedule.infer_shapes(ops, SIGNATURE, input_shape)


def test_weight_shapes_match_their_ops(ops):
    params = SIGNATURE["parameters"]
    for op in ops:
        if op["kind"] == "dw":
            assert params[op["w"]]["shape"][1:] == [1, op["k"], op["k"]]
        if op["kind"] == "conv":
            shape = params[op["w"]]["shape"]
            assert len(shape) == 4 and shape[2] == shape[3] == op["kernel"]
            assert shape[0] % op["groups"] == 0

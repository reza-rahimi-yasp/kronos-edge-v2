"""The whole-model program encodes kernel.py's schedule completely, round-trips, and rejects unsupported inputs."""

# The exporter modules are plain scripts, not a package: make them importable from this test alone.
# (No conftest.py here, for the reason the hybrid experiment's tests give.)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "exporter"))

import json
from pathlib import Path

import numpy as np
import pytest

import schedule
import whole_program

EXPERIMENT = Path(__file__).resolve().parents[2]
SIGNATURE = json.loads((EXPERIMENT / "exporter" / "model_signature.json").read_text())
INPUT_SHAPE = SIGNATURE["input"]["shape"]


@pytest.fixture(scope="module")
def program():
    return whole_program.build(SIGNATURE, INPUT_SHAPE)


@pytest.fixture(scope="module")
def decoded(program):
    return whole_program.decode(program.ints)


def test_header_and_round_trip(program, decoded):
    header = decoded["header"]
    assert header["magic"] == whole_program.MAGIC and header["version"] == whole_program.VERSION
    assert header["n_ops"] == len(program.ops) == len(decoded["ops"])
    assert header["n_tensors"] == len(decoded["tensors"]) == len(program.tensor_ids)
    assert header["n_floats"] == program.n_floats
    assert header["input"] == program.tensor_ids["x"] and header["output"] == program.tensor_ids[program.ops[-1]["out"]]
    assert len(program.ints) == whole_program.HEADER_WORDS + header["n_tensors"] * whole_program.TENSOR_WORDS + header["n_ops"] * whole_program.OP_WORDS


def test_every_op_of_the_schedule_is_encoded_in_order(program, decoded):
    assert [op["kind"] for op in decoded["ops"]] == [op["kind"] for op in program.ops]
    kinds = {}
    for op in decoded["ops"]:
        kinds[op["kind"]] = kinds.get(op["kind"], 0) + 1
    assert kinds == {"conv": 51, "dw": 22, "epilogue": 22, "attn_prep": 7, "bmm": 14, "attn_final": 7}


def test_tensor_shapes_are_the_schedules(program, decoded):
    for name, tid in program.tensor_ids.items():
        assert decoded["tensors"][tid]["shape"] == program.shapes[name], name
    assert decoded["tensors"][decoded["header"]["input"]]["shape"] == INPUT_SHAPE
    assert decoded["tensors"][decoded["header"]["output"]]["shape"] == SIGNATURE["output"]["shape"]


def test_epilogues_run_in_place_and_nothing_else_aliases(program, decoded):
    aliased = {i for i, t in enumerate(decoded["tensors"]) if t["alias"] != whole_program.NONE}
    epilogue_outputs = {op["out"] for op in decoded["ops"] if op["kind"] == "epilogue"}
    assert aliased == epilogue_outputs and len(aliased) == 22
    for op in decoded["ops"]:
        if op["kind"] == "epilogue":
            assert decoded["tensors"][op["out"]]["alias"] == op["x"]  # the output is the input buffer, like epilogue_kernel
            assert op["res"] == whole_program.NONE or op["res"] != op["x"]  # the residual is never the tensor being updated
            assert decoded["tensors"][op["x"]]["alias"] == whole_program.NONE  # a conv output, never an alias itself
    # the final output is the last epilogue, in place on the last conv's buffer
    last = decoded["ops"][-1]
    assert last["kind"] == "epilogue" and last["out"] == decoded["header"]["output"]


def test_operands_are_defined_before_use(decoded):
    defined = {decoded["header"]["input"]}
    producers = {"conv": ["out"], "dw": ["out", "out_pre"], "epilogue": ["out"], "attn_prep": ["q", "k", "vpad"], "bmm": ["out"], "attn_final": ["out"]}
    consumers = {"conv": ["x"], "dw": ["x"], "epilogue": ["x", "res"], "attn_prep": ["qkv", "agg"], "bmm": ["a", "b"], "attn_final": ["r", "ref"]}
    for op in decoded["ops"]:
        for key in consumers[op["kind"]]:
            if op[key] != whole_program.NONE:
                assert op[key] in defined, (op["kind"], key)
        for key in producers[op["kind"]]:
            if op[key] != whole_program.NONE:
                assert op[key] not in defined
                defined.add(op[key])
    assert defined == set(range(decoded["header"]["n_tensors"]))


def test_weight_layout_covers_every_used_vector_once_and_is_aligned(program):
    keys = [entry["key"] for entry in program.layout]
    assert len(keys) == len(set(keys))
    assert sorted(k for k in keys if k.startswith("p")) == sorted(f"p{i}" for i in range(159))  # the head's 8 parameters are unused
    assert sorted(k for k in keys if k.startswith("mean")) == sorted(f"mean{i}" for i in range(34))
    assert sorted(k for k in keys if k.startswith("invstd")) == sorted(f"invstd{i}" for i in range(34))
    end = 0
    for entry in program.layout:
        assert entry["offset"] % whole_program.ALIGN_FLOATS == 0 and entry["offset"] >= end
        assert entry["count"] == int(np.prod(entry["shape"]))
        end = entry["offset"] + entry["count"]
    assert end <= program.n_floats < end + whole_program.ALIGN_FLOATS
    params = {f"p{i}": p["shape"] for i, p in enumerate(SIGNATURE["parameters"])}
    for entry in program.layout:
        if entry["key"] in params:
            assert entry["shape"] == params[entry["key"]]


def test_op_records_reference_weights_inside_the_blob(program, decoded):
    offsets = {entry["offset"]: entry for entry in program.layout}
    weight_keys = {"conv": ("w",), "dw": ("w", "in_g", "in_b", "in_m", "in_is", "out_g", "out_b", "out_m", "out_is"), "epilogue": ("g", "b", "m", "is")}
    for op in decoded["ops"]:
        for key in weight_keys.get(op["kind"], ()):
            if op[key] != whole_program.NONE:
                assert op[key] in offsets, (op["kind"], key)
        if op["kind"] == "conv":
            assert offsets[op["w"]]["shape"] == [op["cout"], op["cin_g"], op["k"], op["k"]]
        if op["kind"] == "dw":
            assert offsets[op["w"]]["shape"][1:] == [1, op["k"], op["k"]]
            for prefix in ("in", "out"):
                mode = op[f"{prefix}_mode"]
                present = [op[f"{prefix}_{k}"] != whole_program.NONE for k in ("g", "b", "m", "is")]
                assert present == {2: [True] * 4, 1: [False, True, False, False], 0: [False] * 4}[mode]


def test_pack_weights_places_each_vector_at_its_offset(program):
    arrays = {entry["key"]: np.full(entry["shape"], float(i + 1), dtype=np.float32) for i, entry in enumerate(program.layout)}
    blob = whole_program.pack_weights(program, arrays)
    assert blob.dtype == np.float32 and blob.shape == (program.n_floats,)
    for i, entry in enumerate(program.layout):
        assert (blob[entry["offset"]:entry["offset"] + entry["count"]] == i + 1).all()
    assert blob[entry["offset"] + entry["count"]:].sum() == 0  # the padding is zero
    wrong = dict(arrays)
    wrong["p0"] = np.zeros((1, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="p0"):
        whole_program.pack_weights(program, wrong)


@pytest.mark.parametrize(
    "input_shape, message",
    [
        pytest.param([1, 3, 10, 10], "multiple of 4", id="epilogue_hw_not_multiple_of_4"),
        pytest.param([4, 4, 1088, 1920], "channels", id="wrong_input_channels"),
        pytest.param([4, 3, 1088], "4-D", id="wrong_rank"),
    ],
)
def test_unsupported_inputs_are_rejected_before_encoding(input_shape, message):
    with pytest.raises(schedule.Unsupported, match=message):
        whole_program.build(SIGNATURE, input_shape)


@pytest.mark.parametrize(
    "mutate, message",
    [
        pytest.param(lambda ints: ints[:-1], "words", id="truncated"),
        pytest.param(lambda ints: [ints[0] + 1] + ints[1:], "not a whole-model program", id="bad_magic"),
        pytest.param(lambda ints: ints[:whole_program.HEADER_WORDS + whole_program.TENSOR_WORDS * ints[2]] + [99] + ints[whole_program.HEADER_WORDS + whole_program.TENSOR_WORDS * ints[2] + 1:], "unknown op kind", id="bad_kind"),
    ],
)
def test_malformed_programs_are_rejected(program, mutate, message):
    with pytest.raises(ValueError, match=message):
        whole_program.decode(mutate(list(program.ints)))

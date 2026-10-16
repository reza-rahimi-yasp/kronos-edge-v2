"""kernel.py's forward as one plugin's program: an INT32 op table plus one FP32 weight blob.

The op sequence is ``schedule.build`` of the hybrid experiment, unchanged; this
module only lays it out for the whole-model plugin, which decodes exactly this
(``plugin/gaia_whole_model_plugin.cu``):

- tensors get integer ids in order of definition. An epilogue's output aliases
  its input: the generated ``epilogue_kernel`` updates its tensor in place and
  the plugin runs it that way, so the two names are one buffer;
- every weight vector the ops consume (conv and depthwise weights, gamma, beta,
  running mean, inverse standard deviation) gets a 64-byte-aligned offset in
  one float blob, in order of first use, so one ``PluginField`` carries the
  whole model;
- ops are fixed-width INT32 records (``OP_WORDS`` each, ``-1`` for unused words).

Pure Python apart from ``pack_weights``, which needs numpy for the blob.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "exporter"))  # the hybrid experiment's schedule

import schedule

MAGIC = 0x47574D31  # "GWM1"
VERSION = 1
HEADER_WORDS = 10
TENSOR_WORDS = 6  # ndim, d0, d1, d2, d3, alias (-1: own buffer)
OP_WORDS = 20
ALIGN_FLOATS = 16  # 64 bytes: cuDNN's largest tensor alignment, and float4 loads
KINDS = {"conv": 1, "dw": 2, "epilogue": 3, "attn_prep": 4, "bmm": 5, "attn_final": 6}
KIND_NAMES = {v: k for k, v in KINDS.items()}
NONE = -1

# Word meaning per kind, after word 0 (the kind). Documents the records; the C++ decoder uses the same order.
LAYOUT = {
    "conv": ("x", "out", "w", "cout", "cin_g", "k", "stride", "pad", "groups"),
    "dw": ("x", "out", "w", "k", "s", "pre", "act", "emit", "out_pre",
           "in_mode", "in_g", "in_b", "in_m", "in_is", "out_mode", "out_g", "out_b", "out_m", "out_is"),
    "epilogue": ("x", "out", "mode", "g", "b", "m", "is", "act", "res"),
    "attn_prep": ("qkv", "agg", "q", "k", "vpad"),
    "bmm": ("a", "b", "out", "transpose_a"),
    "attn_final": ("r", "ref", "out"),
}


class Program:
    """The encoded program plus the maps the exporter and the tests need."""

    def __init__(self, ints, layout, n_floats, tensor_ids, shapes, ops):
        self.ints = ints              # the INT32 field, header first
        self.layout = layout          # [{"key", "offset", "count", "shape"}], weight blob layout in order of first use
        self.n_floats = n_floats      # length of the weight blob
        self.tensor_ids = tensor_ids  # schedule tensor name -> id
        self.shapes = shapes          # schedule tensor name -> shape
        self.ops = ops                # the schedule's ops


def build(signature: dict, input_shape: list[int]) -> Program:
    """Encode the schedule for ``input_shape``; raises ``schedule.Unsupported`` for shapes the ops cannot take."""
    ops = schedule.build(signature)
    shapes = schedule.infer_shapes(ops, signature, input_shape)
    params = [p["shape"] for p in signature["parameters"]]
    buffers = [b["shape"] for b in signature["buffers"]]

    tensor_ids, tensors = {}, []

    def define(name, alias=None):
        if name in tensor_ids:
            raise ValueError(f"tensor {name} defined twice")
        tensor_ids[name] = len(tensors)
        tensors.append((shapes[name], NONE if alias is None else tensor_ids[alias]))
        return tensor_ids[name]

    layout, offsets = [], {}
    cursor = 0

    def weight(key, shape):
        nonlocal cursor
        if key not in offsets:
            count = 1
            for d in shape:
                count *= d
            offsets[key] = cursor
            layout.append({"key": key, "offset": cursor, "count": count, "shape": list(shape)})
            cursor += (count + ALIGN_FLOATS - 1) // ALIGN_FLOATS * ALIGN_FLOATS
        return offsets[key]

    def affine_words(aff):
        if aff["mode"] == 2:
            mean, var = schedule.bn_buffers(aff)
            return [2, weight(f"p{aff['gamma']}", params[aff["gamma"]]), weight(f"p{aff['beta']}", params[aff["beta"]]),
                    weight(f"mean{aff['bn']}", buffers[mean]), weight(f"invstd{aff['bn']}", buffers[var])]
        if aff["mode"] == 1:
            return [1, NONE, weight(f"p{aff['beta']}", params[aff["beta"]]), NONE, NONE]
        return [0, NONE, NONE, NONE, NONE]

    define("x")
    records = []
    for op in ops:
        kind = op["kind"]
        if kind == "conv":
            cout, cin_g, k, _ = params[op["w"]]
            rec = [KINDS[kind], tensor_ids[op["x"]], define(op["out"]), weight(f"p{op['w']}", params[op["w"]]),
                   cout, cin_g, k, op["stride"], op["pad"], op["groups"]]
        elif kind == "dw":
            x = tensor_ids[op["x"]]
            in_aff, out_aff = affine_words(op["in_aff"]), affine_words(op["out_aff"])
            rec = [KINDS[kind], x, define(op["out"]), weight(f"p{op['w']}", params[op["w"]]), op["k"], op["s"], op["pre"], op["act"],
                   op["emit"], define(op["out_pre"]) if op["emit"] else NONE] + in_aff + out_aff
        elif kind == "epilogue":
            x = tensor_ids[op["x"]]
            rec = [KINDS[kind], x, define(op["out"], alias=op["x"])] + affine_words(op["aff"]) + [op["act"], tensor_ids[op["res"]] if op["res"] is not None else NONE]
        elif kind == "attn_prep":
            q, k, vpad = op["out"]
            rec = [KINDS[kind], tensor_ids[op["qkv"]], tensor_ids[op["agg"]], define(q), define(k), define(vpad)]
        elif kind == "bmm":
            rec = [KINDS[kind], tensor_ids[op["a"]], tensor_ids[op["b"]], define(op["out"]), int(op["transpose_a"])]
        elif kind == "attn_final":
            rec = [KINDS[kind], tensor_ids[op["r"]], tensor_ids[op["ref"]], define(op["out"])]
        else:
            raise ValueError(kind)
        if len(rec) > OP_WORDS:
            raise ValueError(f"{kind} record has {len(rec)} words")
        records.append(rec + [NONE] * (OP_WORDS - len(rec)))

    n_floats = cursor
    output_id = tensor_ids[ops[-1]["out"]]
    ints = [MAGIC, VERSION, len(tensors), len(records), n_floats, tensor_ids["x"], output_id, TENSOR_WORDS, OP_WORDS, 0]
    for shape, alias in tensors:
        if len(shape) > 4:
            raise ValueError("tensors are at most 4-D")
        ints += [len(shape)] + list(shape) + [0] * (4 - len(shape)) + [alias]
    for rec in records:
        ints += rec
    for v in ints:
        if not -2**31 <= v < 2**31:
            raise ValueError(f"{v} does not fit an INT32 field")
    return Program(ints, layout, n_floats, tensor_ids, shapes, ops)


def decode(ints: list[int]) -> dict:
    """The header, tensors and op records of an encoded program; raises ValueError if it is malformed."""
    if len(ints) < HEADER_WORDS or ints[0] != MAGIC or ints[1] != VERSION:
        raise ValueError("not a whole-model program")
    header = dict(zip(("magic", "version", "n_tensors", "n_ops", "n_floats", "input", "output", "tensor_words", "op_words"), ints[:9]))
    if header["tensor_words"] != TENSOR_WORDS or header["op_words"] != OP_WORDS:
        raise ValueError("record width mismatch")
    expected = HEADER_WORDS + header["n_tensors"] * TENSOR_WORDS + header["n_ops"] * OP_WORDS
    if len(ints) != expected:
        raise ValueError(f"program has {len(ints)} words, header implies {expected}")
    pos = HEADER_WORDS
    tensors = []
    for _ in range(header["n_tensors"]):
        rec = ints[pos:pos + TENSOR_WORDS]
        pos += TENSOR_WORDS
        tensors.append({"shape": rec[1:1 + rec[0]], "alias": rec[5]})
    ops = []
    for _ in range(header["n_ops"]):
        rec = ints[pos:pos + OP_WORDS]
        pos += OP_WORDS
        kind = KIND_NAMES.get(rec[0])
        if kind is None:
            raise ValueError(f"unknown op kind {rec[0]}")
        ops.append({"kind": kind, **dict(zip(LAYOUT[kind], rec[1:1 + len(LAYOUT[kind])]))})
    return {"header": header, "tensors": tensors, "ops": ops}


def pack_weights(program: Program, arrays):
    """The FP32 blob: every vector of ``program.layout`` at its offset, read from ``arrays`` (weights.npz keys)."""
    import numpy as np

    blob = np.zeros(program.n_floats, dtype=np.float32)
    for entry in program.layout:
        value = np.ascontiguousarray(arrays[entry["key"]], dtype=np.float32)
        if list(value.shape) != entry["shape"]:
            raise ValueError(f"{entry['key']} has shape {list(value.shape)}, the signature says {entry['shape']}")
        blob[entry["offset"]:entry["offset"] + entry["count"]] = value.reshape(-1)
    return blob

"""The op schedule of kernel.py's forward, as data, with torch-free shape inference.

``build(signature)`` walks the same sequence as ``Model.forward`` in the
generated kernel.py (stem, two MBConv stages, two stages of LiteMLA +
MBConv blocks) and records one dict per operation, naming the parameter and
buffer indices each consumes in the order the generated code consumes them.
Tensors are named strings; ``out`` is the op's output name (a list for the
multi-output attention prep). The exporter turns this into a TensorRT
network; ``infer_shapes`` checks an input shape against every plugin's and
layer's requirements without a GPU and raises ``Unsupported`` with the
reason, which is what "reject unsupported inputs" means at build time.
"""

NONE = {"mode": 0}
DW_CONFIGS = {(3, 1), (3, 2), (5, 1)}  # the dw_kernel<K, S> instantiations kernel.py compiles


class Unsupported(ValueError):
    """The input or a weight cannot be handled by the plugins or the native layers as scheduled."""


def build(signature: dict) -> list[dict]:
    params = [p["shape"] for p in signature["parameters"]]
    ops = []
    counter = {"p": 0, "bn": 0, "t": 0}

    def P():
        i = counter["p"]
        counter["p"] += 1
        return i

    def bn():
        gamma, beta = P(), P()
        i = counter["bn"]
        counter["bn"] += 1
        return {"mode": 2, "gamma": gamma, "beta": beta, "bn": i}

    def bias():
        return {"mode": 1, "beta": P()}

    def name(prefix):
        counter["t"] += 1
        return f"{prefix}_{counter['t']}"

    def conv(x, w, stride=1, pad=0, groups=1):
        out = name("conv")
        ops.append({"kind": "conv", "x": x, "w": w, "kernel": params[w][2], "stride": stride, "pad": pad, "groups": groups, "out": out})
        return out

    def dw(x, w, in_aff, pre, out_aff, k, s, act, emit=0):
        out = name("dw")
        op = {"kind": "dw", "x": x, "w": w, "in_aff": in_aff, "pre": pre, "out_aff": out_aff, "k": k, "s": s, "act": act, "emit": emit, "out": out}
        if emit:
            op["out_pre"] = name("dwpre")
        ops.append(op)
        return (out, op["out_pre"]) if emit else out

    def epilogue(x, aff, act, res=None):
        out = name("ep")
        ops.append({"kind": "epilogue", "x": x, "aff": aff, "act": act, "res": res, "out": out})
        return out

    def mb(t_in, stride, res, conv_bias):
        w = P(); a_inv = bias() if conv_bias else bn()
        t = conv(t_in, w)
        w = P(); a_dw = bias() if conv_bias else bn()
        t = dw(t, w, a_inv, 1, a_dw, 3, stride, 1)
        w = P(); a = bn()
        return epilogue(conv(t, w), a, 0, t_in if res else None)

    def ctx(t_in):
        wq = P(); wd = P(); wg = P(); wp = P(); a = bn()
        qkv = conv(t_in, wq)
        agg = dw(qkv, wd, NONE, 0, NONE, 5, 1, 0)
        agg = conv(agg, wg, 1, 0, params[wg][0] // params[wg][1])
        q, k, vpad = name("q"), name("k"), name("vpad")
        ops.append({"kind": "attn_prep", "qkv": qkv, "agg": agg, "out": [q, k, vpad]})
        kv = name("kv")
        ops.append({"kind": "bmm", "a": k, "b": vpad, "transpose_a": False, "out": kv})
        r = name("r")
        ops.append({"kind": "bmm", "a": q, "b": kv, "transpose_a": True, "out": r})
        att = name("att")
        ops.append({"kind": "attn_final", "r": r, "ref": qkv, "out": att})
        return epilogue(conv(att, wp), a, 0, t_in)

    # stem
    w = P(); a_stem = bn()
    y_raw = conv("x", w, 2, 1)
    w = P(); a = bn()
    t, y = dw(y_raw, w, a_stem, 1, a, 3, 1, 1, emit=1)
    w = P(); a = bn()
    y = epilogue(conv(t, w), a, 0, y)
    # stage 0
    y = mb(y, 2, False, False)
    y = mb(y, 1, True, False)
    # stage 1
    y = mb(y, 2, False, False)
    y = mb(y, 1, True, False)
    y = mb(y, 1, True, False)
    # stage 2
    y = mb(y, 2, False, True)
    for _ in range(3):
        y = ctx(y)
        y = mb(y, 1, True, True)
    # stage 3
    y = mb(y, 2, False, True)
    for _ in range(4):
        y = ctx(y)
        y = mb(y, 1, True, True)
    return ops


def affines(op: dict) -> list[dict]:
    """The affine specs an op applies (Gaia's modes: 2 batch norm, 1 bias, 0 none)."""
    if op["kind"] == "dw":
        return [op["in_aff"], op["out_aff"]]
    if op["kind"] == "epilogue":
        return [op["aff"]]
    return []


def affine_parameters(aff: dict) -> list[int]:
    if aff["mode"] == 2:
        return [aff["gamma"], aff["beta"]]
    if aff["mode"] == 1:
        return [aff["beta"]]
    return []


def parameter_indices(op: dict) -> list[int]:
    """Every parameter index the op uses: its weight, then its affine parameters."""
    out = [op["w"]] if "w" in op else []
    for aff in affines(op):
        out += affine_parameters(aff)
    return out


def bn_buffers(aff: dict) -> tuple[int, int]:
    """Buffer indices of (running_mean, running_var) for a batch-norm affine, as kernel.py's bufs[0::3], bufs[1::3]."""
    return 3 * aff["bn"], 3 * aff["bn"] + 1


def dw_output_hw(h: int, w: int, k: int, s: int) -> tuple[int, int]:
    p = k // 2
    return (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1


def infer_shapes(ops: list[dict], signature: dict, input_shape: list[int]) -> dict[str, list[int]]:
    """Shapes of every tensor for ``input_shape``, checking each op's requirements; raises Unsupported."""
    params = [p["shape"] for p in signature["parameters"]]
    buffers = [b["shape"] for b in signature["buffers"]]
    shapes = {"x": list(input_shape)}
    if len(input_shape) != 4:
        raise Unsupported(f"input must be 4-D (N, C, H, W), got {input_shape}")
    if any(d <= 0 for d in input_shape):
        raise Unsupported(f"input has an empty dimension: {input_shape}")

    def check_affine(aff, c):
        for idx in affine_parameters(aff):
            if params[idx] != [c]:
                raise Unsupported(f"affine parameter {idx} has shape {params[idx]}, expected [{c}]")
        if aff["mode"] == 2:
            for idx in bn_buffers(aff):
                if buffers[idx] != [c]:
                    raise Unsupported(f"batch-norm buffer {idx} has shape {buffers[idx]}, expected [{c}]")

    for op in ops:
        kind = op["kind"]
        if kind == "conv":
            b, c, h, w = shapes[op["x"]]
            cout, cin_g, k, k2 = params[op["w"]]
            if k != k2 or k != op["kernel"]:
                raise Unsupported(f"conv weight {op['w']} is not square {op['kernel']}x{op['kernel']}")
            if cin_g * op["groups"] != c:
                raise Unsupported(f"conv {op['out']}: input has {c} channels, weight expects {cin_g * op['groups']} channels")
            ho = (h + 2 * op["pad"] - k) // op["stride"] + 1
            wo = (w + 2 * op["pad"] - k) // op["stride"] + 1
            if ho <= 0 or wo <= 0:
                raise Unsupported(f"conv {op['out']}: input {h}x{w} is smaller than the kernel")
            shapes[op["out"]] = [b, cout, ho, wo]
        elif kind == "dw":
            b, c, h, w = shapes[op["x"]]
            k, s = op["k"], op["s"]
            if (k, s) not in DW_CONFIGS:
                raise Unsupported(f"dw config (k={k}, s={s}) is not compiled")
            if op["emit"] and s != 1:
                raise Unsupported("emit requires stride 1")
            if params[op["w"]] != [c, 1, k, k]:
                raise Unsupported(f"dw weight {op['w']} has shape {params[op['w']]}, expected [{c}, 1, {k}, {k}]")
            check_affine(op["in_aff"], c)
            check_affine(op["out_aff"], c)
            ho, wo = dw_output_hw(h, w, k, s)
            if ho <= 0 or wo <= 0:
                raise Unsupported(f"dw {op['out']}: input {h}x{w} is smaller than the kernel")
            shapes[op["out"]] = [b, c, ho, wo]
            if op["emit"]:
                shapes[op["out_pre"]] = [b, c, h, w]
        elif kind == "epilogue":
            b, c, h, w = shapes[op["x"]]
            if (h * w) % 4 != 0:
                raise Unsupported(f"epilogue {op['out']}: H*W = {h}*{w} must be a multiple of 4")
            if (b * c * h * w) // 4 >= 2**31:
                raise Unsupported(f"epilogue {op['out']}: tensor too large")
            check_affine(op["aff"], c)
            if op["res"] is not None and shapes[op["res"]] != [b, c, h, w]:
                raise Unsupported(f"epilogue {op['out']}: residual {shapes[op['res']]} does not match {[b, c, h, w]}")
            shapes[op["out"]] = [b, c, h, w]
        elif kind == "attn_prep":
            a, bt = shapes[op["qkv"]], shapes[op["agg"]]
            if a != bt:
                raise Unsupported(f"attention prep: qkv {a} and agg {bt} differ")
            b, c, h, w = a
            if c % 48 != 0:
                raise Unsupported(f"attention prep: {c} channels is not a multiple of 48")
            bh, n = b * 2 * (c // 48), h * w
            q, k, vpad = op["out"]
            shapes[q] = shapes[k] = [bh, 16, n]
            shapes[vpad] = [bh, n, 17]
        elif kind == "bmm":
            a, bsh = shapes[op["a"]], shapes[op["b"]]
            if op["transpose_a"]:
                a = [a[0], a[2], a[1]]
            if a[0] != bsh[0] or a[2] != bsh[1]:
                raise Unsupported(f"bmm {op['out']}: {a} x {bsh}")
            shapes[op["out"]] = [a[0], a[1], bsh[2]]
        elif kind == "attn_final":
            bh, n, e = shapes[op["r"]]
            b, _, h, w = shapes[op["ref"]]
            if e != 17 or n != h * w or bh % b != 0:
                raise Unsupported(f"attention final: r {shapes[op['r']]} does not match {b}x{h}x{w}")
            shapes[op["out"]] = [b, (bh // b) * 16, h, w]
        else:
            raise Unsupported(f"unknown op kind {kind}")
    return shapes

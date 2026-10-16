"""Build the hybrid engine: Gaia IPluginV3 plugins for the fused kernels, native TensorRT for the rest.

Runs inside the dev container with the GPU. Walks the op schedule
(schedule.build) and adds, per op:
  conv       IConvolutionLayer with the PT2 weight (bias never: kernel.py's
             convs are bias-free; biases are applied by the plugins)
  dw         GaiaDepthwiseHswish plugin; weight and affine vectors enter as
             constant tensors
  epilogue   GaiaEpilogue plugin (out of place), residual as a plugin input
  attn_prep  GaiaAttnPrep plugin (three outputs)
  bmm        IMatrixMultiplyLayer, kTRANSPOSE on the first operand for q^T
  attn_final GaiaAttnFinal plugin
The weights come from weights.npz (extract_weights.py); the input shape is the
measured one from model_signature.json; schedule.infer_shapes validates it
first. Two engines are written: hybrid_default.engine (TensorRT default
flags, TF32 allowed for the native layers) and hybrid_notf32.engine.

One engine per process: TensorRT keeps the first logger of a process and a
second build in the same process crashed, so the setting is a parameter.

Usage: build_hybrid_engine.py <weights.npz> <signature.json> <plugin .so> <out dir> <default|notf32>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

import plugin_spec
import schedule
import trt_build


def main(weights_path: str, signature_path: str, plugin_lib: str, out_dir: str, setting: str) -> None:
    out = Path(out_dir)
    signature = json.loads(Path(signature_path).read_text())
    weights = np.load(weights_path)
    ops = schedule.build(signature)
    input_shape = signature["input"]["shape"]
    shapes = schedule.infer_shapes(ops, signature, input_shape)  # raises Unsupported with the reason
    trt_build.load_plugins(plugin_lib)
    registry = trt.get_plugin_registry()

    tf32 = {"default": True, "notf32": False}[setting]
    if True:
        log, builder, network = trt_build.new_builder()
        keep = []  # numpy arrays TensorRT references until the build is done

        def const(array, name):
            array = np.ascontiguousarray(array, dtype=np.float32)
            keep.append(array)
            layer = network.add_constant(tuple(array.shape), trt.Weights(array))
            layer.name = name
            return layer.get_output(0)

        def affine_inputs(aff, prefix):
            if aff["mode"] == 2:
                mean, var = schedule.bn_buffers(aff)
                return [const(weights[f"p{aff['gamma']}"], f"{prefix}.gamma"), const(weights[f"p{aff['beta']}"], f"{prefix}.beta"),
                        const(weights[f"mean{aff['bn']}"], f"{prefix}.mean"), const(weights[f"invstd{aff['bn']}"], f"{prefix}.invstd")]
            if aff["mode"] == 1:
                return [const(weights[f"p{aff['beta']}"], f"{prefix}.bias")]
            return []

        tensors = {"x": network.add_input("x", trt.float32, tuple(input_shape))}
        plugin_layers = 0
        for op in ops:
            kind = op["kind"]
            if kind == "conv":
                w = np.ascontiguousarray(weights[f"p{op['w']}"], dtype=np.float32)
                keep.append(w)
                layer = network.add_convolution_nd(tensors[op["x"]], w.shape[0], (w.shape[2], w.shape[3]), trt.Weights(w))
                layer.stride_nd = (op["stride"], op["stride"])
                layer.padding_nd = (op["pad"], op["pad"])
                layer.num_groups = op["groups"]
                layer.name = op["out"]
                tensors[op["out"]] = layer.get_output(0)
            elif kind == "dw":
                plugin = plugin_spec.make_plugin(trt, registry, plugin_spec.DEPTHWISE, {
                    "k": op["k"], "s": op["s"], "pre": op["pre"], "act": op["act"], "emit": op["emit"],
                    "in_mode": op["in_aff"]["mode"], "out_mode": op["out_aff"]["mode"]})
                inputs = [tensors[op["x"]], const(weights[f"p{op['w']}"], f"{op['out']}.w")]
                inputs += affine_inputs(op["in_aff"], f"{op['out']}.in") + affine_inputs(op["out_aff"], f"{op['out']}.out")
                layer = network.add_plugin_v3(inputs, [], plugin)
                layer.name = op["out"]
                tensors[op["out"]] = layer.get_output(0)
                if op["emit"]:
                    tensors[op["out_pre"]] = layer.get_output(1)
                plugin_layers += 1
            elif kind == "epilogue":
                plugin = plugin_spec.make_plugin(trt, registry, plugin_spec.EPILOGUE, {
                    "mode": op["aff"]["mode"], "act": op["act"], "has_res": int(op["res"] is not None)})
                inputs = [tensors[op["x"]]] + affine_inputs(op["aff"], op["out"])
                if op["res"] is not None:
                    inputs.append(tensors[op["res"]])
                layer = network.add_plugin_v3(inputs, [], plugin)
                layer.name = op["out"]
                tensors[op["out"]] = layer.get_output(0)
                plugin_layers += 1
            elif kind == "attn_prep":
                plugin = plugin_spec.make_plugin(trt, registry, plugin_spec.ATTN_PREP, {})
                layer = network.add_plugin_v3([tensors[op["qkv"]], tensors[op["agg"]]], [], plugin)
                layer.name = op["out"][0]
                for i, name in enumerate(op["out"]):
                    tensors[name] = layer.get_output(i)
                plugin_layers += 1
            elif kind == "bmm":
                op0 = trt.MatrixOperation.TRANSPOSE if op["transpose_a"] else trt.MatrixOperation.NONE
                layer = network.add_matrix_multiply(tensors[op["a"]], op0, tensors[op["b"]], trt.MatrixOperation.NONE)
                layer.name = op["out"]
                tensors[op["out"]] = layer.get_output(0)
            elif kind == "attn_final":
                b, _, h, w = shapes[op["ref"]]
                plugin = plugin_spec.make_plugin(trt, registry, plugin_spec.ATTN_FINAL, {"batch": b, "height": h, "width": w})
                layer = network.add_plugin_v3([tensors[op["r"]]], [], plugin)
                layer.name = op["out"]
                tensors[op["out"]] = layer.get_output(0)
                plugin_layers += 1
            else:
                raise RuntimeError(kind)
            # Every produced tensor must have the shape the torch-free inference predicted.
            for name in ([op["out"]] if isinstance(op["out"], str) else op["out"]) + ([op["out_pre"]] if op.get("out_pre") else []):
                got = list(tensors[name].shape)
                if got != shapes[name]:
                    raise RuntimeError(f"{name}: TensorRT shape {got} differs from the schedule's {shapes[name]}")
        output = tensors[ops[-1]["out"]]
        output.name = "y"
        network.mark_output(output)

        t0 = time.time()
        data, info = trt_build.build_serialized(log, builder, network, tf32)
        info.update({
            "setting": setting, "build_seconds": time.time() - t0, "plugin_layers": plugin_layers,
            "plugins": {"library": plugin_lib, "namespace": plugin_spec.NAMESPACE, "version": plugin_spec.VERSION},
            "input_shape": input_shape, "output_shape": shapes[ops[-1]["out"]],
            "weights_npz": weights_path, "io": trt_build.engine_io(data),
        })
        trt_build.write_engine(out / f"hybrid_{setting}.engine", data, info)
        print(json.dumps({k: v for k, v in info.items() if k != "io"}, indent=2), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])

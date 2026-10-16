"""Build the whole-model engine: the network is one input, one GaiaEffVitWholeModel plugin, one output.

Runs inside the dev container with the GPU (the builder runs the plugin to
time it, which creates its cuDNN plans). The plugin's fields are the program
(whole_program.build over the hybrid experiment's schedule), the packed
weights from weights.npz (extract_weights.py), and the two TF32 switches of
the setting, which mirror what torch's defaults did for kernel.py:
  default  cudnn_tf32=1 (torch.backends.cudnn.allow_tf32 default True),
           cublas_tf32=0 (torch.backends.cuda.matmul.allow_tf32 default False)
  notf32   both 0
The TensorRT builder flags are the hybrid experiment's for the same setting;
with no native layer they do not affect this engine. One engine per process.

Usage: build_whole_engine.py <weights.npz> <signature.json> <plugin .so> <out dir> <default|notf32>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exporter"))  # the hybrid experiment's modules

import trt_build
import whole_plugin_spec
import whole_program

SETTINGS = {"default": {"cudnn_tf32": 1, "cublas_tf32": 0, "tf32": True}, "notf32": {"cudnn_tf32": 0, "cublas_tf32": 0, "tf32": False}}


def main(weights_path: str, signature_path: str, plugin_lib: str, out_dir: str, setting: str) -> None:
    out = Path(out_dir)
    signature = json.loads(Path(signature_path).read_text())
    weights = np.load(weights_path)
    input_shape = signature["input"]["shape"]
    program = whole_program.build(signature, input_shape)  # raises Unsupported with the reason
    blob = whole_program.pack_weights(program, weights)
    trt_build.load_plugins(plugin_lib)
    registry = trt.get_plugin_registry()
    flags = SETTINGS[setting]

    log, builder, network = trt_build.new_builder()
    x = network.add_input("x", trt.float32, tuple(input_shape))
    plugin = whole_plugin_spec.make_plugin(trt, registry, program.ints, blob, flags["cudnn_tf32"], flags["cublas_tf32"])
    layer = network.add_plugin_v3([x], [], plugin)
    layer.name = "gaia_whole_model"
    y = layer.get_output(0)
    y.name = "y"
    if list(y.shape) != signature["output"]["shape"]:
        raise RuntimeError(f"the plugin reports output shape {list(y.shape)}, the signature says {signature['output']['shape']}")
    network.mark_output(y)

    t0 = time.time()
    data, info = trt_build.build_serialized(log, builder, network, flags["tf32"])
    info.update({
        "setting": setting, "build_seconds": time.time() - t0, "plugin_layers": 1,
        "plugin": {"library": plugin_lib, "namespace": whole_plugin_spec.NAMESPACE, "name": whole_plugin_spec.NAME, "version": whole_plugin_spec.VERSION,
                   "cudnn_tf32": flags["cudnn_tf32"], "cublas_tf32": flags["cublas_tf32"], "program_words": len(program.ints), "weight_floats": program.n_floats},
        "input_shape": input_shape, "output_shape": list(y.shape), "weights_npz": weights_path, "io": trt_build.engine_io(data),
    })
    trt_build.write_engine(out / f"whole_{setting}.engine", data, info)
    print(json.dumps({k: v for k, v in info.items() if k != "io"}, indent=2), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])

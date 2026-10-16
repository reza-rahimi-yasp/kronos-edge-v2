"""Per-layer timing of an engine with TensorRT's IProfiler, to attribute the hybrid engine's time (diagnostic only).

Runs inside the dev container with the GPU. Deserializes the engine with the
plugin library loaded, warms up, then profiles ``runs`` executions on the
default stream and reports the mean per-layer time, grouped by layer type as
the engine inspector names it (PluginV3 layers are the Gaia kernels; Reformat
layers are the layout changes TensorRT inserts around them). Nothing is
optimized here; this explains the end-to-end numbers.

Usage: profile_layers.py <engine> <plugin .so> <input.bin> <shape a,b,c,d> <out json> [runs]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

import trt_build


def main(engine_path: str, plugin_lib: str, input_path: str, shape: str, out_path: str, runs: int = 50) -> None:
    trt_build.load_plugins(plugin_lib)
    log, _, _ = trt_build.new_builder()
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    # Engines built with the default profiling verbosity list layer names only (strings); classify by name:
    # the schedule names plugin layers dw_/ep_/q_/att_, and TensorRT names its inserted layout changes "Reformat".
    def classify(name: str) -> str:
        if "Reformat" in name or "reformat" in name:
            return "Reformat"
        if "PluginV3" in name or name.split("[")[0].split("_")[0] in ("dw", "ep", "q", "att"):
            return "PluginV3"
        return "native"

    layer_type = {}
    for layer in layers:
        name = layer if isinstance(layer, str) else layer["Name"]
        layer_type[name] = layer["LayerType"] if isinstance(layer, dict) and "LayerType" in layer else classify(name)
    context = engine.create_execution_context()
    dims = [int(d) for d in shape.split(",")]
    x = torch.from_numpy(np.frombuffer(Path(input_path).read_bytes(), dtype=np.float32).reshape(dims).copy()).cuda()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    buffers = {}
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buffers[name] = x
        else:
            buffers[name] = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    class Profiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.times = defaultdict(float)
            self.calls = defaultdict(int)

        def report_layer_time(self, layer_name, ms):
            self.times[layer_name] += ms
            self.calls[layer_name] += 1

    for _ in range(20):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    profiler = Profiler()
    context.profiler = profiler
    for _ in range(runs):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    per_layer = [{"name": n, "type": layer_type.get(n, classify(n)), "mean_ms": t / profiler.calls[n]} for n, t in profiler.times.items()]
    per_layer.sort(key=lambda e: -e["mean_ms"])
    by_type = defaultdict(float)
    count_by_type = defaultdict(int)
    for e in per_layer:
        by_type[e["type"]] += e["mean_ms"]
        count_by_type[e["type"]] += 1
    total = sum(by_type.values())
    report = {
        "engine": engine_path, "runs": runs, "num_layers_in_engine": len(layers), "profiled_layers": len(per_layer),
        "sum_of_layer_means_ms": total,
        "by_type_ms": {t: {"ms": v, "layers": count_by_type[t], "share": v / total} for t, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
        "top_layers": per_layer[:25],
        "note": "IProfiler layer times serialize layer execution and add overhead; use for attribution, not as end-to-end latency.",
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "top_layers"}, indent=2))
    for e in per_layer[:12]:
        print(f"{e['mean_ms']:8.3f} ms  {e['type']:14s} {e['name'][:100]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 50)

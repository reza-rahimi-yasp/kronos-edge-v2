"""Build the matched pure-TensorRT FP32 baseline from the PT2 reference.

Runs inside the dev container with the GPU. The decomposed torch.export
program of model.pt2 is exported to ONNX with the dynamo exporter, simplified
with onnxsim, and built into a strongly typed FP32 engine with the product's
own ``kronos_edge.tensorrt_check.build`` (default flags, TF32 allowed) and
with the same network and TF32 cleared (pure_notf32.engine). Same weights and
input shape as the hybrid engine by construction: both come from the same
PT2 file. The baseline is labelled as built from the decomposed graph, not
from the original eager model.

Usage: build_pure_engine.py <model.pt2> <out dir>
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import onnx
import onnxsim
import torch

import trt_build
from kronos_edge.tensorrt_check import build as product_build


def main(pt2_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    ep = torch.export.load(pt2_path)
    x = ep.example_inputs[0][0]
    t0 = time.time()
    with torch.no_grad():
        program = torch.onnx.export(ep, (x,), None, dynamo=True, verbose=False, report=False, verify=False)
    onnx_path = out / "pure_model.onnx"
    program.save(onnx_path)
    export_s = time.time() - t0
    t0 = time.time()
    simplified, ok = onnxsim.simplify(program.model_proto)
    if not ok:
        raise SystemExit("onnxsim's check failed")
    simplified_path = out / "pure_model_simplified.onnx"
    onnx.save(simplified, simplified_path)
    onnx.checker.check_model(str(simplified_path))
    simplify_s = time.time() - t0

    results = {}
    t0 = time.time()
    data = product_build(simplified_path, (x,))  # the product's node, unmodified: strongly typed, default flags
    results["default"] = {
        "builder": "kronos_edge.tensorrt_check.build (product, unmodified)", "tf32": True, "build_seconds": time.time() - t0,
        "engine_bytes": len(data), "engine_sha256": hashlib.sha256(data).hexdigest(), "io": trt_build.engine_io(data),
    }
    (out / "pure_default.engine").write_bytes(data)
    (out / "pure_default.build.json").write_text(json.dumps(results["default"], indent=2))

    import tensorrt as trt

    log, builder, network = trt_build.new_builder()
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(simplified_path)):
        raise SystemExit("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    t0 = time.time()
    data, info = trt_build.build_serialized(log, builder, network, tf32=False)
    info.update({"builder": "trt_build.build_serialized (same parse as the product; TF32 flag cleared)", "build_seconds": time.time() - t0,
                 "io": trt_build.engine_io(data)})
    results["notf32"] = info
    trt_build.write_engine(out / "pure_notf32.engine", data, info)

    summary = {
        "source": "model.pt2 (decomposed torch.export graph, the kernelgen reference), NOT the original eager model.py",
        "pt2_sha256": hashlib.sha256(Path(pt2_path).read_bytes()).hexdigest(),
        "onnx_export_seconds": export_s, "onnxsim_seconds": simplify_s,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "simplified_sha256": hashlib.sha256(simplified_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxsim": onnxsim.__version__,
        "engines": results,
    }
    (out / "pure_build.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({**summary, "engines": {k: {kk: vv for kk, vv in v.items() if kk != "io"} for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

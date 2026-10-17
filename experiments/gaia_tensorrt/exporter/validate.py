"""Compare engine outputs with the unmodified Gaia kernel's output on the identical input.

Runs inside the dev container with the GPU, or on any output file. For every
engine named on the command line, the engine is deserialized through the
TensorRT Python API with the plugin library loaded, run on input.bin with
the product's unmodified EngineRunner, run again ``repeats`` times to check
determinism, and compared with gaia_output_<setting>.bin:
shape, finiteness, cosine (threshold from --threshold, the product's default
0.99), max absolute error, max error relative to the reference's largest
magnitude, mean absolute error, a strict allclose(rtol=1e-4, atol=1e-5)
diagnostic, and bitwise equality. Raw output files (--output-file) from the
C++ runner are compared the same way without running anything.

Usage: validate.py <out dir> <plugin .so> <threshold> <setting> engine:<path> ... output:<path> ...
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import trt_build
from kronos_edge.benchmark_check import EngineRunner, cosine


def compare(ref: torch.Tensor, other: torch.Tensor, threshold: float) -> dict:
    result = {"shape": list(other.shape), "shape_equal": list(ref.shape) == list(other.shape), "finite": bool(torch.isfinite(other).all())}
    if not result["shape_equal"]:
        result["passed"] = False
        return result
    diff = (ref.double() - other.double()).abs()
    result.update({
        "cosine": cosine(ref, other),
        "threshold": threshold,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_to_ref_absmax": float(diff.max() / ref.abs().max()),
        "allclose_rtol1e-4_atol1e-5": bool(torch.allclose(ref, other, rtol=1e-4, atol=1e-5)),
        "bitwise_equal": bool(torch.equal(ref, other)),
    })
    result["passed"] = bool(result["finite"] and np.isfinite(result["cosine"]) and result["cosine"] >= threshold)
    return result


def main(out_dir: str, plugin_lib: str, threshold: float, setting: str, subjects: list[str]) -> None:
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    shape = meta["input"]["shape"]
    x_bytes = (out / "input.bin").read_bytes()
    assert hashlib.sha256(x_bytes).hexdigest() == meta["input"]["sha256"], "input.bin changed"
    ref_shape = json.loads((out / f"gaia_reference_{setting}.json").read_text())["output"]["shape"]
    ref = torch.from_numpy(np.frombuffer((out / f"gaia_output_{setting}.bin").read_bytes(), dtype=np.float32).reshape(ref_shape).copy()).cuda()
    x = torch.from_numpy(np.frombuffer(x_bytes, dtype=np.float32).reshape(shape).copy()).cuda()
    report = {"setting": setting, "reference": f"gaia_output_{setting}.bin (unmodified Gaia kernel.py)", "threshold": threshold, "subjects": {}}
    trt_build.load_plugins(plugin_lib)
    for subject in subjects:
        kind, path = subject.split(":", 1)
        if kind == "engine":
            runner = EngineRunner({"engine": Path(path).read_bytes(), "user_inputs": (x.clone(),), "device": "cuda"})
            runner.run()
            outputs = runner.outputs
            assert len(outputs) == 1
            first = outputs[0].clone()
            identical = 0
            repeats = 5
            for _ in range(repeats):
                runner.run()
                identical += int(torch.equal(runner.outputs[0], first))
            entry = {"source": "TensorRT Python runtime, kronos_edge.benchmark_check.EngineRunner (default stream)",
                     "determinism": {"repeats": repeats, "identical": identical, "deterministic": identical == repeats},
                     **compare(ref, first, threshold)}
            del runner
        elif kind == "output":
            data = Path(path).read_bytes()
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size != int(np.prod(ref_shape)):
                entry = {"source": "raw output file", "shape_equal": False, "passed": False, "bytes": len(data)}
            else:
                entry = {"source": "raw output file", "sha256": hashlib.sha256(data).hexdigest(),
                         **compare(ref, torch.from_numpy(arr.reshape(ref_shape).copy()).cuda(), threshold)}
        else:
            raise SystemExit(f"unknown subject kind {kind}")
        report["subjects"][path] = entry
        print(path, json.dumps(entry), flush=True)
    assert torch.equal(x.cpu(), torch.from_numpy(np.frombuffer(x_bytes, dtype=np.float32).reshape(shape).copy())), "input mutated"
    report["all_passed"] = all(e["passed"] for e in report["subjects"].values())
    name = "validation_" + setting + "_" + hashlib.sha256(" ".join(subjects).encode()).hexdigest()[:8] + ".json"
    (out / name).write_text(json.dumps(report, indent=2))
    print("report", name, "all_passed", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5:])

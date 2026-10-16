"""Python-side benchmark: unmodified Gaia kernel vs hybrid engine vs pure TensorRT engine, one process per setting.

Runs inside the dev container with the GPU. Timer: the product's unmodified
``kronos_edge.benchmark_check.timed`` (20 warm-ups, 200 CUDA-event samples,
no synchronization between runs, no cache flush) on the default stream, as
the product measures. Three trials, the order of the subjects reversed on
the second. Outputs are checked against the Gaia output before and after
timing (cosine threshold from the command line); a failed check in either
phase writes the report with the problems and exits nonzero, so a run that
published latencies for an invalid output cannot look successful. Settings:
  default  torch defaults, *_default.engine (TF32 allowed in TensorRT)
  notf32   matmul/cudnn allow_tf32 False, *_notf32.engine (TF32 flag cleared)

Usage: benchmark_python.py <model.pt2> <kernel.py> <out dir> <plugin .so> <threshold> <default|notf32>
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

import checks
import trt_build
from kronos_edge.benchmark_check import RUNS, WARMUP, EngineRunner, cosine, flatten_outputs, stats, timed

TRIALS = 3


def main(pt2_path: str, kernel_path: str, out_dir: str, plugin_lib: str, threshold: float, setting: str) -> None:
    out = Path(out_dir)
    if setting == "notf32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    ep = torch.export.load(pt2_path)
    sig = ep.graph_signature
    flat = [ep.state_dict[f].to(device) for f in list(sig.parameters) + list(sig.buffers)]
    x = ep.example_inputs[0][0].to(device)
    meta = json.loads((out / "meta.json").read_text())
    assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() == meta["input"]["sha256"]
    ref_shape = json.loads((out / f"gaia_reference_{setting}.json").read_text())["output"]["shape"]
    ref = torch.from_numpy(np.frombuffer((out / f"gaia_output_{setting}.bin").read_bytes(), dtype=np.float32).reshape(ref_shape).copy()).to(device)

    spec = importlib.util.spec_from_file_location("gaia_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.Model().eval().to(device)
    gaia_args = [t.clone() for t in flat] + [x.clone()]
    trt_build.load_plugins(plugin_lib)
    runners = {
        "hybrid_trt": EngineRunner({"engine": (out / f"hybrid_{setting}.engine").read_bytes(), "user_inputs": (x.clone(),), "device": "cuda"}),
        "pure_trt": EngineRunner({"engine": (out / f"pure_{setting}.engine").read_bytes(), "user_inputs": (x.clone(),), "device": "cuda"}),
    }
    fns = {"gaia_optimized": lambda: candidate(*gaia_args), "hybrid_trt": runners["hybrid_trt"].run, "pure_trt": runners["pure_trt"].run}

    def outputs_of(name):
        if name == "gaia_optimized":
            with torch.no_grad():
                return flatten_outputs(fns[name]())[0]
        fns[name]()
        return runners[name].outputs[0]

    def check(name):
        y = outputs_of(name)
        return {"shape": list(y.shape), "finite": bool(torch.isfinite(y).all()), "cosine": cosine(ref, y),
                "max_abs_diff": float((ref.double() - y.double()).abs().max()),
                "allclose_rtol1e-4_atol1e-5": bool(torch.allclose(ref, y, rtol=1e-4, atol=1e-5)), "bitwise_equal": bool(torch.equal(ref, y))}

    result = {
        "setting": setting,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu": torch.cuda.get_device_name(), "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "tensorrt": trt_build.trt.__version__},
        "protocol": {"timer": "kronos_edge.benchmark_check.timed (unmodified)", "warmup": WARMUP, "runs_per_trial": RUNS, "trials": TRIALS,
                     "cache_flush": False, "stream": "torch default stream", "grad": "no_grad"},
        "subjects": {"gaia_optimized": "unmodified kernel.py Model on the PT2 parameters/buffers/example input",
                     "hybrid_trt": f"hybrid_{setting}.engine via kronos_edge EngineRunner", "pure_trt": f"pure_{setting}.engine via kronos_edge EngineRunner"},
        "sha256": {"kernel.py": hashlib.sha256(Path(kernel_path).read_bytes()).hexdigest(), "model.pt2": hashlib.sha256(Path(pt2_path).read_bytes()).hexdigest(),
                   **{f"{n}_{setting}.engine": hashlib.sha256((out / f"{n}_{setting}.engine").read_bytes()).hexdigest() for n in ("hybrid", "pure")}},
        "checks_before": {}, "trials": [], "checks_after": {},
    }
    report_path = out / f"benchmark_python_{setting}.json"
    with torch.no_grad():
        for name in fns:
            result["checks_before"][name] = check(name)
        if not checks.finish_report(report_path, result, threshold):
            raise SystemExit(f"pre-timing checks failed, nothing timed; see {report_path}: {result['check_problems']}")
        for trial in range(TRIALS):
            order = list(fns) if trial % 2 == 0 else list(fns)[::-1]
            samples = {name: timed(fns[name], True) for name in order}
            result["trials"].append({"order": order, "stats_ms": {n: stats(s) for n, s in samples.items()}, "samples_ms": samples})
            print("trial", trial + 1, {n: stats(s) for n, s in samples.items()}, flush=True)
        for name in fns:
            result["checks_after"][name] = check(name)
    assert torch.equal(x, ep.example_inputs[0][0].to(device)) and torch.equal(gaia_args[-1], x), "input mutated"
    result["aggregate_ms"] = {name: stats([v for t in result["trials"] for v in t["samples_ms"][name]]) for name in fns}
    med = {n: result["aggregate_ms"][n]["median"] for n in fns}
    result["median_ratios"] = {"gaia_over_hybrid": med["gaia_optimized"] / med["hybrid_trt"], "pure_over_hybrid": med["pure_trt"] / med["hybrid_trt"],
                               "gaia_over_pure": med["gaia_optimized"] / med["pure_trt"]}
    passed = checks.finish_report(report_path, result, threshold)
    print(json.dumps({k: result[k] for k in ("environment", "aggregate_ms", "median_ratios", "all_checks_passed", "check_problems")}, indent=2))
    if not passed:
        raise SystemExit(f"post-timing checks failed; the report is kept at {report_path}: {result['check_problems']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6])

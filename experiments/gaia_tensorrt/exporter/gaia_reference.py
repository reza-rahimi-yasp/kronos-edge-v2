"""Run the unmodified Gaia kernel (kernel.py) on the extracted weights and input; record its output.

Runs inside the dev container with the GPU. kernel.py is imported from the
read-only artifact directory exactly as Gaia returned it (load_inline
compiles its extension into TORCH_EXTENSIONS_DIR). Its forward takes the flat
PT2 placeholders: parameters, buffers, then the input, in signature order,
which is what the PT2 reference provides.

Two settings are run, each in its own process (the kernel's cuDNN
convolutions honour torch.backends.cudnn.allow_tf32):
  default  torch defaults (cudnn.allow_tf32 True, matmul.allow_tf32 False)
  notf32   both False
Writes gaia_output_<setting>.bin plus gaia_reference_<setting>.json with the
determinism check (N repeated runs compared byte for byte) and, labelled as
such, the PT2 decomposed graph's output (ep.module()) and its distance from
the Gaia output.

Usage: gaia_reference.py <model.pt2> <kernel.py> <out dir> <default|notf32> [repeats]
"""

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def compare(ref: torch.Tensor, other: torch.Tensor) -> dict:
    diff = (ref.double() - other.double()).abs()
    return {
        "shape_equal": list(ref.shape) == list(other.shape),
        "finite": bool(torch.isfinite(other).all()),
        "cosine": cosine(ref, other),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_to_ref_absmax": float(diff.max() / ref.abs().max()),
        "allclose_rtol1e-4_atol1e-5": bool(torch.allclose(ref, other, rtol=1e-4, atol=1e-5)),
        "bitwise_equal": bool(torch.equal(ref, other)),
    }


def main(pt2_path: str, kernel_path: str, out_dir: str, setting: str, repeats: int = 5) -> None:
    out = Path(out_dir)
    if setting == "notf32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif setting != "default":
        raise SystemExit("setting must be default or notf32")
    ep = torch.export.load(pt2_path)
    sig = ep.graph_signature
    device = torch.device("cuda")
    flat = [ep.state_dict[f].to(device) for f in list(sig.parameters) + list(sig.buffers)]
    x = ep.example_inputs[0][0].to(device)
    expected_input_sha = json.loads((out / "meta.json").read_text())["input"]["sha256"]
    if sha256_bytes(x.cpu().numpy().tobytes()) != expected_input_sha:
        raise SystemExit("the PT2 example input differs from the extracted input.bin")

    t0 = time.time()
    spec = importlib.util.spec_from_file_location("gaia_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # compiles the inline extension
    compile_s = time.time() - t0
    candidate = module.Model().eval().to(device)
    with torch.no_grad():
        outputs = candidate(*[t.clone() for t in flat], x.clone())
        assert isinstance(outputs, tuple) and len(outputs) == 1, type(outputs)
        y = outputs[0].contiguous()
        torch.cuda.synchronize()
        identical = 0
        for _ in range(repeats):
            again = candidate(*[t.clone() for t in flat], x.clone())[0]
            torch.cuda.synchronize()
            identical += int(torch.equal(again, y))
        pt2_module = ep.module()
        # The reference was traced on the CPU, so its graph asserts device=cpu on the input; those
        # assert nodes have no users and are removed to run the same decomposed arithmetic on the GPU.
        removed = 0
        for node in list(pt2_module.graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten._assert_tensor_metadata.default:
                pt2_module.graph.erase_node(node)
                removed += 1
        pt2_module.recompile()
        pt2_module = pt2_module.to(device)
        y_pt2 = pt2_module(x.clone())
        y_pt2 = (y_pt2[0] if isinstance(y_pt2, (tuple, list)) else y_pt2).contiguous()
        torch.cuda.synchronize()
    assert torch.equal(x, ep.example_inputs[0][0].to(device)), "input mutated"
    y_np = y.cpu().numpy()
    (out / f"gaia_output_{setting}.bin").write_bytes(y_np.tobytes())
    (out / f"pt2_output_{setting}.bin").write_bytes(y_pt2.cpu().numpy().tobytes())
    report = {
        "setting": setting,
        "subject": "unmodified Gaia kernel.py (AgentKernel module_source) on the PT2 reference's parameters, buffers and saved example input",
        "kernel_py_sha256": sha256_bytes(Path(kernel_path).read_bytes()),
        "pt2_sha256": sha256_bytes(Path(pt2_path).read_bytes()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "extension_compile_seconds": compile_s,
        "output": {"shape": list(y.shape), "dtype": str(y.dtype), "finite": bool(torch.isfinite(y).all()),
                   "sha256": sha256_bytes(y_np.tobytes()), "absmax": float(y.abs().max())},
        "determinism": {"repeats": repeats, "identical": identical, "deterministic": identical == repeats},
        "pt2_decomposed_graph_vs_gaia": {
            "label": "ep.module() of the decomposed torch.export graph (the kernelgen reference) run on cuda with its "
                     f"{removed} _assert_tensor_metadata nodes (device=cpu asserts from the CPU trace) removed; NOT the original eager model",
            **compare(y, y_pt2),
        },
    }
    (out / f"gaia_reference_{setting}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["determinism"]["deterministic"] or not report["output"]["finite"]:
        raise SystemExit("the Gaia reference is not deterministic or not finite")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 5)

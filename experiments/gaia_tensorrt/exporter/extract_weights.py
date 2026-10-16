"""Extract the weights, the derived batch-norm inverse standard deviations and the saved input from the PT2 reference.

Runs inside the dev container (torch only, no yasp imports). Checks the
reference against exporter/model_signature.json, then writes to the output
directory:
  weights.npz   p<i> for every parameter, mean<i>/invstd<i> for every batch norm
  input.bin     the saved example input as raw little-endian float32 (NCHW)
  input.npy     the same as a NumPy array
  meta.json     hashes, shapes, and where each tensor came from

invstd is computed exactly as kernel.py computes it at run time, on the GPU:
aten._native_batch_norm_legit_no_training on ones/zeros with the concatenated
running variances, eps 1e-5. The CPU value is also recorded for comparison.

Usage: extract_weights.py <model.pt2> <signature.json> <out dir>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def invstd_like_kernel(vars_: list[torch.Tensor]) -> torch.Tensor:
    """kernel.py: V = cat(vars); invstd = batch_norm_legit_no_training(ones, None, None, zeros, V, 0.1, 1e-5)[0].reshape(-1)."""
    V = torch.cat(vars_)
    ones = torch.ones((1, V.numel(), 1, 1), device=V.device, dtype=V.dtype)
    zeros = torch.zeros_like(V)
    return torch.ops.aten._native_batch_norm_legit_no_training.default(ones, None, None, zeros, V, 0.1, 1e-05)[0].reshape(-1)


def main(pt2_path: str, signature_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    signature = json.loads(Path(signature_path).read_text())
    pt2_bytes = Path(pt2_path).read_bytes()
    pt2_sha = sha256_bytes(pt2_bytes)
    if pt2_sha != signature["pt2_sha256"]:
        raise SystemExit(f"model.pt2 sha256 {pt2_sha} differs from the recorded {signature['pt2_sha256']}")
    ep = torch.export.load(pt2_path)
    sig = ep.graph_signature
    params = list(sig.parameters)
    buffers = list(sig.buffers)
    recorded_p = [(p["fqn"], p["shape"]) for p in signature["parameters"]]
    recorded_b = [(b["fqn"], b["shape"]) for b in signature["buffers"]]
    actual_p = [(f, list(ep.state_dict[f].shape)) for f in params]
    actual_b = [(f, list(ep.state_dict[f].shape)) for f in buffers]
    if actual_p != recorded_p or actual_b != recorded_b:
        raise SystemExit("the PT2 signature does not match model_signature.json")
    x = ep.example_inputs[0][0]
    if list(x.shape) != signature["input"]["shape"] or x.dtype != torch.float32 or ep.example_inputs[1]:
        raise SystemExit(f"unexpected example input {tuple(x.shape)} {x.dtype} {ep.example_inputs[1]}")
    for f in params + buffers:
        if ep.state_dict[f].dtype not in (torch.float32, torch.int64):
            raise SystemExit(f"{f} has dtype {ep.state_dict[f].dtype}")

    device = torch.device("cuda")
    bufs = [ep.state_dict[f] for f in buffers]
    vars_gpu = [t.to(device) for t in bufs[1::3]]
    invstd_gpu = invstd_like_kernel(vars_gpu).cpu()
    invstd_cpu = invstd_like_kernel(bufs[1::3])
    means = [m.detach() for m in bufs[0::3]]

    arrays = {}
    meta_tensors = {}
    for i, f in enumerate(params):
        t = ep.state_dict[f].detach().contiguous()
        arrays[f"p{i}"] = t.numpy()
        meta_tensors[f"p{i}"] = {"fqn": f, "shape": list(t.shape), "sha256": tensor_sha(t)}
    off = 0
    for i, m in enumerate(means):
        c = m.numel()
        arrays[f"mean{i}"] = m.contiguous().numpy()
        arrays[f"invstd{i}"] = invstd_gpu[off:off + c].contiguous().numpy()
        meta_tensors[f"mean{i}"] = {"fqn": buffers[3 * i], "shape": [c], "sha256": tensor_sha(m)}
        meta_tensors[f"invstd{i}"] = {"fqn": buffers[3 * i + 1] + " -> 1/sqrt(var+1e-5) via aten._native_batch_norm_legit_no_training on cuda",
                                      "shape": [c], "sha256": tensor_sha(invstd_gpu[off:off + c])}
        off += c
    np.savez(out / "weights.npz", **arrays)
    x_np = x.contiguous().numpy()
    (out / "input.bin").write_bytes(x_np.tobytes())
    np.save(out / "input.npy", x_np)
    meta = {
        "pt2_sha256": pt2_sha,
        "torch": torch.__version__,
        "input": {
            "source": "example_inputs saved in model.pt2 by yasp-toolkit kernelgen export (the get_inputs() tensor; not input.npz)",
            "shape": list(x.shape), "dtype": "float32", "layout": "NCHW raw little-endian",
            "sha256": sha256_bytes(x_np.tobytes()),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
        },
        "invstd": {
            "device": "cuda", "cpu_equals_gpu_bitwise": bool(torch.equal(invstd_gpu, invstd_cpu)),
            "max_abs_diff_cpu_gpu": float((invstd_gpu - invstd_cpu).abs().max()),
            "count": int(invstd_gpu.numel()),
        },
        "tensors": meta_tensors,
        "weights_npz_sha256": sha256_bytes((out / "weights.npz").read_bytes()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

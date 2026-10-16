"""Extract the Gaia-generated CUDA device code verbatim from kernel.py into a header (stdlib only).

The generated kernel.py embeds its CUDA source in a Python string. Two
segments of it are pure device code with no torch dependency: the ``Aff``
struct with the ``hsw``/``bnf`` helpers, and the four ``__global__`` kernels.
This script copies those two segments byte for byte into
``plugins/gaia_device_code.cuh`` and records their hashes in
``plugins/provenance.json``, so a test and the build can both prove the
header is the pristine generated code and not something we edited. The host
wrappers (``dw_forward``, ``epilogue_``, ``attn_prep``, ``attn_final``) are
torch code and are not extracted; the plugins reimplement their launch logic.

Usage: extract_gaia_device_code.py <kernel.py> <plugins dir>
"""

import hashlib
import json
import sys
from pathlib import Path

SEGMENTS = [
    # (name, start anchor (inclusive), end anchor (exclusive)); each anchor must occur exactly once.
    ("affine_helpers", "struct Aff {", "static const float* optp("),
    ("kernels", "template<int K, int S>\n__global__", "std::vector<torch::Tensor> dw_forward("),
]
BEGIN = "// BEGIN verbatim segment "
END = "// END verbatim segment "


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


CUDA_SRC_START, CUDA_SRC_END = 'cuda_src = r"""', 'cpp_src = r"""'


def cuda_source(module_source: str) -> str:
    """The CUDA source string of kernel.py (the ``cuda_src`` literal), where the device code lives."""
    if module_source.count(CUDA_SRC_START) != 1 or module_source.count(CUDA_SRC_END) != 1:
        raise ValueError("kernel.py does not have exactly one cuda_src and one cpp_src literal")
    return module_source[module_source.index(CUDA_SRC_START):module_source.index(CUDA_SRC_END)]


def segments(source: str) -> dict[str, str]:
    source = cuda_source(source)
    out = {}
    for name, start, end in SEGMENTS:
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError(f"anchor for segment {name} is not unique in the source")
        a, b = source.index(start), source.index(end)
        if b <= a:
            raise ValueError(f"segment {name}: end anchor precedes start anchor")
        out[name] = source[a:b].rstrip() + "\n"
    return out


def render(segs: dict[str, str], kernel_sha: str) -> str:
    parts = [
        "// Gaia-generated CUDA device code, extracted verbatim from kernel.py",
        f"// (sha256 {kernel_sha}) by exporter/extract_gaia_device_code.py.",
        "// Do not edit: the segments between the BEGIN/END markers are the generated",
        "// code byte for byte; plugins/provenance.json records their hashes.",
        "#pragma once",
        "#include <cuda_runtime.h>",
        "",
    ]
    for name, text in segs.items():
        parts += [f"{BEGIN}{name}", text.rstrip("\n"), f"{END}{name}", ""]
    return "\n".join(parts)


def parse_header(header: str) -> dict[str, str]:
    """The verbatim segments of a rendered header, keyed by name."""
    out = {}
    for name, _, _ in SEGMENTS:
        start, end = f"{BEGIN}{name}\n", f"\n{END}{name}"
        if header.count(start) != 1 or header.count(end) != 1:
            raise ValueError(f"header lacks a unique {name} segment")
        a = header.index(start) + len(start)
        out[name] = header[a:header.index(end)] + "\n"
    return out


def main(kernel_path: str, plugins_dir: str) -> None:
    source = Path(kernel_path).read_text()
    kernel_sha = sha256(source)
    segs = segments(source)
    header = render(segs, kernel_sha)
    assert parse_header(header) == segs
    plugins = Path(plugins_dir)
    plugins.mkdir(parents=True, exist_ok=True)  # run_all.sh points this at a fresh OUT_DIR/device_code_check
    (plugins / "gaia_device_code.cuh").write_text(header)
    provenance = {
        "source": "kernel.py returned by yasp-toolkit kernelgen compile (module_source.py)",
        "kernel_py_sha256": kernel_sha,
        "segments_sha256": {name: sha256(text) for name, text in segs.items()},
        "header_sha256": sha256(header),
        "note": "Only these segments are pristine Gaia output. Host wrappers, the out-of-place epilogue "
                "(gaia_adapted_epilogue.cuh), the plugins, and the runner were written for this experiment.",
    }
    (plugins / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

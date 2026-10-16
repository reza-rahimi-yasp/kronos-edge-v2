"""The committed device code is the Gaia-generated code, and the one adaptation is mechanical."""


# The exporter modules are plain scripts, not a package: make them importable from this test alone.
# (No conftest.py here: pytest imports conftest modules by that bare name, and a second one would
# shadow the product's tests/conftest.py, which its tests import from directly.)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))

import hashlib
import json
import re
from pathlib import Path

import pytest

import extract_gaia_device_code as extractor

PLUGINS = Path(__file__).resolve().parents[1] / "plugins"


@pytest.fixture(scope="module")
def header():
    return (PLUGINS / "gaia_device_code.cuh").read_text()


@pytest.fixture(scope="module")
def provenance():
    return json.loads((PLUGINS / "provenance.json").read_text())


def test_segments_match_recorded_hashes(header, provenance):
    segments = extractor.parse_header(header)
    assert set(segments) == set(provenance["segments_sha256"])
    for name, text in segments.items():
        assert hashlib.sha256(text.encode()).hexdigest() == provenance["segments_sha256"][name], name
    assert hashlib.sha256(header.encode()).hexdigest() == provenance["header_sha256"]


def test_header_is_device_code_only(header):
    body = "\n".join(line for line in header.splitlines() if not line.lstrip().startswith("//"))
    assert "torch" not in body and "c10::" not in body and "TORCH_CHECK" not in body


@pytest.mark.parametrize("kernel", ["dw_kernel", "epilogue_kernel", "attn_prep_kernel", "attn_final_kernel"])
def test_each_gaia_kernel_is_present_once(header, kernel):
    assert len(re.findall(rf"__global__ void (?:__launch_bounds__\(\d+\) )?{kernel}\(", header)) == 1


def _kernel_body(source: str, name: str) -> str:
    """The text of ``__global__ void <name>(`` up to and including its closing brace."""
    start = source.index(f"__global__ void {name}(")
    brace = source.index("{", start)
    depth, i = 0, brace
    while True:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
        i += 1


def test_out_of_place_epilogue_is_the_generated_kernel_with_renamed_accesses(header):
    adapted = (PLUGINS / "gaia_adapted_epilogue.cuh").read_text()
    pristine = _kernel_body(header, "epilogue_kernel")
    expected = (
        pristine.replace("epilogue_kernel(float4* __restrict__ y, ", "epilogue_kernel_oop(const float4* __restrict__ yin, float4* __restrict__ yout, ")
        .replace("float4 v = y[i];", "float4 v = yin[i];")
        .replace("y[i] = v;", "yout[i] = v;")
    )
    assert expected != pristine  # the three substitutions all applied
    assert _kernel_body(adapted, "epilogue_kernel_oop") == expected


def _fake_kernel_module(header: str) -> str:
    """A kernel.py-shaped module around the committed segments: cuda_src literal, then cpp_src literal."""
    segments = extractor.parse_header(header)
    return (
        'import torch\n\ncuda_src = r"""\n#include <torch/extension.h>\n\n'
        + segments["affine_helpers"]
        + "\nstatic const float* optp(const c10::optional<torch::Tensor>& t){ return nullptr; }\n\n"
        + segments["kernels"]
        + '\nstd::vector<torch::Tensor> dw_forward(torch::Tensor x){ return {x}; }\n"""\n\n'
        + 'cpp_src = r"""\nstd::vector<torch::Tensor> dw_forward(torch::Tensor x);\n"""\n'
    )


def test_extractor_creates_a_fresh_output_directory(header, tmp_path):
    """run_all.sh points the extractor at OUT_DIR/device_code_check, which does not exist on a fresh OUT_DIR."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text(_fake_kernel_module(header))
    fresh = tmp_path / "fresh-out" / "device_code_check"
    assert not fresh.parent.exists()
    extractor.main(str(kernel), str(fresh))
    assert extractor.parse_header((fresh / "gaia_device_code.cuh").read_text()) == extractor.parse_header(header)
    assert json.loads((fresh / "provenance.json").read_text())["segments_sha256"] == json.loads((PLUGINS / "provenance.json").read_text())["segments_sha256"]


HWC_KERNELS = ("dw_kernel_hwc", "epilogue_kernel_hwc", "attn_prep_kernel_hwc", "attn_final_kernel_hwc")


@pytest.fixture(scope="module")
def hwc_header():
    return (PLUGINS / "gaia_hwc_kernels.cuh").read_text()


def test_hwc_kernels_are_labelled_as_written_for_this_experiment(hwc_header):
    head = "\n".join(hwc_header.splitlines()[:12])
    assert "written for this experiment" in head and "not Gaia" in head
    assert "verbatim" not in head.replace("not verbatim", "")


@pytest.mark.parametrize("kernel", HWC_KERNELS)
def test_each_hwc_kernel_is_present_once(hwc_header, kernel):
    assert len(re.findall(rf"__global__ void (?:__launch_bounds__\(\d+\) )?{kernel}\(", hwc_header)) == 1


def test_hwc_kernels_reuse_gaia_arithmetic_helpers_rather_than_redefining_them(hwc_header):
    assert '#include "gaia_device_code.cuh"' in hwc_header
    assert "float hsw(" not in hwc_header and "float bnf(" not in hwc_header and "struct Aff" not in hwc_header
    body = "\n".join(line for line in hwc_header.splitlines() if not line.lstrip().startswith("//"))
    # the same rounding-controlled intrinsics as the generated code; no plain float operators on the data path
    for intrinsic in ("__fmaf_rn", "__fadd_rn", "__fdiv_rn", "fmaxf", "hsw(", "bnf("):
        assert intrinsic in body, intrinsic
    assert "torch" not in body


def test_verbatim_header_is_not_touched_by_the_layout_work(header, provenance):
    assert "hwc" not in header.lower()
    assert provenance["kernel_py_sha256"] == "980698e39f89ac6371c3d766e872f688c4645e1a2150499792dbad373527c032"

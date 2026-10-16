"""The committed device code is the Gaia-generated code, and the one adaptation is mechanical."""

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

"""The whole-model plugin's Python spec, C++ header and build files agree, and stay distinct from the hybrid plugins."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "exporter"))

import re
from pathlib import Path

import plugin_spec
import whole_plugin_spec

WHOLE = Path(__file__).resolve().parents[1]
EXPERIMENT = WHOLE.parent
HEADER = (WHOLE / "plugin" / "gaia_whole_model_plugin.h").read_text()
SOURCE = (WHOLE / "plugin" / "gaia_whole_model_plugin.cu").read_text()


def _cpp_string(name: str) -> str:
    return re.search(rf'constexpr char const\* {name} = "([^"]*)";', HEADER).group(1)


def test_header_constants_match_the_python_spec():
    assert _cpp_string("kPluginNamespace") == whole_plugin_spec.NAMESPACE
    assert _cpp_string("kPluginVersion") == whole_plugin_spec.VERSION
    assert _cpp_string("kPluginName") == whole_plugin_spec.NAME
    fields = re.findall(r'"([a-z_0-9]+)"', re.search(r"kPluginFields\[\] = \{([^}]*)\};", HEADER).group(1))
    assert fields == [name for name, _ in whole_plugin_spec.FIELDS]
    assert f'extern "C" bool {whole_plugin_spec.REGISTER_SYMBOL}();' in HEADER
    assert re.search(rf'extern "C" .*bool {whole_plugin_spec.REGISTER_SYMBOL}\(\)', SOURCE)


def test_distinct_from_the_hybrid_plugins():
    assert whole_plugin_spec.NAMESPACE != plugin_spec.NAMESPACE
    assert whole_plugin_spec.NAME not in plugin_spec.FIELDS
    assert whole_plugin_spec.REGISTER_SYMBOL != "gaia_trt_register_plugins"
    # the hybrid header's symbol is the only other exported one; the whole-model source never defines it
    assert "gaia_trt_register_plugins(" not in SOURCE


def test_program_constants_match_the_decoder():
    import whole_program

    assert int(re.search(r"kProgramMagic = (0x[0-9A-Fa-f]+)", HEADER).group(1), 16) == whole_program.MAGIC
    assert int(re.search(r"kProgramVersion = (\d+)", HEADER).group(1)) == whole_program.VERSION
    assert int(re.search(r"kHeaderWords = (\d+)", HEADER).group(1)) == whole_program.HEADER_WORDS
    assert int(re.search(r"kTensorWords = (\d+)", HEADER).group(1)) == whole_program.TENSOR_WORDS
    assert int(re.search(r"kOpWords = (\d+)", HEADER).group(1)) == whole_program.OP_WORDS
    kinds = dict(re.findall(r"k(Conv|Dw|Epilogue|AttnPrep|Bmm|AttnFinal) = (\d+)", HEADER))
    assert {k.lower(): int(v) for k, v in kinds.items()} == {
        "conv": 1, "dw": 2, "epilogue": 3, "attnprep": 4, "bmm": 5, "attnfinal": 6} == {k.replace("_", ""): v for k, v in whole_program.KINDS.items()}


def test_device_code_is_gaias_and_the_epilogue_runs_in_place():
    assert '#include "gaia_device_code.cuh"' in SOURCE
    assert "gaia_adapted_epilogue" not in SOURCE and "epilogue_kernel_oop" not in SOURCE
    assert "epilogue_kernel<<<" in SOURCE
    for kernel in ("dw_kernel<3, 1>", "dw_kernel<3, 2>", "dw_kernel<5, 1>", "attn_prep_kernel<<<", "attn_final_kernel<<<"):
        assert kernel in SOURCE


def test_no_torch_and_only_official_library_apis():
    body = re.sub(r"//.*", "", SOURCE)  # the comments say what is mirrored from torch; the code never names it
    assert "torch" not in body and "ATen" not in body and "c10::" not in body and "nvrtc" not in body.lower()
    assert "cudnnBackendExecute(" in body and "cublasSgemmStridedBatched(" in body
    # the legacy, deprecated convolution entry points are not used
    assert "cudnnConvolutionForward(" not in body and "cudnnGetConvolutionForwardAlgorithm" not in body


def test_pytorch_library_calls_are_mirrored():
    body = SOURCE
    assert "CUDNN_HEUR_MODE_INSTANT" in body               # torch 2.14.1 Conv_v8 heuristics mode (probe log: mode 0)
    assert "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS" in body and "CUDNN_NUMERICAL_NOTE_TENSOR_CORE" in body
    assert "CUDNN_HEUR_MODE_FALLBACK" in body
    assert "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION" in body                # never an NVRTC engine
    assert re.search(r"engineIndex == 58 \|\| engineIndex == 63", body) and "130601" in body  # ATen's sm120 errata (issue 188288)
    assert re.search(r"kCublasWorkspaceBytes = 32 \* 1024 \* 1024", body)  # torch's cuBLAS workspace on this GPU (probe log)
    assert re.search(r"kTensorAlignment = 32", body)                      # torch's tensor alignment (probe log)
    assert "CUBLAS_TF32_TENSOR_OP_MATH" in body and "CUBLAS_DEFAULT_MATH" in body


def test_runtime_image_pins_torchs_library_builds_and_installs_no_python():
    dockerfile = (WHOLE / "docker" / "runtime" / "Dockerfile").read_text()
    for name, value in re.findall(r"^ARG (\w+)=(\S+)", dockerfile, re.M):
        dockerfile = dockerfile.replace("${" + name + "}", value)
    assert f"libcudnn9-cuda-13={whole_plugin_spec.CUDNN_VERSION}-1" in dockerfile
    assert f"libcublas-13-0={whole_plugin_spec.CUBLAS_VERSION}-1" in dockerfile
    instructions = "\n".join(line for line in dockerfile.splitlines() if not line.startswith("#")).lower()
    assert "python" not in instructions and "pip" not in instructions and "nvrtc" not in instructions
    dev = (EXPERIMENT / "docker" / "dev" / "Dockerfile").read_text()
    runtime = (EXPERIMENT / "docker" / "runtime" / "Dockerfile").read_text()
    version = re.search(r"TRT_DEB_VERSION=([\d.]+-[^ \n]+)", dev).group(1)
    assert f"TRT_DEB_VERSION={version}" in dockerfile and f"TRT_DEB_VERSION={version}" in runtime


def test_makefile_builds_the_library_with_hidden_symbols_and_both_runners():
    makefile = (WHOLE / "Makefile").read_text()
    assert "-fvisibility=hidden" in makefile
    assert "libgaia_trt_whole_model.so" in makefile
    assert "libcudnn.so.9" in makefile and "libcublas.so.13" in makefile  # linked by soname from the dev image's venv
    runner = (EXPERIMENT / "runner" / "runner.cpp").read_text()
    assert "GAIA_TRT_EXTRA_REGISTER" in runner  # the shared runner registers a second library when told to
    assert f"GAIA_TRT_EXTRA_REGISTER={whole_plugin_spec.REGISTER_SYMBOL}" in makefile

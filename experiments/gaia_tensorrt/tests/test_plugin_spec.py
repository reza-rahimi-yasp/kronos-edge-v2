"""The Python plugin spec and the C++ header agree on names, version, namespace and fields."""


# The exporter modules are plain scripts, not a package: make them importable from this test alone.
# (No conftest.py here: pytest imports conftest modules by that bare name, and a second one would
# shadow the product's tests/conftest.py, which its tests import from directly.)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))

import re
from pathlib import Path

import plugin_spec

HEADER = (Path(__file__).resolve().parents[1] / "plugins" / "gaia_plugins.h").read_text()


def _cpp_string(name: str) -> str:
    return re.search(rf'constexpr char const\* {name} = "([^"]*)";', HEADER).group(1)


def _cpp_list(name: str) -> list[str]:
    body = re.search(rf"constexpr char const\* {name}\[\] = \{{([^}}]*)\}};", HEADER).group(1)
    return re.findall(r'"([^"]*)"', body)


def test_namespace_and_version():
    assert plugin_spec.NAMESPACE == _cpp_string("kPluginNamespace")
    assert plugin_spec.VERSION == _cpp_string("kPluginVersion")


def test_names_and_fields():
    cpp = {
        _cpp_string("kDepthwiseName"): _cpp_list("kDepthwiseFields"),
        _cpp_string("kEpilogueName"): _cpp_list("kEpilogueFields"),
        _cpp_string("kAttnPrepName"): _cpp_list("kAttnPrepFields"),
        _cpp_string("kAttnFinalName"): _cpp_list("kAttnFinalFields"),
    }
    assert {name: list(fields) for name, fields in plugin_spec.FIELDS.items()} == cpp


def test_every_plugin_ends_with_the_layout_field_and_the_values_match_the_header():
    assert all(fields[-1] == "layout" for fields in plugin_spec.FIELDS.values())
    assert plugin_spec.LAYOUTS == {"any": 0, "linear": 1, "hwc": 2}
    values = re.search(r"constexpr int32_t kLayoutAny = (\d), kLayoutLinear = (\d), kLayoutHwc = (\d);", HEADER).groups()
    assert tuple(int(v) for v in values) == (plugin_spec.LAYOUTS["any"], plugin_spec.LAYOUTS["linear"], plugin_spec.LAYOUTS["hwc"])


def test_namespace_is_specific_to_this_experiment():
    assert plugin_spec.NAMESPACE not in ("", "tensorrt", "nvidia")
    assert all(name.startswith("Gaia") for name in plugin_spec.FIELDS)


BASELINE_NAMESPACE, BASELINE_LIBRARY, BASELINE_SYMBOL = "gaia_effvit", "libgaia_trt_plugins.so", "gaia_trt_register_plugins"
EXPERIMENT = Path(__file__).resolve().parents[1]


def test_layout_plugins_coexist_with_the_baseline_library():
    """The baseline engines (namespace gaia_effvit, libgaia_trt_plugins.so) must stay loadable in the same process."""
    assert plugin_spec.NAMESPACE == "gaia_effvit_layout" != BASELINE_NAMESPACE
    assert plugin_spec.REGISTER_SYMBOL == "gaia_trt_layout_register_plugins" != BASELINE_SYMBOL
    assert plugin_spec.LIBRARY == "libgaia_trt_layout_plugins.so" != BASELINE_LIBRARY
    assert re.search(rf'extern "C" .*bool {plugin_spec.REGISTER_SYMBOL}\(\);', HEADER)
    assert BASELINE_SYMBOL + "(" not in HEADER.replace(plugin_spec.REGISTER_SYMBOL, "")
    makefile = (EXPERIMENT / "Makefile").read_text()
    assert plugin_spec.LIBRARY in makefile and BASELINE_LIBRARY not in makefile
    assert "-fvisibility=hidden" in makefile  # no symbol interposition between the libraries
    runner = (EXPERIMENT / "runner" / "runner.cpp").read_text()
    assert f"{plugin_spec.REGISTER_SYMBOL}()" in runner and BASELINE_SYMBOL + "(" not in runner
    runtime = (EXPERIMENT / "docker" / "runtime" / "Dockerfile").read_text()
    assert plugin_spec.LIBRARY in runtime and BASELINE_LIBRARY not in runtime
    loader = (EXPERIMENT / "exporter" / "trt_build.py").read_text()
    assert "plugin_spec.REGISTER_SYMBOL" in loader or plugin_spec.REGISTER_SYMBOL in loader


def test_field_values_must_be_complete_ints():
    import pytest

    with pytest.raises(ValueError):
        plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0, "has_res": 1})  # layout missing
    with pytest.raises(ValueError):
        plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0, "has_res": 0, "layout": 0, "extra": 1})
    assert plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0, "has_res": 1, "layout": 2}) == [("mode", 2), ("act", 0), ("has_res", 1), ("layout", 2)]

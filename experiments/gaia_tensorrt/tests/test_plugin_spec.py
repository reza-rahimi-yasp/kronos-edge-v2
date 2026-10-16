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
        _cpp_string("kAttnPrepName"): [],
        _cpp_string("kAttnFinalName"): _cpp_list("kAttnFinalFields"),
    }
    assert {name: list(fields) for name, fields in plugin_spec.FIELDS.items()} == cpp


def test_namespace_is_specific_to_this_experiment():
    assert plugin_spec.NAMESPACE not in ("", "tensorrt", "nvidia")
    assert all(name.startswith("Gaia") for name in plugin_spec.FIELDS)


def test_field_values_must_be_complete_ints():
    import pytest

    with pytest.raises(ValueError):
        plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0})  # has_res missing
    with pytest.raises(ValueError):
        plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0, "has_res": 0, "extra": 1})
    assert plugin_spec.field_values("GaiaEpilogue", {"mode": 2, "act": 0, "has_res": 1}) == [("mode", 2), ("act", 0), ("has_res", 1)]

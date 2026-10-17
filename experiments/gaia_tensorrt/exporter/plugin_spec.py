"""Names, version, namespace and fields of the four plugins, shared with plugins/gaia_plugins.h.

Pure Python: the CPU tests import it to check it against the C++ header.
``make_plugin`` is the only function that touches TensorRT, and only through
the ``trt`` module handed to it, so this module imports nothing heavy.
"""

NAMESPACE = "gaia_effvit"
VERSION = "1"

DEPTHWISE = "GaiaDepthwiseHswish"
EPILOGUE = "GaiaEpilogue"
ATTN_PREP = "GaiaAttnPrep"
ATTN_FINAL = "GaiaAttnFinal"

# Field names in the order the C++ creators advertise them; every field is one INT32.
FIELDS = {
    DEPTHWISE: ("k", "s", "pre", "act", "emit", "in_mode", "out_mode"),
    EPILOGUE: ("mode", "act", "has_res"),
    ATTN_PREP: (),
    ATTN_FINAL: ("batch", "height", "width"),
}


def field_values(name: str, values: dict) -> list[tuple[str, int]]:
    """The (field, value) pairs for ``name`` in declaration order; the dict must name every field, nothing else."""
    if name not in FIELDS:
        raise ValueError(f"unknown plugin {name}")
    expected = FIELDS[name]
    if set(values) != set(expected):
        raise ValueError(f"{name} fields must be exactly {list(expected)}, got {sorted(values)}")
    out = []
    for field in expected:
        value = values[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name}.{field} must be an int, got {value!r}")
        out.append((field, value))
    return out


def make_plugin(trt, registry, name: str, values: dict):
    """Create a plugin instance for the build phase through its registered creator."""
    import numpy as np

    creator = registry.get_creator(name, VERSION, NAMESPACE)
    if creator is None:
        raise RuntimeError(f"plugin creator {NAMESPACE}::{name} v{VERSION} is not registered; load the plugin library first")
    fields = [trt.PluginField(field, np.array([value], dtype=np.int32), trt.PluginFieldType.INT32) for field, value in field_values(name, values)]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"{name} rejected its fields {values}")
    return plugin

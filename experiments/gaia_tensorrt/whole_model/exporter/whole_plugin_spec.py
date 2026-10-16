"""Name, version, namespace, fields and registration symbol of the whole-model plugin, shared with plugin/gaia_whole_model_plugin.h.

Distinct from the hybrid experiment's ``plugin_spec`` on every count (namespace,
names, exported symbol), so both libraries can be loaded in one process. A CPU
test checks these constants against the C++ header.
"""

NAMESPACE = "gaia_effvit_whole"
VERSION = "1"
NAME = "GaiaEffVitWholeModel"
REGISTER_SYMBOL = "gaia_trt_whole_model_register_plugins"

# Field names in the order the C++ creator advertises them, with their PluginFieldType.
FIELDS = (("program", "INT32"), ("weights", "FLOAT32"), ("cudnn_tf32", "INT32"), ("cublas_tf32", "INT32"))

# The library builds the plugin runs on. They are torch 2.14.1+cu130's own (what kernel.py's
# F.conv2d and torch.bmm call): the dev image's venv carries them, and the runtime image pins
# the same builds from NVIDIA's apt repository.
CUDNN_VERSION = "9.24.0.43"
CUBLAS_VERSION = "13.1.1.3"


def make_plugin(trt, registry, program_ints, weights, cudnn_tf32: int, cublas_tf32: int):
    """A build-phase plugin instance through the registered creator; ``weights`` is the packed float32 blob."""
    import numpy as np

    creator = registry.get_creator(NAME, VERSION, NAMESPACE)
    if creator is None:
        raise RuntimeError(f"plugin creator {NAMESPACE}::{NAME} v{VERSION} is not registered; load the plugin library first")
    program = np.ascontiguousarray(np.asarray(program_ints, dtype=np.int32))
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    fields = [
        trt.PluginField("program", program, trt.PluginFieldType.INT32),
        trt.PluginField("weights", weights, trt.PluginFieldType.FLOAT32),
        trt.PluginField("cudnn_tf32", np.array([int(cudnn_tf32)], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("cublas_tf32", np.array([int(cublas_tf32)], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    plugin = creator.create_plugin(NAME, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"{NAME} rejected its fields")
    return plugin

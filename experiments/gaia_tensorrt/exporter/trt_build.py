"""Shared TensorRT build helpers for the experiment (container only).

The hybrid and the pure engines are built the same way the product builds its
engine (``kronos_edge.tensorrt_check.build``: strongly typed network, no
optimization profile for static shapes), except that ``tf32=False`` clears
the TF32 builder flag TensorRT enables by default for FP32 tensors, which is
the matched control setting of the comparison.
"""

import hashlib
import json
import os
from pathlib import Path

import tensorrt as trt

from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.tensorrt_check import forwarding_logger

logger = get_logger(f"{PACKAGE_LOGGER}.experiment")


_LOGGER = None


class _VerboseLogger(trt.ILogger):
    """TensorRT's messages at every severity on stderr; ``errors`` as the forwarding logger has it."""

    def __init__(self):
        super().__init__()
        self.errors = []

    def log(self, severity, msg):
        import sys

        if severity in (trt.ILogger.Severity.ERROR, trt.ILogger.Severity.INTERNAL_ERROR):
            self.errors.append(msg)
        print(f"[TRT {severity}] {msg}", file=sys.stderr, flush=True)

    @property
    def min_severity(self):
        return trt.ILogger.Severity.VERBOSE


def new_builder():
    # TensorRT keeps the first logger registered in the process and ignores later ones, so one logger
    # object must stay alive for the whole process; it is created once here.
    global _LOGGER
    if _LOGGER is None:
        # GAIA_TRT_VERBOSE=1: TensorRT's own verbose logger instead (the builder's per-tactic timings, for diagnosis).
        _LOGGER = _VerboseLogger() if os.environ.get("GAIA_TRT_VERBOSE") == "1" else forwarding_logger(trt, logger)
    log = _LOGGER
    log.errors.clear()
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return log, builder, network


def build_serialized(log, builder, network, tf32: bool, detailed: bool = False) -> tuple[bytes, dict]:
    config = builder.create_builder_config()
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if detailed:
        # Keeps per-layer formats in the engine for the inspector (inspect_engine.py); benchmarked
        # engines are built without it, like the product's.
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    enabled = [name for name, value in trt.BuilderFlag.__members__.items() if config.get_flag(value)]
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("; ".join(log.errors) or "the builder returned no engine")
    data = bytes(engine)
    info = {
        "tensorrt": trt.__version__, "builder_flags": config.flags, "enabled_flags": enabled, "tf32": tf32,
        "strongly_typed": True, "engine_bytes": len(data), "engine_sha256": hashlib.sha256(data).hexdigest(),
        "num_layers": network.num_layers, "profiling_verbosity": str(config.profiling_verbosity),
    }
    return data, info


def load_plugins(path: str, symbol: str = "gaia_trt_layout_register_plugins") -> None:
    """Load a plugin library (RTLD_LOCAL, so several can coexist) and register its creators with the global registry."""
    import ctypes

    lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    register = getattr(lib, symbol)
    register.restype = ctypes.c_bool
    if not register():
        raise RuntimeError(f"{symbol}() failed for {path}")
    global _PLUGIN_LIB
    _PLUGIN_LIB = lib  # keep the handle alive for the process


def engine_io(data: bytes) -> list[dict]:
    log, _, _ = new_builder()
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        raise RuntimeError("the serialized engine does not deserialize")
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io.append({"name": name, "mode": str(engine.get_tensor_mode(name)), "dtype": str(engine.get_tensor_dtype(name)),
                   "shape": list(engine.get_tensor_shape(name))})
    return io


def write_engine(path: Path, data: bytes, info: dict) -> None:
    path.write_bytes(data)
    path.with_suffix(".build.json").write_text(json.dumps(info, indent=2))

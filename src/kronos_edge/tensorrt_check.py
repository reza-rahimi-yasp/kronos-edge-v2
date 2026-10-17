"""TensorRT engine check: build a strongly typed engine from the simplified ONNX.

Runs in the model-check subprocess after the ONNX simplification. The simplified
model is parsed from its file in the artifact directory, so external weight
data is found next to it. The network is strongly typed: the engine computes
in the tensor types the ONNX carries, which is how the configured precision
reaches it (TensorRT 11 has no FP16 builder flag; the cast happens at the
ONNX export). An input the model uses as a shape (its values decide a
tensor's shape) needs an optimization profile; the user's input values are
used for it, so the engine is built for exactly the supplied inputs. The
serialized engine is written as ``model.engine`` and kept in state for the
next node.

tensorrt is imported inside ``build``, so a missing package reaches the
report as the ``tensorrt_import`` stage instead of crashing the subprocess.
TensorRT needs the GPU; tests replace ``build``.
"""

from pathlib import Path

from kronos_edge.artifacts import ENGINE_FILE, SIMPLIFIED_FILE
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import TENSORRT as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.tensorrt_check")


class TensorRTParseError(RuntimeError):
    """The ONNX parser rejected the model; the message lists the parser's errors."""


class TensorRTBuildError(RuntimeError):
    """The builder produced no engine; the message lists the errors TensorRT logged."""


def forwarding_logger(trt, target):
    """A TensorRT logger that routes to ``target`` and keeps TensorRT's errors for the customer detail."""

    class Forwarding(trt.ILogger):
        def __init__(self):
            super().__init__()
            self.errors: list[str] = []

        def log(self, severity, msg):
            if severity in (trt.ILogger.Severity.INTERNAL_ERROR, trt.ILogger.Severity.ERROR):
                self.errors.append(msg)
                target.error("tensorrt: %s", msg, extra={"stage": "tensorrt"})
            elif severity == trt.ILogger.Severity.WARNING:
                target.warning("tensorrt: %s", msg, extra={"stage": "tensorrt"})
            else:
                target.debug("tensorrt: %s", msg, extra={"stage": "tensorrt"})

    return Forwarding()


def build(onnx_path, inputs) -> bytes:
    """Parse the ONNX file into a strongly typed network and build a serialized engine.

    ``inputs`` are the user's tensors in forward-argument order, the order of
    the network inputs; they supply the profile for shape inputs and for any
    dynamic dimension.
    """
    import tensorrt as trt

    log = forwarding_logger(trt, logger)
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise TensorRTParseError("; ".join(errors) or "the parser failed without an error message")
    config = builder.create_builder_config()
    profile = _profile(trt, builder, network, inputs)
    if profile is not None:
        config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise TensorRTBuildError("; ".join(log.errors) or "the builder returned no engine")
    return bytes(engine)


def _profile(trt, builder, network, inputs):
    """An optimization profile pinned to the user's inputs, or None when no input needs one."""
    if network.num_inputs != len(inputs):
        raise TensorRTBuildError(
            f"the network has {network.num_inputs} inputs but the model takes {len(inputs)}; cannot build a profile"
        )
    profile = builder.create_optimization_profile()
    needed = False
    for i in range(network.num_inputs):
        tensor, value = network.get_input(i), inputs[i]
        if tensor.is_shape_tensor:
            values = [int(v) for v in value.detach().cpu().flatten().tolist()]
            profile.set_shape_input(tensor.name, values, values, values)
            logger.info("shape input %s pinned to %s", tensor.name, values, extra={"stage": "tensorrt_build"})
            needed = True
        elif -1 in tuple(tensor.shape):
            shape = tuple(value.shape)
            profile.set_shape(tensor.name, shape, shape, shape)
            logger.info("dynamic input %s pinned to %s", tensor.name, shape, extra={"stage": "tensorrt_build"})
            needed = True
    return profile if needed else None


def build_engine(state: State) -> State:
    stage = "tensorrt_parse"
    try:
        artifacts = Path(state["artifacts_dir"])
        precision = state["precision"]
        logger.info("building a TensorRT engine (%s) from %s", precision, artifacts / SIMPLIFIED_FILE, extra={"stage": stage})
        try:
            engine = build(artifacts / SIMPLIFIED_FILE, state["user_inputs"])
        except ImportError:
            stage = "tensorrt_import"
            raise
        except TensorRTParseError:
            raise
        except Exception:
            stage = "tensorrt_build"
            raise

        stage = "engine_save"
        path = artifacts / ENGINE_FILE
        path.write_bytes(engine)
        logger.info("wrote %s (%d bytes)", path, len(engine), extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"engine": engine}

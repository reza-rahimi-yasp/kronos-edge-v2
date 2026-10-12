"""TensorRT engine check: build a strongly typed engine from the simplified ONNX.

Runs inside the container after the ONNX simplification. The simplified
model is parsed from its file in the artifact directory, so external weight
data is found next to it. The network is strongly typed: the engine computes
in the tensor types the ONNX carries, which is how the configured precision
reaches it (TensorRT 11 has no FP16 builder flag; the cast happens at the
ONNX export). The serialized engine is written as ``model.engine`` and kept
in state for the next node.

tensorrt is imported inside ``build``, so a missing package reaches the
report as the ``tensorrt_import`` stage instead of crashing the container.
TensorRT only exists in the image; tests replace ``build``.
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


def build(onnx_path) -> bytes:
    """Parse the ONNX file into a strongly typed network and build a serialized engine."""
    import tensorrt as trt

    class Forwarding(trt.ILogger):
        """Route TensorRT's log to ours and keep its errors for the customer detail."""

        def __init__(self):
            super().__init__()
            self.errors: list[str] = []

        def log(self, severity, msg):
            if severity in (trt.ILogger.Severity.INTERNAL_ERROR, trt.ILogger.Severity.ERROR):
                self.errors.append(msg)
                logger.error("tensorrt: %s", msg, extra={"stage": "tensorrt"})
            elif severity == trt.ILogger.Severity.WARNING:
                logger.warning("tensorrt: %s", msg, extra={"stage": "tensorrt"})
            else:
                logger.debug("tensorrt: %s", msg, extra={"stage": "tensorrt"})

    log = Forwarding()
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise TensorRTParseError("; ".join(errors) or "the parser failed without an error message")
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    if engine is None:
        raise TensorRTBuildError("; ".join(log.errors) or "the builder returned no engine")
    return bytes(engine)


def build_engine(state: State) -> State:
    stage = "tensorrt_parse"
    try:
        artifacts = Path(state["artifacts_dir"])
        precision = state["precision"]
        logger.info("building a TensorRT engine (%s) from %s", precision, artifacts / SIMPLIFIED_FILE, extra={"stage": stage})
        try:
            engine = build(artifacts / SIMPLIFIED_FILE)
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

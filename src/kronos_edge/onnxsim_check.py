"""ONNX simplification check: run onnxsim on the exported model and keep the result.

Runs inside the container after the ONNX export, on the model in state.
The simplified model stays in state for the next node and is written to the
artifact directory as ``model_simplified.onnx``, with the weights in a
companion ``.data`` file when they exceed the protobuf limit.

onnx and onnxsim are imported inside the node, so a missing package reaches
the report as a stage instead of crashing the container at startup.
"""

from pathlib import Path

from kronos_edge.artifacts import EXTERNAL_DATA_SUFFIX, SIMPLIFIED_FILE
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import ONNXSIM as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.onnxsim_check")

EXTERNAL_DATA_THRESHOLD = 2**31 - 1  # protobuf's limit on a single serialized message


def simplify(model):
    """onnxsim on a ModelProto; returns the simplified model and whether onnxsim's own check passed."""
    import onnxsim

    return onnxsim.simplify(model)


def simplify_onnx(state: State) -> State:
    stage = "onnx_simplify"
    try:
        logger.info("simplifying the ONNX model with onnxsim", extra={"stage": stage})
        try:
            simplified, check_ok = simplify(state["onnx_model"])
        except ImportError:
            stage = "onnx_simplifier"
            raise
        import onnx

        stage = "onnx_simplified_invalid"
        if not check_ok:
            raise ValueError("onnxsim reported that its check of the simplified model failed")

        stage = "onnx_simplified_save"
        path = Path(state["artifacts_dir"]) / SIMPLIFIED_FILE
        external = simplified.ByteSize() > EXTERNAL_DATA_THRESHOLD
        onnx.save(
            simplified, path, save_as_external_data=external,
            location=f"{SIMPLIFIED_FILE}{EXTERNAL_DATA_SUFFIX}", size_threshold=0,  # all weights, like ONNXProgram.save
        )
        if external:
            # onnx.save replaced the in-memory weights with file references; pull them back for the next node.
            onnx.external_data_helper.load_external_data_for_model(simplified, str(path.parent))
        logger.info("wrote %s%s", path, " with external data" if external else "", extra={"stage": stage})

        stage = "onnx_simplified_invalid"
        try:
            onnx.checker.check_model(path)  # by path: works past the 2 GB limit
        except Exception:
            path.unlink(missing_ok=True)
            (Path(state["artifacts_dir"]) / f"{SIMPLIFIED_FILE}{EXTERNAL_DATA_SUFFIX}").unlink(missing_ok=True)
            raise
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"simplified_onnx": simplified}

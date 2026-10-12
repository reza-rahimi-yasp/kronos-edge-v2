"""ONNX simplification check: run onnxsim on the exported model and keep the result.

Runs inside the container after the ONNX export, on the model in state.
The simplified model stays in state for the next node and is written to the
artifact directory as ``model_simplified.onnx``.
"""

from pathlib import Path

import onnx
import onnxsim

from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import ONNXSIM as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.onnxsim_check")

SIMPLIFIED_FILE = "model_simplified.onnx"


def simplify(model):
    """onnxsim on a ModelProto; returns the simplified model and whether onnxsim's own check passed."""
    return onnxsim.simplify(model)


def simplify_onnx(state: State) -> State:
    stage = "onnx_simplify"
    try:
        logger.info("simplifying the ONNX model with onnxsim", extra={"stage": stage})
        simplified, check_ok = simplify(state["onnx_model"])

        stage = "onnx_simplified_invalid"
        if not check_ok:
            raise ValueError("onnxsim reported that its check of the simplified model failed")
        onnx.checker.check_model(simplified)

        stage = "onnx_simplified_save"
        path = Path(state["artifacts_dir"]) / SIMPLIFIED_FILE
        onnx.save(simplified, path)
        logger.info("wrote %s", path, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"simplified_onnx": simplified}

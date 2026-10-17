"""ONNX export check: export the eagerly-run model with the dynamo exporter.

Runs in the model-check subprocess after the eager check, on the model and inputs
already in state. A copy of the model and its floating-point inputs is cast
to the configured precision first, whatever precision the user wrote the
model in, so the ONNX, and the TensorRT engine built from it, carry that
precision; the eager model in state stays as the user wrote it. No copy is
made when everything already has that dtype. The exported model stays in state for the next node and is written
to the artifact directory as ``model.onnx`` through ``ONNXProgram.save``,
which moves the weights to a companion ``.data`` file when they exceed the
protobuf limit. onnx and onnxscript are imported lazily by the exporter, so
a missing package is the ``onnx_exporter`` stage.
"""

import copy
import re
from pathlib import Path

import torch
from torch.onnx._internal.exporter._errors import TorchExportError

from kronos_edge.artifacts import EXPORTED_FILE
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import ONNX as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.onnx_check")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY = "## Exception summary"


DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def cast_copy(model, dtype):
    """A copy of the model with its floating-point tensors in ``dtype``; the original is untouched."""
    return copy.deepcopy(model).to(dtype)


def cast_inputs(inputs, dtype):
    return tuple(t.to(dtype) if t.is_floating_point() else t.clone() for t in inputs)


def _already(model, inputs, dtype) -> bool:
    tensors = [*model.parameters(), *model.buffers(), *inputs]
    return all(t.dtype == dtype for t in tensors if t.is_floating_point())


def export(model, inputs):
    """Export with torch.onnx (dynamo=True) under no_grad, like the eager pass; returns the ONNXProgram."""
    with torch.no_grad():
        return torch.onnx.export(model, inputs, None, dynamo=True, verbose=False, report=False, verify=False)


class OnnxExportFailed(RuntimeError):
    """The exporter's cause only: its 'Exception summary' section, colour codes stripped."""

    def __init__(self, exc: Exception):
        text = _ANSI.sub("", str(exc))
        _, found, summary = text.partition(_SUMMARY)
        super().__init__(summary.strip() if found else text.strip())


def export_onnx(state: State) -> State:
    stage = "onnx_precision"
    try:
        model, inputs = state["model"], state["user_inputs"]
        precision = state.get("precision", "fp32")
        dtype = DTYPES[precision]
        if _already(model, inputs, dtype):
            inputs = tuple(t.clone() for t in inputs)  # the export runs the model; keep the state's inputs pristine
        else:
            logger.info("casting a copy of the model and its float inputs to %s", dtype, extra={"stage": stage})
            model, inputs = cast_copy(model, dtype), cast_inputs(inputs, dtype)

        stage = "onnx_capture"
        logger.info("exporting to ONNX (%s) with dynamo=True", precision, extra={"stage": stage})
        try:
            program = export(model, inputs)
        except ImportError:
            # The dynamo exporter imports onnx and onnxscript lazily; this environment lacks them.
            stage = "onnx_exporter"
            raise
        except TorchExportError as exc:
            raise OnnxExportFailed(exc) from exc
        except Exception as exc:
            stage = "onnx_convert"
            raise OnnxExportFailed(exc) from exc

        stage = "onnx_save"
        path = Path(state["artifacts_dir"]) / EXPORTED_FILE
        program.save(path)
        logger.info("wrote %s", path, extra={"stage": stage})
        model = program.model_proto  # materializes the IR; guarded so a failure is still a stage
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"onnx_model": model}

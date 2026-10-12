"""ONNX export check: export the eagerly-run model with the dynamo exporter.

Runs inside the container after the eager check, on the model and inputs
already in state. The exported program is discarded; only success or the
failure stage and its cause are reported.
"""

import re

import torch
from torch.onnx._internal.exporter._errors import TorchExportError

from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import ONNX as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.onnx_check")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY = "## Exception summary"


def export(model, inputs):
    """Export with torch.onnx (dynamo=True) under no_grad, like the eager pass; nothing is written."""
    with torch.no_grad():
        return torch.onnx.export(model, inputs, None, dynamo=True, verbose=False, report=False, verify=False)


class OnnxExportFailed(RuntimeError):
    """The exporter's cause only: its 'Exception summary' section, colour codes stripped."""

    def __init__(self, exc: Exception):
        text = _ANSI.sub("", str(exc))
        _, found, summary = text.partition(_SUMMARY)
        super().__init__(summary.strip() if found else text.strip())


def export_onnx(state: State) -> State:
    stage = "onnx_capture"
    try:
        logger.info("exporting to ONNX with dynamo=True", extra={"stage": stage})
        try:
            export(state["model"], state["user_inputs"])
        except ImportError:
            # The dynamo exporter imports onnx and onnxscript lazily; the image lacks them.
            stage = "onnx_exporter"
            raise
        except TorchExportError as exc:
            raise OnnxExportFailed(exc) from exc
        except Exception as exc:
            stage = "onnx_convert"
            raise OnnxExportFailed(exc) from exc
    except Exception as exc:
        return fail(logger, stage, exc)
    return {}

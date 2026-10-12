"""pt2 export check: trace the eagerly-run model with torch.export and save the reference kernelgen optimizes.

Runs inside the container right after the eager check, on the model and
inputs already in state and before the ONNX export, so the reference the
host's kernelgen step needs exists as early as possible. The model is traced
in inference mode under ``torch.no_grad()`` and decomposed to Core ATen,
which is what ``yasp-toolkit kernelgen export`` produces and what
``kernelgen compile --reference`` expects; the toolkit itself cannot run here
because the image has no yasp-agent and the host never runs user code. The
model is exported as the user wrote it: precision is applied at the ONNX
export only. The exported program stays in state and is written to the
artifact directory as ``model.pt2`` with ``torch.export.save``, weights
bundled.
"""

import re
from pathlib import Path

import torch

from kronos_edge.artifacts import PT2_FILE
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import PT2 as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.pt2_check")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Pt2ExportFailed(RuntimeError):
    """torch.export's message, colour codes stripped."""

    def __init__(self, exc: Exception):
        super().__init__(_ANSI.sub("", str(exc)).strip())


def export(model, inputs):
    """torch.export in inference mode, decomposed to Core ATen, as ``yasp-toolkit kernelgen export`` does."""
    with torch.no_grad():
        exported = torch.export.export(model, tuple(inputs))
    return exported.run_decompositions()


def save(program, path) -> None:
    torch.export.save(program, str(path))


def export_pt2(state: State) -> State:
    stage = "pt2_capture"
    try:
        logger.info("tracing with torch.export for the kernelgen reference", extra={"stage": stage})
        try:
            program = export(state["model"], state["user_inputs"])
        except Exception as exc:
            raise Pt2ExportFailed(exc) from exc

        stage = "pt2_save"
        path = Path(state["artifacts_dir"]) / PT2_FILE
        save(program, path)
        logger.info("wrote %s", path, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"exported_program": program}

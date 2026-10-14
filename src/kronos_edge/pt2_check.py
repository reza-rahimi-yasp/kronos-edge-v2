"""pt2 export check: trace the eagerly-run model with torch.export and save the reference kernelgen optimizes.

Runs inside the container right after the eager check, on the model and
inputs already in state and before the ONNX export, so the reference the
host's kernelgen step needs exists as early as possible. The model is traced
in inference mode under ``torch.no_grad()`` and decomposed to Core ATen,
which is what ``yasp-toolkit kernelgen export`` produces and what
``kernelgen compile --reference`` expects; the toolkit itself cannot run here
because the image has no yasp-agent and the host never runs user code. The
model is exported as the user wrote it: precision is applied at the ONNX
export only. It is traced from independent CPU copies of the model and the
inputs: ``torch.export.save`` records the trace device in every weight and
factory op, and a reference traced on the GPU failed on the kernelgen
service with ``Torch not compiled with CUDA enabled``, the error a torch
built without CUDA raises when loading such an archive, while CPU-traced
references were accepted (the service's build is inferred from that, not
observed). The copies remove that loading failure and nothing more. The
eager model and the user's inputs in state are never moved or written to.
The exported program stays in state and is written to
the artifact directory as ``model.pt2`` with ``torch.export.save``, weights
bundled.
"""

import copy
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


def cpu_copies(model, inputs):
    """Independent CPU copies of the model and the inputs, dtypes and mode kept; the originals are untouched.

    Copies even when already on the CPU, so nothing the trace touches is shared with the eager model.
    """
    return copy.deepcopy(model).to("cpu"), tuple(t.detach().to("cpu", copy=True) for t in inputs)


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
        logger.info("tracing CPU copies of the model and inputs with torch.export for the kernelgen reference", extra={"stage": stage})
        try:
            program = export(*cpu_copies(state["model"], state["user_inputs"]))
        except Exception as exc:
            raise Pt2ExportFailed(exc) from exc

        stage = "pt2_save"
        path = Path(state["artifacts_dir"]) / PT2_FILE
        save(program, path)
        logger.info("wrote %s", path, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"exported_program": program}

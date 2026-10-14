"""pt2 export check: export the eagerly-run model with yasp's exporter and save the reference kernelgen optimizes.

Runs inside the container right after the eager check, on the model and
inputs already in state and before the ONNX export, so the reference the
host's kernelgen step needs exists as early as possible. The export is
``yasp.compile.common.exporter`` (``export`` then ``save``), the functions
``yasp-toolkit kernelgen export`` runs: inference mode, Core ATen
decompositions, ``torch.export.save``. The image carries that package for
this and nothing else of the toolkit; the exporter needs no key, and the host
never runs user code. It is imported inside the node, so an image without it
reports the ``pt2_exporter`` stage. The
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
the artifact directory as ``model.pt2`` by the exporter's ``save``, weights
bundled.
"""

import copy
import re
from pathlib import Path

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


def import_exporter():
    """yasp's exporter module, imported here so a missing package reaches the report as the ``pt2_exporter`` stage."""
    from yasp.compile.common import exporter

    return exporter


def export_pt2(state: State) -> State:
    stage = "pt2_exporter"
    try:
        exporter = import_exporter()

        stage = "pt2_capture"
        logger.info("exporting CPU copies of the model and inputs with yasp's exporter for the kernelgen reference", extra={"stage": stage})
        model, inputs = cpu_copies(state["model"], state["user_inputs"])
        try:
            program = exporter.export(model, inputs)
        except Exception as exc:
            raise Pt2ExportFailed(exc) from exc

        stage = "pt2_save"
        path = Path(state["artifacts_dir"]) / PT2_FILE
        exporter.save(program, path)
        logger.info("wrote %s", path, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"exported_program": program}

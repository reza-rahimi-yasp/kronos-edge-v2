"""Kernelgen check: generate an optimized kernel with Gaia, then show the report it came with.

Runs on the host, after the export and the evaluator check: kernelgen is a hosted
service reached through the ``yasp-toolkit`` command of the yasp-agent package,
which needs the user's Yasp API key (``YASP_API_KEY`` in the environment, or the
OS keyring filled by ``yasp-toolkit keyring set``; see ``.env.example``), so it
never runs in a container. The node follows the documented starting workflow
(https://compile.yasp.ai/help/gaia/generate-your-first-kernel/):

- ``yasp-toolkit kernelgen compile`` on the exported ``model.pt2`` for the config's
  GPU, with its in-compile evaluations routed by the config's ``eval_routing`` to
  the user's Gaia worker (https://compile.yasp.ai/help/reference/kernelgen/#compile).
  The reference alone is uploaded: no ``--source``, so no user Python runs on the
  service's side. The command polls the job until it is terminal; the kernel lands
  in the artifact directory as ``kernel.py`` next to the job's
  ``kernelgen_report.json``.
- ``yasp-toolkit eval inspect --file kernelgen_report.json``
  (https://compile.yasp.ai/help/reference/eval/#inspect) renders that report,
  speedup and correctness included, for the user. Gaia returns only a kernel that
  passed its worker's checks against the reference; the check reads no field of
  the report itself, since the docs publish no schema for it.

The toolkit can exit 0 after an error that stops its flow early, so the node
requires the files it asked for instead of trusting the exit code. The host
never imports yasp-agent: the toolkit is a subprocess, so this module stays free
of the model stack.
"""

import os
import subprocess
from pathlib import Path

from kronos_edge.artifacts import ENGINE_FILE, KERNEL_FILE, PT2_FILE
from kronos_edge.artifacts import KERNELGEN_REPORT_FILE as REPORT_FILE
from kronos_edge.log import get_logger
from kronos_edge.messages import KERNELGEN as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)

TOOLKIT = "yasp-toolkit"


class ToolkitFailed(RuntimeError):
    """The toolkit exited with an error, or returned without the file it was asked for; the message is its output."""


def run_toolkit(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run yasp-toolkit with ``args``, capturing its output; a missing command raises FileNotFoundError."""
    # COLUMNS: the toolkit renders wide tables, which wrap at the 80 columns it assumes without a terminal
    # (https://compile.yasp.ai/help/install/#optional-widen-table-output).
    return subprocess.run(
        [TOOLKIT, *args], capture_output=True, text=True, env={**os.environ, "COLUMNS": "200"}, timeout=timeout,
    )


def run_logged(stage: str, args: list[str], timeout: float | None = None) -> tuple[subprocess.CompletedProcess, str]:
    """Run the toolkit and return the result with its combined output; the output goes to the log, and to the detail on failure."""
    logger.info("running %s %s", TOOLKIT, " ".join(args), extra={"stage": stage})
    result = run_toolkit(args, timeout=timeout)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    for line in output.splitlines():
        logger.debug("%s: %s", TOOLKIT, line, extra={"stage": stage})
    if result.returncode != 0:
        raise ToolkitFailed(f"{TOOLKIT} exited {result.returncode}\n{output}")
    return result, output


def kernelgen(state: State) -> State:
    stage = "kernelgen_reference"
    try:
        artifacts = Path(state["artifacts_path"]).resolve()
        reference = artifacts / PT2_FILE
        if not reference.is_file():
            raise FileNotFoundError(f"{reference} was not written by the export")

        stage = "kernelgen_engine"  # the TensorRT output this step waits for
        engine = artifacts / ENGINE_FILE
        if not engine.is_file():
            raise FileNotFoundError(f"{engine} was not written by the container")

        stage = "kernelgen_compile"
        config = state["config"]
        kernel, report = artifacts / KERNEL_FILE, artifacts / REPORT_FILE
        args = [
            "kernelgen", "compile", "--reference", str(reference), "--gpu-name", config.gpu,
            "--output", str(kernel), "--report", str(report), "--no-progress", "--eval-routing", config.eval_routing,
        ]
        logger.info("generating a kernel with kernelgen for %s", config.gpu, extra={"stage": stage})
        try:
            _, output = run_logged(stage, args)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        for path in (kernel, report):
            if not path.is_file():
                raise ToolkitFailed(f"{TOOLKIT} exited 0 without writing {path.name}\n{output}")
        logger.info("wrote %s and %s", kernel, report, extra={"stage": stage})

        stage = "kernelgen_inspect"
        _, rendered = run_logged(stage, ["eval", "inspect", "--file", str(report)])
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"kernel_path": str(kernel), "kernelgen_report": rendered}

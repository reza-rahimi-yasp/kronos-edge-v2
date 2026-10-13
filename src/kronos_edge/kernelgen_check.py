"""Kernelgen check: optimize the model with yasp kernelgen, then score the kernel on this machine's evaluator.

Runs on the host, after the container: kernelgen is a hosted service reached
through the ``yasp-toolkit`` command of the yasp-agent package, which needs
the user's Yasp API key (``YASP_API_KEY`` in the environment, or the OS
keyring filled by ``yasp-toolkit keyring set``; see ``.env.example``), so it
cannot run inside the environment container. Two nodes:

- ``kernelgen``: once the container has written the reference ``model.pt2``
  and the TensorRT ``model.engine`` it waits for, runs
  ``yasp-toolkit kernelgen compile`` on the reference for the config's GPU,
  with its in-compile evaluations routed by the config's ``eval_routing``.
  The command polls the job until it is terminal; the optimized kernel lands
  in the artifact directory as ``kernel.py`` next to the job's
  ``kernelgen_report.json``.
- ``evaluate_kernelgen``: runs ``yasp-toolkit eval compare`` on that kernel
  against the reference on this machine, the evaluator registered from it
  under the config's ``eval_routing`` key and advertising the config's GPU,
  and writes the evaluator's ``results.json`` as ``kernelgen_eval.json``. The
  kernel must pass the evaluator's correctness checks; the speedup it
  measured is reported, not judged. No expectation is derived from the
  TensorRT engine or the eager run.

The toolkit exits 0 after an HTTP error that stops its flow early, so each
node requires the file it asked for instead of trusting the exit code. The
host never imports yasp-agent: the toolkit is a subprocess, so this module
stays free of the model stack.
"""

import json
import os
import subprocess
from pathlib import Path

from kronos_edge.artifacts import ENGINE_FILE, KERNEL_FILE, PT2_FILE
from kronos_edge.artifacts import KERNELGEN_EVAL_FILE as EVAL_FILE
from kronos_edge.artifacts import KERNELGEN_REPORT_FILE as REPORT_FILE
from kronos_edge.log import get_logger
from kronos_edge.messages import KERNELGEN as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)

TOOLKIT = "yasp-toolkit"
# The title of the panel the toolkit prints when it finds no API key in the environment or the keyring.
MISSING_TOKEN = "Missing authentication token"
# The hosted API uses this code when the key is missing or invalid.
UNAUTHORIZED = "ERROR_UNAUTHORIZED"
# `eval compare` keys its results.json by implementation slot; the kernel is the only one.
RESULT_KEY = "opt_1"


class MissingCredentials(RuntimeError):
    """The toolkit found no valid Yasp API key."""


class ToolkitFailed(RuntimeError):
    """The toolkit exited with an error, or returned without the file it was asked for; the message is its output."""


def run_toolkit(args: list[str]) -> subprocess.CompletedProcess:
    """Run yasp-toolkit with ``args``, capturing its output; a missing command raises FileNotFoundError."""
    # COLUMNS: the toolkit renders panels; at the 80 columns it assumes without a terminal they wrap mid-sentence.
    return subprocess.run([TOOLKIT, *args], capture_output=True, text=True, env={**os.environ, "COLUMNS": "200"})


def _invoke(stage: str, args: list[str], expected: Path) -> None:
    """Run the toolkit and require ``expected`` to exist afterwards; its output goes to the log, and to the detail on failure."""
    logger.info("running %s %s", TOOLKIT, " ".join(args), extra={"stage": stage})
    result = run_toolkit(args)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    for line in output.splitlines():
        logger.debug("%s: %s", TOOLKIT, line, extra={"stage": stage})
    if MISSING_TOKEN in output or UNAUTHORIZED in output:
        raise MissingCredentials(output)
    if result.returncode != 0:
        raise ToolkitFailed(f"{TOOLKIT} exited {result.returncode}\n{output}")
    if not expected.is_file():
        raise ToolkitFailed(f"{TOOLKIT} exited 0 without writing {expected.name}\n{output}")


def kernelgen(state: State) -> State:
    stage = "kernelgen_reference"
    try:
        artifacts = Path(state["artifacts_path"]).resolve()
        reference = artifacts / PT2_FILE
        if not reference.is_file():
            raise FileNotFoundError(f"{reference} was not written by the container")

        stage = "kernelgen_engine"  # the TensorRT output this step waits for
        engine = artifacts / ENGINE_FILE
        if not engine.is_file():
            raise FileNotFoundError(f"{engine} was not written by the container")

        stage = "kernelgen_compile"
        config = state["config"]
        kernel = artifacts / KERNEL_FILE
        args = [
            "kernelgen", "compile", "--reference", str(reference), "--source", state["model_path"],
            "--gpu-name", config.gpu, "--output", str(kernel), "--report", str(artifacts / REPORT_FILE), "--no-progress",
            "--eval-routing", config.eval_routing,
        ]
        logger.info("optimizing the model with kernelgen for %s", config.gpu, extra={"stage": stage})
        try:
            _invoke(stage, args, kernel)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        except MissingCredentials:
            stage = "kernelgen_credentials"
            raise
        logger.info("wrote %s", kernel, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"kernel_path": str(kernel)}


def evaluate_kernelgen(state: State) -> State:
    stage = "kernelgen_eval"
    try:
        artifacts = Path(state["artifacts_path"]).resolve()
        config = state["config"]
        results = artifacts / EVAL_FILE
        args = [
            "eval", "compare", "--reference", str(artifacts / PT2_FILE), "--implementation", state["kernel_path"],
            "--source", state["model_path"], "--gpu-name", config.gpu, "--output", str(results), "--no-progress",
            "--routing", config.eval_routing,
        ]
        logger.info(
            "evaluating %s on this machine (%s, routing %s)", state["kernel_path"], config.gpu, config.eval_routing,
            extra={"stage": stage},
        )
        try:
            _invoke(stage, args, results)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        except MissingCredentials:
            stage = "kernelgen_credentials"
            raise

        stage = "kernelgen_eval_results"
        result = _read_result(results)

        stage = "kernelgen_eval_incorrect"
        checks = result.get("correctness_checks") or {}
        if not checks.get("all_checks_passed"):
            raise ValueError(_describe_failure(result))

        stage = "kernelgen_eval_results"
        speedup = float(result["speedup"])
        logger.info("the kernel passed the correctness checks; speedup %.2fx", speedup, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"kernelgen_result": result, "kernelgen_speedup": speedup}


def _read_result(path: Path) -> dict:
    """The kernel's entry of the evaluator's results.json."""
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"{path.name} is not JSON: {exc}") from None
    result = payload.get(RESULT_KEY) if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} has no {RESULT_KEY} entry")
    return result


def _describe_failure(result: dict) -> str:
    """Which checks failed, at which evaluator stage, with whatever the evaluator captured."""
    checks = result.get("correctness_checks") or {}
    failed = [k for k, v in checks.items() if k.startswith("all_") and k != "all_checks_passed" and v is False]
    parts = [f"evaluator stage {result.get('stage')!r}"]
    parts.append(f"failed checks: {', '.join(failed)}" if failed else "no correctness_checks passed in the report")
    context = result.get("stage_context") or {}
    for name in ("stderr", "build_stderr", "traceback"):
        text = context.get(name)
        if text and str(text).strip():
            parts.append(f"{name}:\n{str(text).strip()}")
    return "; ".join(parts)

"""Preflight: the Python environment the check was started in must be the matched registry entry.

One local workflow serves a native install and a container built from the
entry's Dockerfile alike: whichever it is, the check runs the model in this
interpreter, so this is what is verified. A probe runs as a subprocess of
this interpreter (``python -m kronos_edge.preflight``) and prints what it
finds as JSON: torch's version and CUDA build, TensorRT's version, and the
GPU torch sees. The workflow process itself never imports the model stack.
Versions match by component: the registry's ``11.4.0`` accepts TensorRT's
``11.4.0.106`` and ``2.14.1`` accepts torch's ``2.14.1+cu130``. Every
difference is reported at once, with the entry's Dockerfile as the recipe.
The driver version is not checked against this machine, and nothing asks
whether this is a container. The ``yasp-toolkit`` command must be on the
path, since the export and kernelgen steps run it.
"""

import json
import shutil
import subprocess
import sys

from kronos_edge.config import Environment
from kronos_edge.kernelgen_check import TOOLKIT
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.model_check import process_environment
from kronos_edge.messages import PREFLIGHT as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: the probe runs this module under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.preflight")


def probe() -> dict:
    """What this interpreter has, as the packages report it; ``None`` where a package or the GPU is missing.

    Runs in the subprocess: the model-stack imports stay out of the workflow process.
    """
    report = {"torch": None, "cuda": None, "tensorrt": None, "gpu": None}
    try:
        import torch
    except ImportError:
        pass
    else:
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    try:
        import tensorrt
    except ImportError:
        pass
    else:
        report["tensorrt"] = tensorrt.__version__
    return report


def probe_command() -> list[str]:
    return [sys.executable, "-m", "kronos_edge.preflight"]


def run_probe() -> dict:
    """Run the probe as a subprocess of this interpreter and return its report; its last stdout line is the JSON."""
    result = subprocess.run(probe_command(), capture_output=True, text=True, env=process_environment())
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0:
        raise RuntimeError(f"probe exited {result.returncode}\n{result.stdout}{result.stderr}".rstrip())
    try:
        report = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        raise RuntimeError(f"probe printed no report\n{result.stdout}{result.stderr}".rstrip())
    return report


def toolkit_on_path() -> str | None:
    return shutil.which(TOOLKIT)


def matches(installed: str | None, required: str) -> bool:
    """Whether an installed version is the required one: same leading components, build labels aside."""
    if installed is None:
        return False
    components = installed.split("+", 1)[0].split(".")
    wanted = required.split(".")
    return components[: len(wanted)] == wanted


def differences(report: dict, environment: Environment) -> list[str]:
    """Each way this environment differs from the registry entry, as a line for the user; empty when it is the entry."""
    lines = []
    if not matches(report.get("torch"), environment.torch):
        lines.append(f"torch: requested {environment.torch}, {_found(report.get('torch'))}")
    if not matches(report.get("cuda"), environment.cuda):
        found = "found a torch without CUDA" if report.get("torch") and not report.get("cuda") else _found(report.get("cuda"))
        lines.append(f"cuda: requested {environment.cuda}, {found}")
    if not matches(report.get("tensorrt"), environment.tensorrt):
        lines.append(f"tensorrt: requested {environment.tensorrt}, {_found(report.get('tensorrt'))}")
    if report.get("gpu") != environment.gpu:
        found = f"found {report['gpu']}" if report.get("gpu") else "none available to torch"
        lines.append(f"gpu: requested {environment.gpu}, {found}")
    return lines


def _found(version: str | None) -> str:
    return f"found {version}" if version else "not installed"


def preflight(state: State) -> State:
    stage = "preflight_probe"
    try:
        environment = state["environment"]
        logger.info("probing this Python environment with %s", " ".join(probe_command()), extra={"stage": stage})
        report = run_probe()
        logger.info("found %s", ", ".join(f"{k}={v}" for k, v in report.items()), extra={"stage": stage})

        stage = "environment_mismatch"
        lines = differences(report, environment)
        if lines:
            raise ValueError(
                "\n".join([*lines, f"the recipe for environment {environment.name} is {environment.dockerfile}"])
            )

        stage = "kernelgen_toolkit"
        toolkit = toolkit_on_path()
        if toolkit is None:
            raise FileNotFoundError(f"{TOOLKIT} is not on the PATH")
        logger.info("this environment is %s; %s is %s", environment.name, TOOLKIT, toolkit, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {}


if __name__ == "__main__":
    print(json.dumps(probe()))

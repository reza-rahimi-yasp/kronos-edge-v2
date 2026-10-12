"""Run the eager check inside the matched environment's container, on the GPU."""

import json
import subprocess
from pathlib import Path

from kronos_edge.log import get_logger
from kronos_edge.state import State, fail

logger = get_logger(__name__)

DEVICE = "cuda"

CUSTOMER_MESSAGES = {
    "container_run": "The check could not be run in the environment container. Contact support.",
}


def build_command(image: str, model_path, npz_path, device: str, log_level: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{Path(model_path).resolve()}:/work/model.py:ro",
        "-v", f"{Path(npz_path).resolve()}:/work/input.npz:ro",
        image,
        "python", "-m", "kronos_edge.eager_check", "/work/model.py", "/work/input.npz",
        "--device", device, "--log-level", log_level,
    ]


def run_container(image: str, model_path, npz_path, device: str, log_level: str) -> str:
    """Return the container's stdout. Its stderr (the logs) is inherited, so it streams through."""
    result = subprocess.run(build_command(image, model_path, npz_path, device, log_level), stdout=subprocess.PIPE, text=True)
    if not result.stdout.strip():
        raise RuntimeError(f"docker run exited {result.returncode} with no report")
    return result.stdout


def run_in_container(state: State) -> State:
    stage = "container_run"
    try:
        image = state["environment"].image
        logger.info("running eager check in %s on %s", image, DEVICE, extra={"stage": stage})
        output = run_container(image, state["model_path"], state["npz_path"], DEVICE, state.get("log_level", "INFO"))
        report = json.loads(output.strip().splitlines()[-1])
    except Exception as exc:
        return fail(logger, stage, exc)
    if report.get("error"):
        logger.info("container reported failure at stage %s", report["error"], extra={"stage": stage})
        return {"error": report["error"], "error_detail": report.get("error_detail", "")}
    return {}

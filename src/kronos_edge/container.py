"""Run the eager check inside the matched environment's container, on the GPU.

The container writes its report to a file in a directory the host mounts, so
nothing the user's model prints on stdout can touch the protocol. The files
the checks produce for the user go to the mounted artifact directory.
"""

import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kronos_edge import artifacts as artifact_files
from kronos_edge.log import get_logger
from kronos_edge.messages import CONTAINER as CUSTOMER_MESSAGES, EAGER, ONNX, ONNXSIM, TENSORRT
from kronos_edge.state import State, fail

logger = get_logger(__name__)

DEVICE = "cuda"
REPORT_FILE = "report.json"



class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str | None
    error_detail: str = ""


def build_command(
    image: str, model_path, npz_path, report_dir, artifacts_dir, precision: str, device: str, log_level: str
) -> list[str]:
    return [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{Path(model_path).resolve()}:/work/model.py:ro",
        "-v", f"{Path(npz_path).resolve()}:/work/input.npz:ro",
        "-v", f"{Path(report_dir).resolve()}:/work/out",
        "-v", f"{Path(artifacts_dir).resolve()}:/work/artifact",
        image,
        "python", "-m", "kronos_edge.eager_check", "/work/model.py", "/work/input.npz",
        "--report", f"/work/out/{REPORT_FILE}", "--artifacts", "/work/artifact",
        "--precision", precision, "--device", device, "--log-level", log_level,
    ]


def run_container(
    image: str, model_path, npz_path, report_dir, artifacts_dir, precision: str, device: str, log_level: str
) -> int:
    """Run the container with stdout and stderr inherited; return its exit code."""
    return subprocess.run(
        build_command(image, model_path, npz_path, report_dir, artifacts_dir, precision, device, log_level)
    ).returncode


def _read_report(report_dir, code: int) -> Report:
    path = Path(report_dir) / REPORT_FILE
    if not path.exists():
        raise RuntimeError(f"container exited {code} without writing a report")
    report = Report.model_validate_json(path.read_text())
    if report.error is not None and report.error not in EAGER | ONNX | ONNXSIM | TENSORRT:
        raise ValueError(f"report names unknown failure stage {report.error!r}")
    expected = 0 if report.error is None else 1
    if code != expected:
        reported = "no error" if report.error is None else f"error at {report.error}"
        raise RuntimeError(f"container exited {code} but reported {reported}")
    return report


def run_in_container(state: State) -> State:
    stage = "container_run"
    try:
        image = state["environment"].image
        artifacts = Path(state["artifacts_path"])
        artifacts.mkdir(parents=True, exist_ok=True)
        artifact_files.clear(artifacts)
        precision = state["config"].precision
        logger.info(
            "running checks in %s on %s at %s, artifacts in %s", image, DEVICE, precision, artifacts, extra={"stage": stage}
        )
        with tempfile.TemporaryDirectory() as report_dir:
            code = run_container(
                image, state["model_path"], state["npz_path"], report_dir, artifacts, precision, DEVICE,
                state.get("log_level", "INFO"),
            )
            report = _read_report(report_dir, code)
    except Exception as exc:
        return fail(logger, stage, exc)
    if report.error is not None:
        logger.info("container reported failure at stage %s", report.error, extra={"stage": stage})
        return {"error": report.error, "error_detail": report.error_detail}
    return {}

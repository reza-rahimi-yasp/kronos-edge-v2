"""Run the eager check inside the matched environment's container, on the GPU.

The container writes its report to a file in a directory the host mounts, so
nothing the user's model prints on stdout can touch the protocol. The files
the checks produce for the user go to the mounted artifact directory.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from kronos_edge import artifacts as artifact_files
from kronos_edge.log import get_logger
from kronos_edge.messages import BENCHMARK, CONTAINER as CUSTOMER_MESSAGES, EAGER, ONNX, ONNXSIM, PT2, TENSORRT
from kronos_edge.state import State, fail

logger = get_logger(__name__)

DEVICE = "cuda"
REPORT_FILE = "report.json"



class Latency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float
    median: float
    p95: float


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eager_ms: Latency
    tensorrt_ms: Latency
    speedup: float
    metric: str
    threshold: float
    values: list[Annotated[float, Field(allow_inf_nan=False)]]


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str | None
    error_detail: str = ""
    benchmark: Benchmark | None = None


def build_command(
    image: str, model_path, npz_path, report_dir, artifacts_dir, precision: str, threshold: float, device: str,
    log_level: str,
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
        "--precision", precision, "--threshold", str(threshold), "--device", device, "--log-level", log_level,
    ]


def run_container(
    image: str, model_path, npz_path, report_dir, artifacts_dir, precision: str, threshold: float, device: str,
    log_level: str,
) -> int:
    """Run the container with stdout and stderr inherited; return its exit code."""
    return subprocess.run(
        build_command(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level)
    ).returncode


def _read_report(report_dir, code: int) -> Report:
    path = Path(report_dir) / REPORT_FILE
    if not path.exists():
        raise RuntimeError(f"container exited {code} without writing a report")
    report = Report.model_validate_json(path.read_text())
    if report.error is not None and report.error not in EAGER | PT2 | ONNX | ONNXSIM | TENSORRT | BENCHMARK:
        raise ValueError(f"report names unknown failure stage {report.error!r}")
    expected = 0 if report.error is None else 1
    if code != expected:
        reported = "no error" if report.error is None else f"error at {report.error}"
        raise RuntimeError(f"container exited {code} but reported {reported}")
    if report.error is None and report.benchmark is None:
        raise ValueError("report claims success but carries no benchmark")
    return report


def run_in_container(state: State) -> State:
    stage = "container_run"
    try:
        image = state["environment"].image
        artifacts = Path(state["artifacts_path"])
        artifacts.mkdir(parents=True, exist_ok=True)
        artifact_files.clear(artifacts)
        config = state["config"]
        logger.info(
            "running checks in %s on %s at %s, artifacts in %s", image, DEVICE, config.precision, artifacts,
            extra={"stage": stage},
        )
        with tempfile.TemporaryDirectory() as report_dir:
            code = run_container(
                image, state["model_path"], state["npz_path"], report_dir, artifacts, config.precision,
                config.threshold, DEVICE, state.get("log_level", "INFO"),
            )
            report = _read_report(report_dir, code)
    except Exception as exc:
        return fail(logger, stage, exc)
    if report.error is not None:
        logger.info("container reported failure at stage %s", report.error, extra={"stage": stage})
        return {"error": report.error, "error_detail": report.error_detail}
    return {"benchmark": report.benchmark.model_dump()}

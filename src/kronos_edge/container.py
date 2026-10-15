"""Run the eager check inside the matched environment's container, on the GPU, then export the kernelgen reference.

The container writes its report to a file in a directory the host mounts, so
nothing the user's model prints on stdout can touch the protocol. The files
the checks produce for the user go to the mounted artifact directory.

The kernelgen reference is produced by the documented ``yasp-toolkit kernelgen
export`` (https://compile.yasp.ai/help/reference/kernelgen/#export), run in a
second container from the export image: CPU-only torch and yasp-agent, no
GPU, no key. It reads model.py the way the exporter documents (``Model``,
``get_inputs``, ``get_init_inputs``), not the benchmarked instance or
input.npz. The config's dependencies are installed there first, under a
constraints file of the image, since the exporter imports model.py. Its
output is captured: the user's model prints on it, and stdout carries the
check's result.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from kronos_edge import artifacts as artifact_files
from kronos_edge.log import get_logger
from kronos_edge.artifacts import PT2_FILE
from kronos_edge.messages import BENCHMARK, CONTAINER as CUSTOMER_MESSAGES, EAGER, ONNX, ONNXSIM, TENSORRT
from kronos_edge.state import State, fail

logger = get_logger(__name__)

DEVICE = "cuda"
REPORT_FILE = "report.json"

EXPORT_IMAGE = "kronos-edge:gaia-export"
EXPORT_DOCKERFILE = "docker/gaia-export/Dockerfile"
# The exit code the export script uses when the config's dependencies could not be installed.
DEPENDENCIES_FAILED = 90
# Runs in the export image: the config's dependencies (its arguments) under a constraints file of the image, then the
# documented export. The requirements arrive as arguments, never in the script's text.
EXPORT_SCRIPT = (
    'if [ "$#" -gt 0 ]; then '
    "pip freeze > /tmp/constraints.txt && pip install --no-cache-dir --constraint /tmp/constraints.txt \"$@\" "
    f"|| exit {DEPENDENCIES_FAILED}; "
    "fi; "
    f"exec yasp-toolkit kernelgen export --input /work/model.py --output /work/artifact/{PT2_FILE}"
)



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
    log_level: str, dependencies: list[str],
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
        *[arg for package in dependencies for arg in ("--dependency", package)],
    ]


def run_container(
    image: str, model_path, npz_path, report_dir, artifacts_dir, precision: str, threshold: float, device: str,
    log_level: str, dependencies: list[str],
) -> int:
    """Run the container with stdout and stderr inherited; return its exit code."""
    return subprocess.run(
        build_command(
            image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies
        )
    ).returncode


def _read_report(report_dir, code: int) -> Report:
    path = Path(report_dir) / REPORT_FILE
    if not path.exists():
        raise RuntimeError(f"container exited {code} without writing a report")
    report = Report.model_validate_json(path.read_text())
    if report.error is not None and report.error not in EAGER | ONNX | ONNXSIM | TENSORRT | BENCHMARK:
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
                config.threshold, DEVICE, state.get("log_level", "INFO"), config.dependencies,
            )
            report = _read_report(report_dir, code)
    except Exception as exc:
        return fail(logger, stage, exc)
    if report.error is not None:
        logger.info("container reported failure at stage %s", report.error, extra={"stage": stage})
        return {"error": report.error, "error_detail": report.error_detail}
    return {"benchmark": report.benchmark.model_dump()}


def build_export_command(model_path, artifacts_dir, dependencies: list[str]) -> list[str]:
    """No GPU and no environment from the host: no credential enters the container that runs the user's model."""
    return [
        "docker", "run", "--rm",
        "-v", f"{Path(model_path).resolve()}:/work/model.py:ro",
        "-v", f"{Path(artifacts_dir).resolve()}:/work/artifact",
        EXPORT_IMAGE,
        "sh", "-c", EXPORT_SCRIPT, "kronos-export", *dependencies,
    ]


def run_export(model_path, artifacts_dir, dependencies: list[str]) -> subprocess.CompletedProcess:
    """Run the export container, capturing its output; a missing docker command raises FileNotFoundError."""
    return subprocess.run(build_export_command(model_path, artifacts_dir, dependencies), capture_output=True, text=True)


def export_reference(state: State) -> State:
    stage = "export_run"
    try:
        artifacts = Path(state["artifacts_path"])
        logger.info(
            "exporting the kernelgen reference with yasp-toolkit kernelgen export in %s", EXPORT_IMAGE,
            extra={"stage": stage},
        )
        result = run_export(state["model_path"], artifacts, state["config"].dependencies)
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        for line in output.splitlines():
            logger.debug("%s", line, extra={"stage": stage})
        if result.returncode == DEPENDENCIES_FAILED:
            stage = "export_dependencies"
            raise RuntimeError(f"pip exited with an error in {EXPORT_IMAGE}\n{output}")
        stage = "export_reference"
        if result.returncode != 0:
            raise RuntimeError(f"yasp-toolkit kernelgen export exited {result.returncode}\n{output}")
        reference = artifacts / PT2_FILE
        if not reference.is_file():
            raise RuntimeError(f"yasp-toolkit kernelgen export exited 0 without writing {PT2_FILE}\n{output}")
        logger.info("wrote %s", reference, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {}

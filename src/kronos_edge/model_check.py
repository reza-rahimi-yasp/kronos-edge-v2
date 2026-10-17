"""Run the model checks (the eager graph) as a subprocess of this interpreter, on the GPU.

The subprocess writes its report to a file in a temporary directory, so
nothing the user's model prints on stdout can touch the protocol; its stdout
and stderr are inherited. The files the checks produce for the user go to a
new run directory under the artifact root, which later nodes find in
``run_path``. The child is this interpreter with this package's source
directory leading ``PYTHONPATH``, so it runs the same code from any working
directory; it is the environment the preflight verified, less the Yasp
credentials, which the user's model never sees.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from kronos_edge import artifacts as artifact_files
from kronos_edge.kernelgen_check import without_credentials
from kronos_edge.log import get_logger
from kronos_edge.messages import BENCHMARK, EAGER, MODEL_CHECK as CUSTOMER_MESSAGES, ONNX, ONNXSIM, TENSORRT
from kronos_edge.state import State, fail

logger = get_logger(__name__)

DEVICE = "cuda"
REPORT_FILE = "report.json"
SOURCE_DIR = Path(__file__).resolve().parents[1]


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


def process_environment() -> dict[str, str]:
    """This process's environment without the Yasp credentials, with the package's source directory first on PYTHONPATH."""
    env = without_credentials(dict(os.environ))
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE_DIR), inherited]) if inherited else str(SOURCE_DIR)
    return env


def build_command(
    model_path, npz_path, report_dir, artifacts_dir, precision: str, threshold: float, device: str, log_level: str,
) -> list[str]:
    return [
        sys.executable, "-m", "kronos_edge.eager_check", str(Path(model_path).resolve()), str(Path(npz_path).resolve()),
        "--report", str(Path(report_dir).resolve() / REPORT_FILE), "--artifacts", str(Path(artifacts_dir).resolve()),
        "--precision", precision, "--threshold", str(threshold), "--device", device, "--log-level", log_level,
    ]


def run_process(
    model_path, npz_path, report_dir, artifacts_dir, precision: str, threshold: float, device: str, log_level: str,
) -> int:
    """Run the eager check as a subprocess with stdout and stderr inherited; return its exit code."""
    return subprocess.run(
        build_command(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level),
        env=process_environment(),
    ).returncode


def _read_report(report_dir, code: int) -> Report:
    path = Path(report_dir) / REPORT_FILE
    if not path.exists():
        raise RuntimeError(f"the model check exited {code} without writing a report")
    report = Report.model_validate_json(path.read_text())
    if report.error is not None and report.error not in EAGER | ONNX | ONNXSIM | TENSORRT | BENCHMARK:
        raise ValueError(f"report names unknown failure stage {report.error!r}")
    expected = 0 if report.error is None else 1
    if code != expected:
        reported = "no error" if report.error is None else f"error at {report.error}"
        raise RuntimeError(f"the model check exited {code} but reported {reported}")
    if report.error is None and report.benchmark is None:
        raise ValueError("report claims success but carries no benchmark")
    return report


def run_model_check(state: State) -> State:
    stage = "model_check_run"
    try:
        artifacts = artifact_files.create_run_directory(state["artifacts_path"])
        config = state["config"]
        logger.info(
            "running the model checks in a subprocess on %s at %s, artifacts in %s", DEVICE, config.precision, artifacts,
            extra={"stage": stage},
        )
        with tempfile.TemporaryDirectory() as report_dir:
            code = run_process(
                state["model_path"], state["npz_path"], report_dir, artifacts, config.precision,
                config.threshold, DEVICE, state.get("log_level", "INFO"),
            )
            report = _read_report(report_dir, code)
    except Exception as exc:
        return fail(logger, stage, exc)
    if report.error is not None:
        logger.info("the model check reported failure at stage %s", report.error, extra={"stage": stage})
        return {"run_path": str(artifacts), "error": report.error, "error_detail": report.error_detail}
    return {"run_path": str(artifacts), "benchmark": report.benchmark.model_dump()}

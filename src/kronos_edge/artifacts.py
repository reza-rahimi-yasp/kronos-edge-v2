"""The files the checks write for the user, and the run directory each run writes them to.

No heavy imports: the workflow loads this to create the run directory before
the model check starts. Every run writes into a new directory under the
artifact root, named for the UTC time the run started, so a run never shows a
stale file next to a fresh one and never touches an earlier run's directory or
anything else in the root. Every file a node writes, in the model-check
subprocess, by the toolkit, or in the workflow, is listed here.
"""

import itertools
from datetime import datetime, timezone
from pathlib import Path

EXPORTED_FILE = "model.onnx"
SIMPLIFIED_FILE = "model_simplified.onnx"
ENGINE_FILE = "model.engine"
BENCHMARK_FILE = "benchmark.json"
PT2_FILE = "model.pt2"  # the reference `yasp-toolkit kernelgen export` writes from model.py
KERNELGEN_GUIDANCE_FILE = "kernelgen_guidance.txt"  # the guidance we send with kernelgen compile
KERNEL_FILE = "kernel.py"  # the optimized kernel kernelgen returns
KERNELGEN_REPORT_FILE = "kernelgen_report.json"  # the kernelgen job's report
MANAGED_FILES = (
    EXPORTED_FILE, SIMPLIFIED_FILE, ENGINE_FILE, BENCHMARK_FILE,
    PT2_FILE, KERNELGEN_GUIDANCE_FILE, KERNEL_FILE, KERNELGEN_REPORT_FILE,
)

# Weights above the protobuf limit go to a companion file with this suffix.
EXTERNAL_DATA_SUFFIX = ".data"


# A run directory's name: the UTC time the run started, ISO 8601 basic format.
RUN_NAME_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_run_directory(root) -> Path:
    """Create the root if missing and a new, empty directory in it for this run; return the run directory.

    The name is the run's UTC start time. It is created exclusively: if the name
    is taken, by a run that started in the same second or by anything else, the
    next free ``-2``, ``-3``, ... suffix is used, so an existing path is never reused.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().astimezone(timezone.utc).strftime(RUN_NAME_FORMAT)
    for n in itertools.count(1):
        path = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path

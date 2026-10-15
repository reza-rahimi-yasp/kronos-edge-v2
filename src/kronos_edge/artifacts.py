"""The files the checks write for the user, and the host-side cleanup between runs.

No heavy imports: the host loads this to clear the artifact directory before
the container starts. Every file a node writes, in the container or on the
host, is listed here, so a rerun in
the same directory never shows a stale file next to a fresh one. Files an
earlier version wrote and this one no longer does are cleared too, so an
artifact directory carried over from that version does not keep them. Files
the user put there themselves are never touched.
"""

from pathlib import Path

EXPORTED_FILE = "model.onnx"
SIMPLIFIED_FILE = "model_simplified.onnx"
ENGINE_FILE = "model.engine"
BENCHMARK_FILE = "benchmark.json"
PT2_FILE = "model.pt2"  # the reference `yasp-toolkit kernelgen export` writes in the export container
KERNEL_FILE = "kernel.py"  # the optimized kernel kernelgen returns
KERNELGEN_REPORT_FILE = "kernelgen_report.json"  # the kernelgen job's report
MANAGED_FILES = (
    EXPORTED_FILE, SIMPLIFIED_FILE, ENGINE_FILE, BENCHMARK_FILE,
    PT2_FILE, KERNEL_FILE, KERNELGEN_REPORT_FILE,
)
# Written by earlier versions, never by this one: cleared, not produced or read.
RETIRED_FILES = ("kernelgen_eval.json",)

# Weights above the protobuf limit go to a companion file with this suffix.
EXTERNAL_DATA_SUFFIX = ".data"


def clear(directory) -> None:
    """Remove the managed and retired files and their external-data companions; leave everything else."""
    for name in (*MANAGED_FILES, *RETIRED_FILES):
        for path in (Path(directory) / name, Path(directory) / f"{name}{EXTERNAL_DATA_SUFFIX}"):
            path.unlink(missing_ok=True)

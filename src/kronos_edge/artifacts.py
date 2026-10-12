"""The files the checks write for the user, and the host-side cleanup between runs.

No heavy imports: the host loads this to clear the artifact directory before
the container starts. Every file a node writes is listed here, so a rerun in
the same directory never shows a stale file next to a fresh one. Files the
user put there themselves are never touched.
"""

from pathlib import Path

EXPORTED_FILE = "model.onnx"
SIMPLIFIED_FILE = "model_simplified.onnx"
ENGINE_FILE = "model.engine"
MANAGED_FILES = (EXPORTED_FILE, SIMPLIFIED_FILE, ENGINE_FILE)

# Weights above the protobuf limit go to a companion file with this suffix.
EXTERNAL_DATA_SUFFIX = ".data"


def clear(directory) -> None:
    """Remove the managed files and their external-data companions; leave everything else."""
    for name in MANAGED_FILES:
        for path in (Path(directory) / name, Path(directory) / f"{name}{EXTERNAL_DATA_SUFFIX}"):
            path.unlink(missing_ok=True)

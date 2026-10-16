#!/bin/bash
# Run a command in the dev image with the experiment's standard mounts.
#   container.sh <container-name-suffix> <gpu|nogpu> <command...>
# Mounts (read-only unless noted): the artifact directory with model.pt2 and
# kernel.py at /artifact, the repository's src at /src (for kronos_edge), the
# experiment at /exp, and the output directory (writable) at /out. No
# environment from the host, no secrets, no docker socket. Runs as the
# calling user so outputs are not root-owned. With GPU_LOCK set, a GPU
# container runs under flock(1) on that file for its whole lifetime, so two
# experiments sharing the GPU never measure concurrently.
set -euo pipefail
: "${ARTIFACT_DIR:?set ARTIFACT_DIR to the directory holding model.pt2 and kernel.py}"
: "${OUT_DIR:?set OUT_DIR to a writable output directory}"
IMAGE=${DEV_IMAGE:-gaia-trt-plugins-dev:local}
EXP=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$EXP/../.." && pwd)
suffix=$1; gpu=$2; shift 2
gpu_args=()
lock=()
if [ "$gpu" = gpu ]; then
    gpu_args=(--gpus all)
    if [ -n "${GPU_LOCK:-}" ]; then lock=(flock "$GPU_LOCK"); fi
fi
mkdir -p "$OUT_DIR"
exec "${lock[@]}" docker run --rm --name "gaia-trt-$suffix-$$" "${gpu_args[@]}" \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_EXTENSIONS_DIR=/out/torch_ext -e PYTHONPATH=/src:/exp/exporter \
    -v "$ARTIFACT_DIR":/artifact:ro -v "$REPO/src":/src:ro -v "$EXP":/exp:ro -v "$OUT_DIR":/out \
    -w /out "$IMAGE" "$@"

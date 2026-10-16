#!/bin/bash
# Reproduces the experiment stage by stage. Everything that touches the model
# stack runs in containers; the host only runs docker, git and stdlib Python.
#
#   ARTIFACT_DIR=<dir with model.pt2 and kernel.py> OUT_DIR=<writable dir> scripts/run_all.sh <stage|all>
#
# Stages, in order:
#   images     build the dev image (from kronos-edge:rtx5090-cuda130, which must exist; official TRT headers)
#              and compile plugins + runner; checks the committed device code against ARTIFACT_DIR/kernel.py
#   extract    weights, invstd, input.bin from model.pt2 (GPU: invstd is computed where kernel.py computes it)
#   reference  unmodified kernel.py output, default and no-TF32 settings, determinism check
#   engines    hybrid_{default,notf32}.engine and pure_{default,notf32}.engine
#   runtime    build the Python-free runtime image from build/
#   validate   every engine vs the Gaia output through the TensorRT Python runtime, then the C++ runner's outputs
#   bench      C++ runner (hybrid vs pure, non-blocking and legacy stream) and the Python timer, both settings
# GPU work is sequential; containers are named gaia-trt-<stage>-<pid> and removed on exit.
set -euo pipefail
: "${ARTIFACT_DIR:?}" "${OUT_DIR:?}"
EXP=$(cd "$(dirname "$0")/.." && pwd)
RUNTIME_IMAGE=${RUNTIME_IMAGE:-gaia-trt-plugins-runtime:local}
DEV_IMAGE=${DEV_IMAGE:-gaia-trt-plugins-dev:local}
THRESHOLD=${THRESHOLD:-0.99}
SHAPE=4,3,1088,1920
C="$EXP/scripts/container.sh"
mkdir -p "$OUT_DIR/telemetry"

telemetry() {  # host-side nvidia-smi snapshot, read-only
    nvidia-smi --query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu \
        --format=csv >> "$OUT_DIR/telemetry/$1.csv"
}

runtime_run() {  # C++ runner in the Python-free image; /data is OUT_DIR
    docker run --rm --name "gaia-trt-runtime-$$" --gpus all -u "$(id -u):$(id -g)" -v "$OUT_DIR":/data "$RUNTIME_IMAGE" "$@"
}

stage_images() {
    docker build -f "$EXP/docker/dev/Dockerfile" -t "$DEV_IMAGE" "$EXP/docker/dev"
    docker run --rm --name "gaia-trt-make-$$" -u "$(id -u):$(id -g)" -v "$EXP":/exp -w /exp "$DEV_IMAGE" make
    python3 -I "$EXP/exporter/extract_gaia_device_code.py" "$ARTIFACT_DIR/kernel.py" "$OUT_DIR/device_code_check" >/dev/null
    cmp "$OUT_DIR/device_code_check/gaia_device_code.cuh" "$EXP/plugins/gaia_device_code.cuh"  # the committed header is this kernel.py's code
}

stage_extract() {
    "$C" extract gpu python /exp/exporter/extract_weights.py /artifact/model.pt2 /exp/exporter/model_signature.json /out
}

stage_reference() {
    for s in default notf32; do
        "$C" reference-$s gpu python /exp/exporter/gaia_reference.py /artifact/model.pt2 /artifact/kernel.py /out $s 5
    done
}

stage_engines() {
    for s in default notf32; do
        "$C" hybrid-$s gpu python /exp/exporter/build_hybrid_engine.py /out/weights.npz /exp/exporter/model_signature.json /exp/build/libgaia_trt_plugins.so /out $s
    done
    "$C" pure gpu python /exp/exporter/build_pure_engine.py /artifact/model.pt2 /out
}

stage_runtime() {
    docker build -f "$EXP/docker/runtime/Dockerfile" -t "$RUNTIME_IMAGE" "$EXP/build"
}

stage_validate() {
    for s in default notf32; do
        "$C" validate-$s gpu python /exp/exporter/validate.py /out /exp/build/libgaia_trt_plugins.so "$THRESHOLD" $s engine:/out/hybrid_$s.engine engine:/out/pure_$s.engine
        for e in hybrid pure; do
            runtime_run run --engine /data/${e}_$s.engine --input /data/input.bin --shape $SHAPE --output /data/cpp_output_${e}_$s.bin --repeat 5 --meta /data/cpp_run_${e}_$s.json
        done
        "$C" validate-cpp-$s gpu python /exp/exporter/validate.py /out /exp/build/libgaia_trt_plugins.so "$THRESHOLD" $s output:/out/cpp_output_hybrid_$s.bin output:/out/cpp_output_pure_$s.bin
    done
}

stage_bench() {
    for s in default notf32; do
        for stream in nonblocking legacy; do
            telemetry "cpp_bench_${s}_${stream}_before"
            runtime_run bench --engine-a /data/hybrid_$s.engine --engine-b /data/pure_$s.engine --input /data/input.bin --shape $SHAPE \
                --json /data/benchmark_cpp_${s}_${stream}.json --output-a /data/cpp_bench_output_hybrid_${s}_${stream}.bin \
                --output-b /data/cpp_bench_output_pure_${s}_${stream}.bin --stream $stream
            telemetry "cpp_bench_${s}_${stream}_after"
            "$C" validate-bench-$s-$stream gpu python /exp/exporter/validate.py /out /exp/build/libgaia_trt_plugins.so "$THRESHOLD" $s \
                output:/out/cpp_bench_output_hybrid_${s}_${stream}.bin output:/out/cpp_bench_output_pure_${s}_${stream}.bin
        done
        telemetry "python_bench_${s}_before"
        "$C" bench-python-$s gpu python /exp/exporter/benchmark_python.py /artifact/model.pt2 /artifact/kernel.py /out /exp/build/libgaia_trt_plugins.so "$THRESHOLD" $s
        telemetry "python_bench_${s}_after"
    done
}

case "${1:-}" in
    images|extract|reference|engines|runtime|validate|bench) "stage_$1" ;;
    all) for st in images extract reference engines runtime validate bench; do echo "### stage $st"; "stage_$st"; done ;;
    *) echo "usage: $0 images|extract|reference|engines|runtime|validate|bench|all" >&2; exit 2 ;;
esac

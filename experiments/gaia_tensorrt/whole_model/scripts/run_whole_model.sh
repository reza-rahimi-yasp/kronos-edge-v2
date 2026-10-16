#!/bin/bash
# Reproduces the whole-model experiment stage by stage. Everything that touches
# the model stack runs in containers; the host runs docker, flock and stdlib
# Python. Every GPU stage holds GPU_LOCK (flock) for its whole lifetime, so
# nothing else measures on the GPU meanwhile.
#
#   ARTIFACT_DIR=<dir with model.pt2 and kernel.py> BASELINE_DIR=<hybrid experiment output> OUT_DIR=<writable dir> \
#       whole_model/scripts/run_whole_model.sh <stage|all>
#
# Stages, in order:
#   images     tag the dev image gaia-trt-whole-model-dev:local from ../docker/dev/Dockerfile and compile
#              the hybrid library (make in ..) and this library + runner_whole (make here)
#   baseline   copy the reused inputs from BASELINE_DIR: weights.npz, input.bin, meta.json, the Gaia
#              reference outputs and the pure TensorRT engines (hashes recorded in reused_baseline.sha256)
#   engines    whole_default.engine and whole_notf32.engine (GPU; the builder does not run a one-plugin
#              network, so the plugin's cuDNN plans are selected per execution context at run time)
#   runtime    build the interpreter-free runtime image gaia-trt-whole-model-runtime:local
#   validate   each engine vs the Gaia output through the TensorRT Python runtime (5 repeats), then the
#              C++ runner_whole in the runtime image (5 repeats), its outputs validated the same way; the
#              plugin's plan log (the cuDNN engine of every convolution) goes to *_whole_<setting>.plans.log
#   bench      runner_whole: whole vs pure (non-blocking and legacy stream), both settings; then the
#              Python timer: unmodified kernel.py vs whole engine vs pure engine, both settings
set -euo pipefail
: "${ARTIFACT_DIR:?}" "${OUT_DIR:?}"
WHOLE=$(cd "$(dirname "$0")/.." && pwd)
EXP=$(cd "$WHOLE/.." && pwd)
GPU_LOCK=${GPU_LOCK:-$OUT_DIR/../gpu.lock}
DEV_IMAGE=${DEV_IMAGE:-gaia-trt-whole-model-dev:local}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-gaia-trt-whole-model-runtime:local}
THRESHOLD=${THRESHOLD:-0.99}
SHAPE=4,3,1088,1920
export DEV_IMAGE
C="$EXP/scripts/container.sh"
mkdir -p "$OUT_DIR/telemetry"

gpu() {  # run a command with the GPU lock held for its whole lifetime
    flock "$GPU_LOCK" "$@"
}

telemetry() {  # host-side nvidia-smi snapshot, read-only
    nvidia-smi --query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu \
        --format=csv >> "$OUT_DIR/telemetry/$1.csv"
}

runtime_run() {  # runner_whole in the interpreter-free image; /data is OUT_DIR
    gpu docker run --rm --name "gaia-trt-whole-model-runtime-$$" --gpus all -u "$(id -u):$(id -g)" -e GAIA_TRT_WHOLE_LOG_PLANS=1 -v "$OUT_DIR":/data "$RUNTIME_IMAGE" "$@"
}

stage_images() {
    docker build -f "$EXP/docker/dev/Dockerfile" -t "$DEV_IMAGE" "$EXP/docker/dev"
    docker run --rm --name "gaia-trt-whole-model-make-$$" -u "$(id -u):$(id -g)" -v "$EXP":/exp -w /exp "$DEV_IMAGE" bash -c "make && cd whole_model && make"
}

stage_baseline() {
    : "${BASELINE_DIR:?set BASELINE_DIR to the hybrid experiment output directory}"
    for f in weights.npz input.bin meta.json gaia_output_default.bin gaia_output_notf32.bin gaia_reference_default.json gaia_reference_notf32.json \
             pure_default.engine pure_notf32.engine pure_default.build.json pure_notf32.build.json; do
        cp -p "$BASELINE_DIR/$f" "$OUT_DIR/$f"
    done
    (cd "$OUT_DIR" && sha256sum weights.npz input.bin meta.json gaia_output_default.bin gaia_output_notf32.bin pure_default.engine pure_notf32.engine > reused_baseline.sha256)
}

stage_engines() {
    for s in default notf32; do
        gpu "$C" whole-model-engine-$s gpu python /exp/whole_model/exporter/build_whole_engine.py /out/weights.npz /exp/exporter/model_signature.json \
            /exp/whole_model/build/libgaia_trt_whole_model.so /out $s
    done
}

stage_runtime() {
    cp -p "$EXP/build/libgaia_trt_plugins.so" "$WHOLE/build/"
    docker build -f "$WHOLE/docker/runtime/Dockerfile" -t "$RUNTIME_IMAGE" "$WHOLE/build"
    docker run --rm --entrypoint /bin/sh "$RUNTIME_IMAGE" -c 'for p in python python3 pip pip3 nvcc; do command -v $p && exit 1; done; ls /usr/lib/x86_64-linux-gnu | grep -c nvrtc && exit 1; echo "no interpreter, no compiler, no NVRTC"; dpkg -l | grep -E "libnvinfer11|libcudnn9|libcublas-13"'
}

stage_validate() {
    for s in default notf32; do
        gpu env GAIA_TRT_WHOLE_LOG_PLANS=1 "$C" whole-model-validate-$s gpu python /exp/exporter/validate.py /out /exp/whole_model/build/libgaia_trt_whole_model.so "$THRESHOLD" $s \
            engine:/out/whole_$s.engine 2> >(tee "$OUT_DIR/python_validate_whole_$s.plans.log" >&2)
        runtime_run run --engine /data/whole_$s.engine --input /data/input.bin --shape $SHAPE --output /data/cpp_output_whole_$s.bin --repeat 5 --meta /data/cpp_run_whole_$s.json \
            2> "$OUT_DIR/cpp_run_whole_$s.plans.log"
        gpu "$C" whole-model-validate-cpp-$s gpu python /exp/exporter/validate.py /out /exp/whole_model/build/libgaia_trt_whole_model.so "$THRESHOLD" $s output:/out/cpp_output_whole_$s.bin
    done
}

stage_bench() {
    for s in default notf32; do
        for stream in nonblocking legacy; do
            telemetry "cpp_bench_${s}_${stream}_before"
            runtime_run bench --engine-a /data/whole_$s.engine --engine-b /data/pure_$s.engine --input /data/input.bin --shape $SHAPE \
                --json /data/benchmark_cpp_whole_vs_pure_${s}_${stream}.json --output-a /data/cpp_bench_output_whole_${s}_${stream}.bin \
                --output-b /data/cpp_bench_output_pure_${s}_${stream}.bin --stream $stream 2> "$OUT_DIR/cpp_bench_${s}_${stream}.stderr.log"
            telemetry "cpp_bench_${s}_${stream}_after"
            gpu "$C" whole-model-validate-bench-$s-$stream gpu python /exp/exporter/validate.py /out /exp/whole_model/build/libgaia_trt_whole_model.so "$THRESHOLD" $s \
                output:/out/cpp_bench_output_whole_${s}_${stream}.bin output:/out/cpp_bench_output_pure_${s}_${stream}.bin
        done
        telemetry "python_bench_${s}_before"
        gpu "$C" whole-model-bench-python-$s gpu python /exp/exporter/benchmark_python.py /artifact/model.pt2 /artifact/kernel.py /out \
            /exp/whole_model/build/libgaia_trt_whole_model.so "$THRESHOLD" $s whole,pure
        telemetry "python_bench_${s}_after"
    done
}

case "${1:-}" in
    images|baseline|engines|runtime|validate|bench) "stage_$1" ;;
    all) for st in images baseline engines runtime validate bench; do echo "### stage $st"; "stage_$st"; done ;;
    *) echo "usage: $0 images|baseline|engines|runtime|validate|bench|all" >&2; exit 2 ;;
esac

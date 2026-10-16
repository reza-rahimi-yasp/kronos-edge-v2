# Minimal runtime image: CUDA base, the TensorRT 11.4.0.106 runtime library,
# the C++ runner and the plugin library. No Python, no torch, no compiler.
# Build context is the experiment's build/ directory (runner and
# libgaia_trt_plugins.so from `make` in the dev image). Engines, inputs and
# outputs are mounted at run time.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04
ARG TRT_DEB_VERSION=11.4.0.106-1+cuda13.4
RUN apt-get update && apt-get install -y --no-install-recommends libnvinfer11=${TRT_DEB_VERSION} \
    && rm -rf /var/lib/apt/lists/*
COPY runner libgaia_trt_plugins.so /opt/gaia_trt/
ENTRYPOINT ["/opt/gaia_trt/runner"]

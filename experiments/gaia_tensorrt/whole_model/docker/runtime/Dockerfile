# Interpreter-free runtime image for the whole-model engine: CUDA base, the
# TensorRT 11.4.0.106 runtime, and the cuDNN and cuBLAS builds torch
# 2.14.1+cu130 ships (nvidia-cudnn-cu13 9.24.0.43, nvidia-cublas 13.1.1.3),
# pinned from NVIDIA's apt repository so the plugin selects the same cuDNN
# engines at run time as at build time and as kernel.py's PyTorch did. No
# interpreter, no torch, no compiler, no NVRTC: cuDNN's runtime-compiled
# engines are not loadable here and the plugin never selects one (its plan
# log records the engines used).
# Build context: a directory holding runner_whole, libgaia_trt_whole_model.so
# and libgaia_trt_plugins.so (the runner registers both libraries).
FROM nvidia/cuda:13.0.3-base-ubuntu24.04
ARG TRT_DEB_VERSION=11.4.0.106-1+cuda13.4
ARG CUDNN_DEB_VERSION=9.24.0.43-1
ARG CUBLAS_DEB_VERSION=13.1.1.3-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer11=${TRT_DEB_VERSION} \
        libcudnn9-cuda-13=${CUDNN_DEB_VERSION} \
        libcublas-13-0=${CUBLAS_DEB_VERSION} \
    && rm -rf /var/lib/apt/lists/*
COPY runner_whole libgaia_trt_whole_model.so libgaia_trt_plugins.so /opt/gaia_trt/
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64
ENTRYPOINT ["/opt/gaia_trt/runner_whole"]

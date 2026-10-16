# Build-time image for the Gaia-to-TensorRT experiment. Derived from the
# check's own environment image, kronos-edge:rtx5090-cuda130, which is built
# from the tracked docker/rtx5090-cuda130/Dockerfile and carries the registry
# entry's versions (torch 2.14.1+cu130, the TensorRT 11.4.0.106 Python wheels,
# onnx, onnxsim, numpy). On top of it, pinned from NVIDIA's CUDA apt
# repository the base image already configures:
#   - the CUDA 13.0 compiler and runtime headers (nvcc builds the plugins and
#     kernel.py's inline extension), g++, make, ninja and the Python headers
#     that torch.utils.cpp_extension.load_inline needs;
#   - the official TensorRT C++ headers and link stubs of the same 11.4.0.106
#     build the wheels implement.
# Prerequisite: build the base first, from the repository root:
#   docker build -f docker/rtx5090-cuda130/Dockerfile -t kronos-edge:rtx5090-cuda130 .
# Nothing in this image reaches the runtime image.
ARG BASE=kronos-edge:rtx5090-cuda130
FROM ${BASE}
ARG TRT_DEB_VERSION=11.4.0.106-1+cuda13.4
ARG CUDA_APT_VERSION=13-0
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_APT_VERSION} cuda-cudart-dev-${CUDA_APT_VERSION} \
        g++ make ninja-build python3-dev \
        libnvinfer-headers-dev=${TRT_DEB_VERSION} \
        libnvinfer-headers-plugin-dev=${TRT_DEB_VERSION} \
        libnvinfer11=${TRT_DEB_VERSION} \
        libnvinfer-dev=${TRT_DEB_VERSION} \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

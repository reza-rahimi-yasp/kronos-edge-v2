# Build-time image for the Gaia-to-TensorRT experiment. Derived from the
# task-owned diagnostic image (torch 2.14.1+cu130, TensorRT 11.4.0.106 Python
# wheels, CUDA 13.0 nvcc, g++), adding only NVIDIA's official TensorRT C++
# headers and link stubs for the same 11.4.0.106 build, so the plugins and the
# runner compile against the exact API the wheels implement. Nothing in this
# image reaches the runtime image.
ARG BASE=reftiming-kronos-kernel:local
FROM ${BASE}
ARG TRT_DEB_VERSION=11.4.0.106-1+cuda13.4
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer-headers-dev=${TRT_DEB_VERSION} \
        libnvinfer-headers-plugin-dev=${TRT_DEB_VERSION} \
        libnvinfer11=${TRT_DEB_VERSION} \
        libnvinfer-dev=${TRT_DEB_VERSION} \
        cmake make \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

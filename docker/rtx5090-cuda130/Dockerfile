# One environment, one image: NVIDIA GeForce RTX 5090, driver 580.95.05,
# CUDA 13.0, TensorRT 11.4.0, torch 2.14.1. Must match the entry of the same
# name in src/kronos_edge/environments.yaml.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG TORCH_VERSION=2.14.1
ARG TENSORRT_VERSION=11.4.0.106

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir "tensorrt-cu13==${TENSORRT_VERSION}" langgraph numpy pyyaml pydantic

COPY src /opt/kronos-edge/src
ENV PYTHONPATH=/opt/kronos-edge/src

LABEL ai.yasp.kronos-edge.gpu="NVIDIA GeForce RTX 5090" \
      ai.yasp.kronos-edge.driver="580.95.05" \
      ai.yasp.kronos-edge.cuda="13.0" \
      ai.yasp.kronos-edge.tensorrt="11.4.0" \
      ai.yasp.kronos-edge.torch="2.14.1"

CMD ["python"]

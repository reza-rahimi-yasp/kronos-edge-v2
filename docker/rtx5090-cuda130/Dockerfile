# The recipe for one supported environment: NVIDIA GeForce RTX 5090, driver
# 580.95.05, CUDA 13.0, TensorRT 11.4.0, torch 2.14.1. Must match the entry of
# the same name in src/kronos_edge/environments.yaml (a test compares the
# labels below with it). The check never builds or runs this image: it is one
# way to get the environment, and the check's preflight verifies whichever
# environment it is started in, this image or a native install.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG TORCH_VERSION=2.14.1
ARG TENSORRT_VERSION=11.4.0.106

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir "tensorrt-cu13==${TENSORRT_VERSION}" langgraph numpy "onnx==1.23.2" "onnxscript==0.7.2" "onnxsim==0.8.1" pyyaml pydantic

# Packages model.py imports are the user's to pip-install into this environment; resolve them against the
# same index as torch so packages with CUDA builds (torchvision, for one) match the image's CUDA.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130

COPY src /opt/kronos-edge/src
ENV PYTHONPATH=/opt/kronos-edge/src

LABEL ai.yasp.kronos-edge.gpu="NVIDIA GeForce RTX 5090" \
      ai.yasp.kronos-edge.driver="580.95.05" \
      ai.yasp.kronos-edge.cuda="13.0" \
      ai.yasp.kronos-edge.tensorrt="11.4.0" \
      ai.yasp.kronos-edge.torch="2.14.1"

CMD ["python"]

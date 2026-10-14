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
    && pip install --no-cache-dir "tensorrt-cu13==${TENSORRT_VERSION}" langgraph numpy "onnx==1.23.2" "onnxscript==0.7.2" "onnxsim==0.8.1" pyyaml pydantic

# yasp's exporter (`yasp.compile.common.exporter`, what `yasp-toolkit kernelgen export` runs) produces the kernelgen
# reference. It comes from the yasp package index, which needs a scoped registry key to read: the build takes it as
# a BuildKit secret, a netrc file mounted for this step only (`docker build --secret id=yasp-netrc,src=<file>`),
# so the key is never a build argument, an environment variable, a URL, or a layer. The package is installed under
# a constraints file of everything already here, so torch's CUDA build and the registry versions cannot move.
# Nothing else of the toolkit enters the image, and the exporter needs no key at run time.
ARG YASP_COMPILE_COMMON_VERSION=1.10.0
ARG YASP_INDEX_URL=https://nexus.yasp.ai/repository/pypi-hosted/simple/
RUN --mount=type=secret,id=yasp-netrc,target=/root/.netrc,required=true \
    pip freeze > /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt "yasp-compile-common==${YASP_COMPILE_COMMON_VERSION}" \
        --extra-index-url "${YASP_INDEX_URL}" \
    && rm /tmp/constraints.txt

# The user's `dependencies` are pip-installed at run time; resolve them against the same index as torch so
# packages with CUDA builds (torchvision, for one) match the image's CUDA.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130

COPY src /opt/kronos-edge/src
ENV PYTHONPATH=/opt/kronos-edge/src

LABEL ai.yasp.kronos-edge.gpu="NVIDIA GeForce RTX 5090" \
      ai.yasp.kronos-edge.driver="580.95.05" \
      ai.yasp.kronos-edge.cuda="13.0" \
      ai.yasp.kronos-edge.tensorrt="11.4.0" \
      ai.yasp.kronos-edge.torch="2.14.1"

CMD ["python"]

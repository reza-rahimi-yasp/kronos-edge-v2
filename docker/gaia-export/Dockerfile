# The kernelgen reference export: `yasp-toolkit kernelgen export`, installed the way the yasp install guide
# documents it (https://compile.yasp.ai/help/install/): a CPU-only torch==2.9.1 first, then yasp-agent from the
# yasp package index. Nothing else: no model stack, no GPU, no repository code. The host runs the documented
# export command in a container from this image on the user's model.py, so the host never runs user code, and
# a CPU-only torch traces the reference on the CPU. The yasp index is read with a registry key, which the build
# takes as a BuildKit secret (a netrc file mounted for that one step), never as an argument, a variable, a URL,
# or a layer. yasp-agent is installed under a constraints file of what is already here, so torch stays put.
FROM python:3.12-slim

ARG TORCH_VERSION=2.9.1
ARG YASP_AGENT_VERSION=0.5.2
ARG YASP_INDEX_URL=https://nexus.yasp.ai/repository/pypi-hosted/simple/

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

RUN --mount=type=secret,id=yasp-netrc,target=/root/.netrc,required=true \
    pip freeze > /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt "yasp-agent[gaia]==${YASP_AGENT_VERSION}" \
        --extra-index-url "${YASP_INDEX_URL}" \
    && rm /tmp/constraints.txt

# The user's `dependencies` are pip-installed at run time; resolve them against the CPU torch index so a
# package with torch builds (torchvision, for one) matches this torch.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

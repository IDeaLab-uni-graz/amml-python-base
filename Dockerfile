# Flag upon which it is decided to use either the CPU or the CUDA version
# Allowed values: cuda, cpu
ARG hardware="cpu"

# In general, we are using:
# - python, pytorch and CUDA version specified below
# - apt-based userland for the docker image

ARG PYTORCH_VERSION="2.15.0"
ARG PYTHON_VERSION="3.12"
ARG CUDA_VERSION="13.2"

# Note that we are not specifying PyTorch in base_requirements.txt to follow the official installation instructions as closely as possible.

# Both hardware variants use the official Python image + official PyTorch wheels (multi-arch: amd64 + arm64).
# The Debian release is pinned explicitly, because the CUDA toolkit apt repository (see toolkit-cuda) is per Debian release.
FROM python:${PYTHON_VERSION}-slim-trixie AS base-cuda

ARG PYTORCH_VERSION
ARG CUDA_VERSION

# Mirrors what the pytorch/pytorch and nvidia/cuda images set, so the NVIDIA runtime exposes the GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root
WORKDIR /opt/build

# The CUDA wheels bundle the CUDA runtime/cuDNN/NCCL as nvidia-* pip packages; only the host driver is needed.
# CUDA_VERSION "13.2" -> wheel index "cu132"
RUN pip install --upgrade pip \
    && pip install --no-cache-dir  \
      "torch==${PYTORCH_VERSION}"  \
      torchvision  \
      --index-url "https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d .)"

FROM python:${PYTHON_VERSION}-slim-trixie AS base-cpu

ARG PYTORCH_VERSION
ARG PYTHON_VERSION
ARG CUDA_VERSION

USER root
WORKDIR /opt/build

RUN pip install --upgrade pip \
    && pip install --no-cache-dir  \
      "torch==${PYTORCH_VERSION}"  \
      torchvision  \
      --index-url https://download.pytorch.org/whl/cpu

FROM base-${hardware} AS slim
LABEL authors="sceptri"

ARG PYTORCH_VERSION
ARG PYTHON_VERSION
ARG CUDA_VERSION

USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      libblas-dev \
      liblapack-dev \
      libfreetype6-dev \
      libpng-dev \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build

COPY --chown=root:root slim_requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r slim_requirements.txt

RUN git config --system --add safe.directory /opt/project   # writes /etc/gitconfig

# Entrypoint improves QoL when passing mulitple commands etc.
COPY --chown=root:root entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]

FROM slim AS toolkit-cpu

# CUDA compiler + headers for building CUDA extensions (torch.utils.cpp_extension, flash-attn, ...).
# Runtime libraries already come with the PyTorch wheels; this only adds what compiling needs.
FROM slim AS toolkit-cuda

ARG CUDA_VERSION
ARG TARGETARCH

# NVIDIA calls its arm64 server repo "sbsa"
RUN if [ "${TARGETARCH}" = "arm64" ]; then REPO_ARCH=sbsa; else REPO_ARCH=x86_64; fi \
    && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && CUDA_PKG=$(echo ${CUDA_VERSION} | tr . -) \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cuda-minimal-build-${CUDA_PKG} \
      libcublas-dev-${CUDA_PKG} \
      libcusparse-dev-${CUDA_PKG} \
      libcusolver-dev-${CUDA_PKG} \
      libcurand-dev-${CUDA_PKG} \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV PATH=${CUDA_HOME}/bin:${PATH}

FROM toolkit-${hardware} AS full

COPY --chown=root:root full_requirements.txt .
RUN pip install --no-cache-dir -r full_requirements.txt

# Entrypoint improves QoL when passing mulitple commands etc.
COPY --chown=root:root entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]

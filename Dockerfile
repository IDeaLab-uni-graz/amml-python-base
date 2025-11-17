# Flag upon which it is decided to use either the CPU or the CUDA version
# Allowed values: cuda, cpu, rocm
ARG hardware="cpu"

# In general, we are using:
# - python 3.12
# - pytorch 2.9.0
# - apt-based userland for the docker image

ARG PYTORCH_VERSION="2.9.1"
ARG PYTHON_VERSION="3.12"

# Note that we are not specifying PyTorch in base_requirements.txt to follow the official installation instructions as closely as possible.

# To err on the side of caution, I used the devel tag
# https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda12.8-cudnn9-devel AS base-cuda
# WARNING: I could not test the CUDA version as I have an AMD GPU locally

FROM python:${PYTHON_VERSION}-slim AS base-cpu

ARG PYTORCH_VERSION
ARG PYTHON_VERSION

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
ARG ROCM_VERSION

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

FROM slim AS full

COPY --chown=root:root full_requirements.txt .
RUN pip install --no-cache-dir -r full_requirements.txt

# Entrypoint improves QoL when passing mulitple commands etc.
COPY --chown=root:root entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]
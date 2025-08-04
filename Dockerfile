# Flag upon which it is decided to use either the CPU, CUDA or ROCm version
# Allowed values: cuda, cpu, rocm
ARG hardware="cpu"

# In general, we are using:
# - python 3.12
# - pytorch 2.7.1
# - apt-based userland for the docker image

ARG PYTORCH_VERSION="2.7.1"
ARG PYTHON_VERSION="3.12"
ARG ROCM_VERSION="6.3"
ARG HSA_GFX_VERSION="11"

FROM ubuntu:24.04  AS base-rocm

ARG PYTORCH_VERSION
ARG PYTHON_VERSION
ARG ROCM_VERSION
ARG HSA_GFX_VERSION

USER root
WORKDIR /opt/build

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git software-properties-common build-essential \
    curl gnupg wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir --parents --mode=0755 /etc/apt/keyrings && \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
      gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" \
      | tee /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
      | tee /etc/apt/preferences.d/rocm-pin-600

# To allow for installation of a specific Python version
RUN add-apt-repository ppa:deadsnakes/ppa

ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "python${PYTHON_VERSION}"  \
    "python${PYTHON_VERSION}-venv" \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s $(which python${PYTHON_VERSION}) /usr/bin/python

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py; \
    python get-pip.py; \
    rm get-pip.py

RUN python -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rocm \
    && rm -rf /var/lib/apt/lists/*

# Add root to the render and video groups for ROCm to work
ENV LOGNAME=root
RUN groupadd render && \
    usermod -a -G render,video $LOGNAME

# See: https://github.com/harakas/amd_igpu_yolo_v8?tab=readme-ov-file
ENV HSA_ENABLE_SDMA="0"
ENV HSA_OVERRIDE_GFX_VERSION="${HSA_GFX_VERSION}.0.0"

RUN pip install --upgrade pip \
    && pip install --no-cache-dir  \
      "torch==${PYTORCH_VERSION}"  \
      torchvision  \
      --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}

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

COPY --chown=root:root requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT ["python"]
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

FROM slim AS full

COPY --chown=root:root full_requirements.txt .
RUN pip install --no-cache-dir -r full_requirements.txt

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
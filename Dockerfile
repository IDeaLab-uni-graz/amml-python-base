# Flag upon which it is decided to use either the CPU, CUDA or ROCm version
# Allowed values: cuda, cpu, rocm
ARG hardware="cpu"

# In general, we are using:
# - python 3.12
# - pytorch 2.7.1
# - apt-based userland for the docker image

FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1 AS base-rocm

# To err on the side of caution, I used the devel tag
# https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel AS base-cuda
# WARNING: I could not test the CUDA version as I have an AMD GPU locally

FROM python:3.12-slim AS base-cpu

USER root
WORKDIR /opt/build

RUN pip install --upgrade pip \
    && pip install --no-cache-dir  \
      'torch==2.7.1'  \
      torchvision  \
      torchaudio  \
      --index-url https://download.pytorch.org/whl/cpu

FROM base-${hardware} AS slim
LABEL authors="sceptri"

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
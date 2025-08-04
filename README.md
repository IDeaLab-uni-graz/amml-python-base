# AMML-Python-Base

Docker base images for the needs of the Martin Holler's team at the IDea_Lab, University of Graz.

## Images

We provide several images centered around Python, and more specifically, PyTorch. For AMD64, we supply both the CPU and CUDA versions of Pytorch,
whereas only the CPU version is provided for ARM64.

> [!WARNING]
> The ROCm support is **currently unsupported**.
> 
> In the `Dockerfile` (and accompanying `docker-compose.yaml`) there is a ROCm version (both in slim- and full variants).
> It is based on `ubuntu:24.04` and we manually add Python, Pytorch and ROCm, because we tried to decrease the size of the Docker image 
> (the size of the compressed official image `rocm/pytorch` is around 25 GBs). Unfortunately, it has very little effect on the resulting filesize.
> 
> Moreover, the status at the moment is that we override the HSA GFX version 
> (defaults to 11.0.0, the only one build in the runner) to use a more general, less optimized
> ROCm HIP drivers. This is done because AMD iGPUs are not officially supported with ROCm (it seems the official Docker image had some issues with this).
>
> Lastly, I tried to benchmark pytorch on CPU and using the ROCm acceleration (see `tests/test_cpu_vs_gpu.py`)
> ```
> Initialisation of tensors
> CPU_time =  0.2050325870513916
> GPU_time =  2.8424596786499023
> Matrix multiplication
> CPU_time =  2.375737428665161
> GPU_time =  0.17871427536010742
> Broadcasting
> CPU_time =  0.02744007110595703
> GPU_time =  0.04163670539855957
> Outer product of tensors
> CPU_time =  0.03771543502807617
> GPU_time =  0.013091325759887695
> ```
> In other words, it seems that the ROCm support solely for the AMD iGPUs is not worth it.

Lastly, for each hardware version a *full* and a *slim* version, depending on the number of python libraries installed is provided. 

## Running Locally

For ease-of-use, we also include a _docker compose_ file to ease the building and running process. 
As an example, one can use the following command, which build and runs the _ROCm_ (i.e., AMD GPU) _full_ image.
```bash
docker-compose run --build --rm amml-python-base-rocm
```

### Testing the Github Workflow Locally

I had some success with [`act`](https://nektosact.com/) to simulate GitHub worklow, as typically ran on the GitHub runners, locally. 
Unfortunately, it seems to get stuck on the push to DockerHub phase for me.

For `act` to work, one needs to create a secrets file, e.g., `.secrets`, with `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` and **`GITHUB_TOKEN`** 
(personal access token for GitHub, which is usually supplied to the runner automatically). Then, one can use
```bash
act --secret-file .secrets
```

## Template Guide

### Structure


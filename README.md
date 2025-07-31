# AMML-Python-Base

Docker base images for the needs of the Martin Holler's team at the IDea_Lab, University of Graz.

## Images

We provide several images centered around Python, and more specifically, PyTorch. Regarding the GPU architecture
CUDA, ROCm and CPU are all supported, but only for the AMD64 CPU architecture!. 

> [!WARNING]
> As mentioned above, the ARM64 CPU architecture is not currently supported (the CUDA and ROCm builds are failing).
> However, it is planned in near future.

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


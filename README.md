# AMML Python Base

Docker base images for the needs of the Martin Holler's team (AMML) at the IDea_Lab, University of Graz. Public builds can be found on DockerHub under the name

```
sceptri/amml-python-base-<hardware>-<slim version?>
```
for example `sceptri/amml-python-base-cpu` or `sceptri/amml-python-base-cuda-slim`.

## Recommended Usage

Below are documented main use-cases and their respective recommended workflows:

- "One needs to try out something very simple in Python without the need for version control, but they want to use a controlled reproducible environment." - Use the DockerHub built version of the images from this repo directly via `docker run`, e.g.,
  ```shell
  docker run -it sceptri/amml-python-base-cpu-slim:latest bash
  ```
- "One works on a (simple) Python project which is not based around machine learning or requires a very specific structure of the project." - Use the simple [AMML Python template](https://github.com/IDeaLab-uni-graz/amml-python-template), which provides `docker-compose.yaml` to ease with the running/building of the project-specific Docker image, e.g.[^1],
  ```shell
  docker-compose run --build --rm amml-project-cpu
  ```
- "One works on a (standard) machine learning Python project utilizing the common workflows and apps, e.g., MLFlow, of the AMML team." - Use the advanced [AMML Python ML template](https://github.com/IDeaLab-uni-graz/amml-python-ml-template), which provides all the bells and whistles one might need throughout the project development along with a basic structure and example usage. With Docker compose, one can just run, e.g.[^1],
  ```shell
  docker-compose run --build --rm amml-project-cuda-slim
  ```
  
[^1]: Change the `amml-project-cpu` for the corresponding project name, see `docker-compose.yaml` in the project directory.

## Images

We provide several images centered around Python, and more specifically, PyTorch. For AMD64 architecture we supply both the CPU and CUDA versions of Pytorch, whereas only the CPU version is provided for ARM64 architecture.

> Let us note that ROCm support was deprecated, and is currently archived in the `rocm-support` branch.

For each hardware version a *full* and a *slim* variants, depending on the number of python libraries installed, are defined. The *slim* version is intended only for small experiments and/or Python environments unrelated to machine learning. 

> Note that PyTorch is installed separately outside of `requirements.txt` to follow the official installation instructions more closely. See the `Dockerfile` for more details.

If you have any suggestion what to include/exclude from the Python module requirements in the *full*/*slim* versions, please open a Pull Request or an Issue.

**Provided Images:**
- [`sceptri/amml-python-base-cpu`](https://hub.docker.com/r/sceptri/amml-python-base-cpu) - default image to use for local development
- [`sceptri/amml-python-base-cuda`](https://hub.docker.com/r/sceptri/amml-python-base-cuda) - default image for servers and/or workstations with NVIDIA GPUs
- [`sceptri/amml-python-base-cpu-slim`](https://hub.docker.com/r/sceptri/amml-python-base-cpu-slim) - lightweight version of the CPU image for local development, in general use only if appropriate for projects not following the AMML standards/best practices
- [`sceptri/amml-python-base-cuda-slim`](https://hub.docker.com/r/sceptri/amml-python-base-cuda-slim) - lightweight version of the NVIDIA GPU image, in general use only if appropriate for projects not following the AMML standards/best practices


> [!TIP]
> If you need a newer version of one of the base images, you might need to "force pull" the image from DockerHub to replace the locally cached version
> ```shell
> docker pull sceptri/amml-python-base-cpu:latest
> ```
> This also applies when using one of the templates (where the image is import via the `Dockerfile` keyword `FROM`)!

## Development
### Building and Running Locally

For ease-of-use, we also include a _docker compose_ file to ease the building and running process. As an example, one can use the following command, which build and runs the _CPU full_ image.

```shell
docker-compose run --build --rm amml-python-base-cpu
```
### Testing the GitHub Workflow Locally

I had some success with [`act`](https://nektosact.com/) to simulate GitHub workflow, as typically ran on the GitHub runners, locally. 
Unfortunately, it seems to get stuck on the push to DockerHub phase for me.

For `act` to work, one needs to create a secrets file, e.g., `.secrets`, with `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` and **`GITHUB_TOKEN`** 
(personal access token for GitHub, which is usually supplied to the runner automatically). Then, one can use

```shell
act --secret-file .secrets
```

### Tests

Currently, during the CI/CD pipeline the images are only built and deployed to DockerHub. In future, it might be appropriate run certain tests, which should then live in the `tests` folder.

Note that `tests` directory is also the place to store scripts for manual testing (e.g., benchmarking, checking GPU acceleration support in PyTorch).
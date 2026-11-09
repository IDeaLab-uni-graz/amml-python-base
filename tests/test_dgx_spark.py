"""Smoke tests for the packages shipped in the AMML Python base images, aimed at NVIDIA DGX Spark.

Run inside the image, e.g.,

    docker run --rm --gpus all --shm-size=16g -v $(pwd)/tests:/opt/build/tests \
        devidealab/amml-python-base-cuda:latest pytest -v tests/test_dgx_spark.py

GPU tests are skipped when no GPU is visible and *full*-only packages are skipped in the *slim* images,
so the same file also runs on the CPU images. On a DGX Spark, nothing should be skipped with the CUDA *full* image.
"""

import shutil

import pytest
import numpy as np
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU visible")


# ---------- Scientific stack (slim) ----------

def test_numpy():
    a = np.random.default_rng(0).standard_normal((200, 200))
    assert np.allclose(a @ np.linalg.inv(a), np.eye(200), atol=1e-8)


def test_scipy():
    from scipy import linalg, optimize, sparse
    from scipy.sparse.linalg import spsolve

    a = np.random.default_rng(0).standard_normal((50, 50)) + 50 * np.eye(50)
    b = np.ones(50)
    assert np.allclose(linalg.solve(a, b), spsolve(sparse.csr_matrix(a), b))
    assert np.isclose(optimize.minimize_scalar(lambda x: (x - 2) ** 2).x, 2)


def test_pandas_with_parquet(tmp_path):
    import pandas as pd

    pytest.importorskip("fastparquet")
    df = pd.DataFrame({"group": ["a", "b", "a"], "value": [1.0, 2.0, 3.0]})
    assert df.groupby("group")["value"].sum().to_dict() == {"a": 4.0, "b": 2.0}

    df.to_parquet(tmp_path / "df.parquet", engine="fastparquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "df.parquet", engine="fastparquet"), df)


def test_matplotlib_and_pillow(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(tmp_path / "plot.png")
    plt.close(fig)
    assert Image.open(tmp_path / "plot.png").size[0] > 0


def test_amml_utils_import():
    import amml_utils  # noqa: F401


# ---------- PyTorch on the GPU ----------

@requires_cuda
def test_gpu_kernels_match_device():
    """Prebuilt kernels must exist for this GPU (sm_121 on Spark; sm_120 kernels are binary compatible)."""
    major, minor = torch.cuda.get_device_capability()
    archs = torch.cuda.get_arch_list()
    print(torch.cuda.get_device_name(), (major, minor), archs)
    assert any(a == f"sm_{major}{m}" for a in archs for m in range(minor + 1))


@requires_cuda
def test_matmul_bf16_and_autograd():
    x = torch.randn(2048, 2048, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).sum()
    y.backward()
    assert torch.isfinite(x.grad).all()
    # cuBLAS result matches the CPU reference
    a = torch.randn(256, 256)
    assert torch.allclose((a.cuda() @ a.cuda()).cpu(), a @ a, atol=1e-3, rtol=1e-3)


@requires_cuda
def test_scaled_dot_product_attention():
    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert out.shape == q.shape and torch.isfinite(out).all()


@requires_cuda
def test_unified_memory_reporting():
    """On Spark nvidia-smi shows no GPU memory, but torch must still report it."""
    free, total = torch.cuda.mem_get_info()
    assert 0 < free <= total
    x = torch.empty(1024**3, dtype=torch.uint8, device="cuda")  # 1 GiB
    assert torch.cuda.memory_allocated() >= x.numel()


@requires_cuda
def test_torch_compile():
    """Exercises Triton code generation for this GPU architecture."""
    fn = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2 + 1)
    x = torch.randn(1024, device="cuda")
    assert torch.allclose(fn(x), torch.nn.functional.gelu(x) * 2 + 1, atol=1e-5)


@requires_cuda
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc only in the CUDA full image")
def test_cuda_extension_compiles():
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        "add_one",
        cpp_sources="torch::Tensor add_one(torch::Tensor x);",
        cuda_sources="""
        #include <ATen/cuda/CUDAContext.h>
        __global__ void kernel(float* p, int n) { int i = threadIdx.x; if (i < n) p[i] += 1; }
        torch::Tensor add_one(torch::Tensor x) {
            kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), x.numel());
            return x;
        }""",
        functions=["add_one"],
    )
    assert torch.equal(ext.add_one(torch.zeros(256, device="cuda")).cpu(), torch.ones(256))


@requires_cuda
def test_torchvision_on_gpu():
    from torchvision.models import resnet18
    from torchvision.ops import nms

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=torch.float, device="cuda")
    assert nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5).tolist() == [0, 2]

    model = resnet18(weights=None).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert model(torch.randn(4, 3, 224, 224, device="cuda")).shape == (4, 1000)


def test_dataloader_workers():
    """Worker processes share tensors via /dev/shm, which fails with Docker's 64 MB default."""
    data = torch.utils.data.TensorDataset(torch.randn(256, 3, 128, 128))
    loader = torch.utils.data.DataLoader(data, batch_size=64, num_workers=4)
    assert sum(batch[0].shape[0] for batch in loader) == 256


# ---------- ML tooling (full) ----------

def test_lightning_fit():
    pl = pytest.importorskip("pytorch_lightning")

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(16, 1)

        def training_step(self, batch, _):
            x, y = batch
            return torch.nn.functional.mse_loss(self.layer(x), y)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    data = torch.utils.data.TensorDataset(torch.randn(256, 16), torch.randn(256, 1))
    trainer = pl.Trainer(max_epochs=2, accelerator="auto", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(Model(), torch.utils.data.DataLoader(data, batch_size=32, num_workers=2))
    assert trainer.state.finished
    assert trainer.strategy.root_device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_optuna():
    optuna = pytest.importorskip("optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    study.optimize(lambda trial: (trial.suggest_float("x", -10, 10) - 2) ** 2, n_trials=30)
    assert abs(study.best_params["x"] - 2) < 2


def test_mlflow(tmp_path):
    mlflow = pytest.importorskip("mlflow")

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    with mlflow.start_run() as run:
        mlflow.log_param("lr", 0.1)
        mlflow.log_metric("loss", 0.5)
    stored = mlflow.get_run(run.info.run_id).data
    assert stored.params["lr"] == "0.1" and stored.metrics["loss"] == 0.5


def test_ray_gpu_task():
    ray = pytest.importorskip("ray")

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        gpus = 1 if torch.cuda.is_available() else 0

        @ray.remote(num_gpus=gpus)
        def square_sum(n):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return (torch.arange(n, device=device, dtype=torch.float) ** 2).sum().item()

        assert ray.get(square_sum.remote(10)) == 285.0
    finally:
        ray.shutdown()


def test_misc_full_packages():
    box = pytest.importorskip("box")
    tabulate = pytest.importorskip("tabulate")
    pytest.importorskip("IPython")
    pytest.importorskip("jupyterlab")
    pytest.importorskip("nc_py_api")

    config = box.Box({"model": {"lr": 0.1}})
    assert config.model.lr == 0.1
    assert "lr" in tabulate.tabulate([["lr", 0.1]])

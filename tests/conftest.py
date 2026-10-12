import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

import torch

from kronos_edge import benchmark_check, container, eager_check, env_check, tensorrt_check

ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 5090",
    "driver": "580.95.05",
    "cuda": "13.0",
    "tensorrt": "11.4.0",
    "torch": "2.14.1",
}

# config.yaml: the environment fields plus the check settings.
CONFIG = {**ENVIRONMENT, "precision": "fp32", "metric": "cosine", "threshold": 0.99}

REGISTRY = {
    "environments": [
        {
            "name": "rtx5090-cuda130",
            "image": "kronos-edge:rtx5090-cuda130",
            "dockerfile": "docker/rtx5090-cuda130/Dockerfile",
            **ENVIRONMENT,
        }
    ]
}

GOOD_MODEL = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture(autouse=True)
def reset_package_logger():
    """configure() binds a handler to the stderr of the test that called main(); drop it afterwards."""
    yield
    package_logger = logging.getLogger("kronos_edge")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """The default artifact directory is relative to the working directory; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def host_matches_registry(monkeypatch):
    """Tests never touch docker or nvidia-smi; the host looks like the registry entry."""
    monkeypatch.setattr(env_check, "image_exists", lambda image: True)
    monkeypatch.setattr(env_check, "host_gpu", lambda: (ENVIRONMENT["gpu"], ENVIRONMENT["driver"]))


@pytest.fixture(autouse=True)
def eager_check_runs_in_process(monkeypatch):
    """Stand in for `docker run`: run the eager sub-graph here, on CPU, write its report, return its exit code."""

    def fake_run_container(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        state = eager_check.run(model_path, npz_path, artifacts_dir, precision=precision, threshold=threshold, device="cpu")
        (Path(report_dir) / container.REPORT_FILE).write_text(eager_check.report(state))
        return 0 if state.get("error") is None else 1

    monkeypatch.setattr(container, "run_container", fake_run_container)


FAKE_ENGINE = b"fake serialized engine"


@pytest.fixture(autouse=True)
def tensorrt_stub(monkeypatch):
    """Stand in for TensorRT, which only exists in the image: hand back fixed bytes instead of building.

    Returns the real worker for the one test that exercises its import failure.
    """
    real_build = tensorrt_check.build
    monkeypatch.setattr(tensorrt_check, "build", lambda onnx_path, inputs: FAKE_ENGINE)
    return real_build


class FakeEngineRunner:
    """Stand in for a TensorRT engine: runs the eager model, optionally perturbing its outputs."""

    def __init__(self, state, perturb=lambda outputs: outputs):
        self.model, self.inputs, self.perturb = state["model"], state["user_inputs"], perturb
        self.outputs = None

    def run(self):
        with torch.no_grad():
            self.outputs = self.perturb(benchmark_check.flatten_outputs(self.model(*self.inputs)))


@pytest.fixture(autouse=True)
def engine_runner_stub(monkeypatch):
    """The engine only runs on the GPU in the image; here it is the eager model behind the runner interface."""
    monkeypatch.setattr(benchmark_check, "load_engine", FakeEngineRunner)


@pytest.fixture
def env_files(tmp_path):
    """A valid config.yaml and a registry that matches it."""
    config = write_yaml(tmp_path / "config.yaml", dict(CONFIG))
    environments = write_yaml(tmp_path / "environments.yaml", REGISTRY)
    return config, environments


@pytest.fixture
def artifacts(tmp_path):
    """An existing artifact directory, as the container sees the one the host created and mounted."""
    path = tmp_path / "art"
    path.mkdir()
    return path


@pytest.fixture
def good_model(tmp_path):
    model = tmp_path / "good_model.py"
    model.write_text(GOOD_MODEL)
    npz = tmp_path / "good_input.npz"
    np.savez(npz, np.zeros((1, 3, 8, 8), dtype=np.float32))
    return model, npz

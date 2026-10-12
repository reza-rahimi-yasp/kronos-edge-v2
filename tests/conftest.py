import logging

import numpy as np
import pytest
import yaml

from kronos_edge import env_check

ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 5090",
    "driver": "580.95.05",
    "cuda": "13.0",
    "tensorrt": "11.4.0",
    "torch": "2.14.1",
}

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
def host_matches_registry(monkeypatch):
    """Tests never touch docker or nvidia-smi; the host looks like the registry entry."""
    monkeypatch.setattr(env_check, "image_exists", lambda image: True)
    monkeypatch.setattr(env_check, "host_gpu", lambda: (ENVIRONMENT["gpu"], ENVIRONMENT["driver"]))


@pytest.fixture
def env_files(tmp_path):
    """A valid config.yaml and a registry that matches it."""
    config = write_yaml(tmp_path / "config.yaml", dict(ENVIRONMENT))
    environments = write_yaml(tmp_path / "environments.yaml", REGISTRY)
    return config, environments


@pytest.fixture
def good_model(tmp_path):
    model = tmp_path / "good_model.py"
    model.write_text(GOOD_MODEL)
    npz = tmp_path / "good_input.npz"
    np.savez(npz, np.zeros((1, 3, 8, 8), dtype=np.float32))
    return model, npz

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

import torch

from kronos_edge import benchmark_check, eager_check, kernelgen_check, model_check, preflight, tensorrt_check

ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 5090",
    "driver": "580.95.05",
    "cuda": "13.0",
    "tensorrt": "11.4.0",
    "torch": "2.14.1",
}

# config.yaml: the environment fields plus the check settings.
CONFIG = {**ENVIRONMENT, "precision": "fp32", "metric": "cosine", "threshold": 0.99, "eval_routing": "reza-fractal"}

REGISTRY = {
    "environments": [
        {
            "name": "rtx5090-cuda130",
            "dockerfile": "docker/rtx5090-cuda130/Dockerfile",
            **ENVIRONMENT,
        }
    ]
}

# What the preflight probe reports from a Python environment that is the registry entry: the versions as the
# packages print them (torch with its CUDA build label, TensorRT with its build number) and the GPU torch sees.
PROBE = {"torch": "2.14.1+cu130", "cuda": "13.0", "tensorrt": "11.4.0.106", "gpu": "NVIDIA GeForce RTX 5090"}

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


def run_directory(root):
    """The one run directory a single run created under the artifact root."""
    runs = [p for p in Path(root).iterdir() if p.is_dir()]
    assert len(runs) == 1, f"expected one run directory under {root}, found {runs}"
    return runs[0]


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
def environment_matches_registry(monkeypatch):
    """Tests never probe the real environment: this one looks like the registry entry, with the toolkit on the path."""
    monkeypatch.setattr(preflight, "run_probe", lambda: dict(PROBE))
    monkeypatch.setattr(preflight, "toolkit_on_path", lambda: "/usr/local/bin/yasp-toolkit")


@pytest.fixture(autouse=True)
def eager_check_runs_in_process(monkeypatch):
    """Stand in for the model-check subprocess: run the eager sub-graph here, on CPU, mark its prerequisites and write
    its report as eager_check.main does, return its exit code."""

    def fake_run_process(model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level):
        state = eager_check.run(
            model_path, npz_path, artifacts_dir, precision=precision, threshold=threshold, device="cpu",
            on_prerequisites=(Path(report_dir) / model_check.PREREQUISITES_FILE).touch,
        )
        (Path(report_dir) / model_check.REPORT_FILE).write_text(eager_check.report(state))
        return 0 if state.get("error") is None else 1

    monkeypatch.setattr(model_check, "run_process", fake_run_process)


FAKE_ENGINE = b"fake serialized engine"


@pytest.fixture(autouse=True)
def tensorrt_stub(monkeypatch):
    """Stand in for TensorRT, which the test environment lacks: hand back fixed bytes instead of building.

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
        with torch.no_grad():  # like a real engine, never mutate the bound inputs
            self.outputs = self.perturb(benchmark_check.flatten_outputs(self.model(*[t.clone() for t in self.inputs])))


@pytest.fixture(autouse=True)
def engine_runner_stub(monkeypatch):
    """The engine only runs on the GPU; here it is the eager model behind the runner interface."""
    monkeypatch.setattr(benchmark_check, "load_engine", FakeEngineRunner)


FAKE_KERNEL = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"

# The report is passed to `eval inspect` and shown; the check never reads its fields.
FAKE_REPORT = json.dumps({"optimized": True})

# What `eval inspect --file <report>` prints for it: the report rendered for a reader.
FAKE_INSPECT = "kernelgen_report.json\n  optimized: True\n  speedup: 2.000\n  correctness: passed\n"

FAKE_TASK_LINE = "  task 00000000-0000-0000-0000-000000000000 (completed)\n"

FAKE_REFERENCE = b"reference written by yasp-toolkit kernelgen export"

# What `kernelgen export` prints: the user's model may print here too, so it never reaches the check's stdout.
FAKE_EXPORT_OUTPUT = "  wrote      model.pt2\n"


def evaluator_listing(*rows):
    """What `eval workers` prints: a table with one (gpu, routing) row each, or the line for no match."""
    if not rows:
        return "No evaluators match.\n"
    lines = [
        f"{len(rows)} evaluator{'' if len(rows) == 1 else 's'}",
        "┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓",
        "┃ GPU Name ┃ Routing ┃ Version ┃ Tags ┃",
        "┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩",
        *(f"│ {gpu} │ {routing} │ 1.10.0  │ host: {gpu.split()[-1].lower()} │" for gpu, routing in rows),
        "└──────────┴─────────┴─────────┴──────┘",
    ]
    return "\n".join(lines) + "\n"


def fake_toolkit(args, timeout=None):
    """What a successful yasp-toolkit run leaves behind: the reference, the kernel and its report, the rendered report, or a listing.

    `eval workers` lists an evaluator advertising the config's GPU under the routing key asked for,
    so by default a run finds the persistent worker it needs registered.
    """

    def option(name):
        return Path(args[args.index(name) + 1])

    stdout = FAKE_TASK_LINE
    if args[:2] == ["kernelgen", "export"]:
        option("--output").write_bytes(FAKE_REFERENCE)
        stdout = FAKE_EXPORT_OUTPUT
    elif args[:2] == ["kernelgen", "compile"]:
        option("--output").write_text(FAKE_KERNEL)
        option("--report").write_text(FAKE_REPORT)
    elif args[:2] == ["eval", "inspect"]:
        assert option("--file").is_file(), "inspect is asked for a report that is not there"
        stdout = FAKE_INSPECT
    elif args[:2] == ["eval", "workers"]:
        stdout = evaluator_listing((CONFIG["gpu"], str(option("--tag")).removeprefix("routing=")))
    else:
        raise AssertionError(f"unexpected toolkit command {args}")
    return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=stdout, stderr="")


@pytest.fixture(autouse=True)
def toolkit_stub(monkeypatch):
    """Stand in for yasp-toolkit, which needs a Yasp API key and the hosted service: write what a successful run would."""
    monkeypatch.setattr(kernelgen_check, "run_toolkit", fake_toolkit)


@pytest.fixture
def env_files(tmp_path):
    """A valid config.yaml and a registry that matches it."""
    config = write_yaml(tmp_path / "config.yaml", dict(CONFIG))
    environments = write_yaml(tmp_path / "environments.yaml", REGISTRY)
    return config, environments


@pytest.fixture
def artifacts(tmp_path):
    """An existing run directory, as the model-check subprocess sees the one the workflow created."""
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

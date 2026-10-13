import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

import torch

from kronos_edge import benchmark_check, container, eager_check, env_check, evaluator_check, kernelgen_check, tensorrt_check

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

    def fake_run_container(image, model_path, npz_path, report_dir, artifacts_dir, precision, threshold, device, log_level, dependencies):
        state = eager_check.run(
            model_path, npz_path, artifacts_dir, precision=precision, threshold=threshold, device="cpu",
            dependencies=dependencies,
        )
        (Path(report_dir) / container.REPORT_FILE).write_text(eager_check.report(state))
        return 0 if state.get("error") is None else 1

    monkeypatch.setattr(container, "run_container", fake_run_container)


class _Installed(list):
    """The lists pip was asked to install; `real` is the worker itself for the tests that exercise it."""

    real = None


@pytest.fixture(autouse=True)
def pip_stub(monkeypatch):
    """Tests never shell out: record what the container would pip install instead of installing it."""
    installed = _Installed()
    installed.real = eager_check.install
    monkeypatch.setattr(eager_check, "install", lambda packages: installed.append(list(packages)))
    return installed


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
        with torch.no_grad():  # like a real engine, never mutate the bound inputs
            self.outputs = self.perturb(benchmark_check.flatten_outputs(self.model(*[t.clone() for t in self.inputs])))


@pytest.fixture(autouse=True)
def engine_runner_stub(monkeypatch):
    """The engine only runs on the GPU in the image; here it is the eager model behind the runner interface."""
    monkeypatch.setattr(benchmark_check, "load_engine", FakeEngineRunner)


# One evaluator result, the shape of an entry of `eval compare`'s results.json (and of a kernelgen
# report's `eval_result`): correctness flags plus the measured speedup.
PASSING_RESULT = {
    "stage": "benchmark",
    "correctness_checks": {
        "all_checks_passed": True, "all_shapes_match": True, "all_dtypes_match": True,
        "all_output_values_match": True, "results": [{"all_checks_passed": True}],
    },
    "reference_benchmark": {"runtimes_ms": [2.0, 2.1], "median_runtime_ms": 2.05},
    "candidate_benchmark": {"runtimes_ms": [1.0, 1.05], "median_runtime_ms": 1.025},
    "speedup": 2.0,
}

FAKE_KERNEL = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"

FAKE_TASK_LINE = "  task 00000000-0000-0000-0000-000000000000 (completed)\n"


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


def fake_toolkit(args, result=PASSING_RESULT):
    """What a successful yasp-toolkit run leaves behind: the kernel and report, or the results.json.

    `eval workers` lists an evaluator advertising the config's GPU under the routing key asked for,
    so by default a run finds its evaluator registered and starts none.
    """

    def option(name):
        return Path(args[args.index(name) + 1])

    stdout = FAKE_TASK_LINE
    if args[:2] == ["kernelgen", "compile"]:
        option("--output").write_text(FAKE_KERNEL)
        option("--report").write_text(json.dumps({"gpu_name": args[args.index("--gpu-name") + 1], "eval_result": result}))
    elif args[:2] == ["eval", "compare"]:
        option("--output").write_text(json.dumps({"opt_1": result}))
    elif args[:2] == ["eval", "workers"]:
        stdout = evaluator_listing((CONFIG["gpu"], str(option("--tag")).removeprefix("routing=")))
    else:
        raise AssertionError(f"unexpected toolkit command {args}")
    return subprocess.CompletedProcess(["yasp-toolkit", *args], 0, stdout=stdout, stderr="")


@pytest.fixture(autouse=True)
def toolkit_stub(monkeypatch):
    """Stand in for yasp-toolkit, which needs a Yasp API key and the hosted service: write what a successful run would."""
    monkeypatch.setattr(kernelgen_check, "run_toolkit", fake_toolkit)


class FakeEvaluator:
    """Stand in for a `yasp-eval register` process: alive until `exit_code` is set, with canned output."""

    def __init__(self, args, code=None, output=""):
        self.args, self.code, self.text = args, code, output

    def exit_code(self):
        return self.code

    def output(self):
        return self.text


class EvaluatorRecorder:
    """The evaluators the check started and stopped; `exits` is the (code, output) a started one reports."""

    def __init__(self):
        self.started, self.stopped, self.exits = [], [], None

    def start(self, args):
        evaluator = FakeEvaluator(args, *(self.exits or ()))
        self.started.append(evaluator)
        return evaluator

    def stop(self, evaluator):
        self.stopped.append(evaluator)


@pytest.fixture(autouse=True)
def evaluators(monkeypatch):
    """Tests never start a real evaluator: record what the check would start and stop, and poll without sleeping."""
    recorder = EvaluatorRecorder()
    monkeypatch.setattr(evaluator_check, "start_evaluator", recorder.start)
    monkeypatch.setattr(evaluator_check, "stop_evaluator", recorder.stop)
    monkeypatch.setattr(evaluator_check, "POLL_INTERVAL_S", 0)
    monkeypatch.setattr(evaluator_check, "READY_TIMEOUT_S", 1)
    return recorder


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

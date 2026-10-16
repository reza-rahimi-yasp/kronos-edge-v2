"""The C++ runner's timing defaults are the product's benchmark protocol, read from both sources as text."""

import re
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]


def _product_constant(name: str) -> int:
    source = (REPO / "src" / "kronos_edge" / "benchmark_check.py").read_text()
    return int(re.search(rf"^{name} = (\d+)$", source, re.M).group(1))


def _runner_default(option: str) -> int:
    source = (EXPERIMENT / "runner" / "runner.cpp").read_text()
    return int(re.search(rf'a\.get\("{option}", "(\d+)"\)', source).group(1))


def test_runner_defaults_match_kronos_timer():
    assert _runner_default("warmup") == _product_constant("WARMUP") == 20
    assert _runner_default("runs") == _product_constant("RUNS") == 200
    assert _runner_default("trials") == 3


def test_tensorrt_versions_agree_across_images():
    product = (REPO / "src" / "kronos_edge" / "environments.yaml").read_text()
    dev = (EXPERIMENT / "docker" / "dev" / "Dockerfile").read_text()
    runtime = (EXPERIMENT / "docker" / "runtime" / "Dockerfile").read_text()
    version = re.search(r"TRT_DEB_VERSION=([\d.]+)-", dev).group(1)
    assert re.search(r"TRT_DEB_VERSION=([\d.]+)-", runtime).group(1) == version
    assert f'tensorrt: "{".".join(version.split(".")[:3])}"' in product


def test_runtime_image_installs_no_python():
    runtime = (EXPERIMENT / "docker" / "runtime" / "Dockerfile").read_text()
    assert "python" not in runtime.lower().replace("no python", "")
    assert "pip" not in runtime.lower()


def test_gpu_containers_run_under_the_shared_lock_when_gpu_lock_is_set():
    container = (EXPERIMENT / "scripts" / "container.sh").read_text()
    run_all = (EXPERIMENT / "scripts" / "run_all.sh").read_text()
    assert 'flock "$GPU_LOCK"' in container and "GPU_LOCK" in run_all
    assert "gaia-trt-layouts-dev:local" in run_all and "gaia-trt-layouts-runtime:local" in run_all  # the baseline's images are never overwritten
    assert "gaia-trt-plugins-dev:local" not in run_all and "gaia-trt-plugins-runtime:local" not in run_all

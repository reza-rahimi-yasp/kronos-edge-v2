"""The benchmark check: latency distributions for eager and the engine, and output agreement by the configured metric.

The engine runs only on the GPU in the image; `benchmark_check.load_engine` is stubbed with a runner that
executes the eager model, so timing, comparison, threshold, and the report are exercised for real on CPU.
"""

import json
import math

import numpy as np
import pytest
import torch

from conftest import CONFIG, FakeEngineRunner, write_yaml
from kronos_edge import benchmark_check, eager_check
from kronos_edge.workflow import run_check
from test_eager_check import CONV_RELU, TWO_INPUTS, X, XY, error_records, write_model, write_npz

TWO_OUTPUTS = TWO_INPUTS.replace("return x + y", "return x + y, x * y")

# A dense, non-zero output so a perturbation's effect on the cosine is predictable.
SCALE = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2


def get_inputs():
    return [torch.randn(64, 64)]
"""
ONES = np.ones((64, 64), dtype=np.float32)


def run_with(tmp_path, env_files, source=CONV_RELU, arrays=(X,), **settings):
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, **settings})
    return run_check(config, write_model(tmp_path, source), write_npz(tmp_path, list(arrays)), environments_path=environments)


def test_cosine_similarity():
    a, b = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])
    assert math.isclose(benchmark_check.cosine(a, b), 1.0, rel_tol=1e-6)
    assert math.isclose(benchmark_check.cosine(a, -b), -1.0, rel_tol=1e-6)
    assert benchmark_check.cosine(torch.zeros(3), torch.zeros(3)) == 1.0
    assert benchmark_check.cosine(torch.zeros(3), b) == 0.0
    assert math.isclose(benchmark_check.cosine(torch.ones(2, 2, dtype=torch.int64), torch.ones(4).half()), 1.0, rel_tol=1e-6)


@pytest.mark.parametrize(
    "outputs, flat",
    [
        pytest.param(torch.ones(2), [torch.ones(2)], id="tensor"),
        pytest.param((torch.ones(1), torch.zeros(1)), [torch.ones(1), torch.zeros(1)], id="tuple"),
        pytest.param([torch.ones(1), (torch.zeros(1), torch.ones(3))], [torch.ones(1), torch.zeros(1), torch.ones(3)], id="nested"),
        pytest.param({"b": torch.ones(2), "a": torch.zeros(1)}, [torch.ones(2), torch.zeros(1)], id="dict_insertion_order"),
    ],
)
def test_flatten_outputs(outputs, flat):
    got = benchmark_check.flatten_outputs(outputs)
    assert [tuple(t.shape) for t in got] == [tuple(t.shape) for t in flat]


def test_flatten_rejects_non_tensor_leaves():
    with pytest.raises(TypeError, match="str"):
        benchmark_check.flatten_outputs(("fine", torch.ones(1)))


DICT_OUTPUT = SCALE.replace("return x * 2", 'return {"positive": x * 2, "negative": -x}')
INPLACE_INPUT = SCALE.replace("return x * 2", "x.add_(1)\n        return x * x")


def test_dict_outputs_are_compared_in_exporter_order(tmp_path, env_files):
    state = run_with(tmp_path, env_files, source=DICT_OUTPUT, arrays=[ONES])

    assert state["outcome"] == "success"
    assert state["benchmark"]["values"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_inplace_input_mutation_does_not_skew_the_comparison(tmp_path, monkeypatch, env_files):
    """Every node runs the model on copies, so the engine and the reference see the user's original values."""
    seen = {}

    def recording(state):
        seen["engine_input"] = state["user_inputs"][0].clone()
        return FakeEngineRunner(state)

    monkeypatch.setattr(benchmark_check, "load_engine", recording)

    state = run_with(tmp_path, env_files, source=INPLACE_INPUT, arrays=[ONES])

    assert state["outcome"] == "success"
    assert state["benchmark"]["values"] == [pytest.approx(1.0)]
    assert torch.equal(seen["engine_input"], torch.ones(64, 64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_score_fails_accuracy(tmp_path, caplog, monkeypatch, env_files, bad):
    monkeypatch.setattr(
        benchmark_check, "load_engine",
        lambda state: FakeEngineRunner(state, perturb=lambda outs: [torch.full_like(outs[0], bad)]),
    )

    state = run_with(tmp_path, env_files, source=SCALE, arrays=[ONES], threshold=0.0)

    assert state["error"] == "accuracy"
    assert "not a finite number" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["accuracy"]


def test_timed_returns_one_sample_per_run_after_warmup(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark_check, "WARMUP", 3)
    monkeypatch.setattr(benchmark_check, "RUNS", 7)

    samples = benchmark_check.timed(lambda: calls.append(1), on_cuda=False)

    assert len(calls) == 10 and len(samples) == 7
    assert all(s >= 0 for s in samples)


def test_stats_mean_median_p95():
    stats = benchmark_check.stats([1.0, 2.0, 3.0, 4.0, 100.0])

    assert stats["mean"] == 22.0 and stats["median"] == 3.0
    assert 4.0 <= stats["p95"] <= 100.0


def test_success_reports_the_distribution_and_the_metric(tmp_path, env_files):
    state = run_with(tmp_path, env_files)

    assert state["outcome"] == "success"
    bench = state["benchmark"]
    for side in ("eager_ms", "tensorrt_ms"):
        assert set(bench[side]) == {"mean", "median", "p95"}
        assert all(v >= 0 for v in bench[side].values())
    assert bench["speedup"] == pytest.approx(bench["eager_ms"]["median"] / bench["tensorrt_ms"]["median"])
    assert bench["metric"] == "cosine" and bench["threshold"] == 0.99
    assert bench["values"] == [pytest.approx(1.0)]
    on_disk = json.loads((tmp_path / ".artifact" / "benchmark.json").read_text())
    assert on_disk == bench
    message = state["customer_message"]
    assert "benchmark succeeded" in message
    assert "eager:" in message and "TensorRT:" in message and "p95" in message and "median" in message
    assert "faster than eager" in message or "slower than eager" in message
    assert "Cosine similarity: 1.0000 (threshold 0.99)" in message


def test_each_output_is_compared(tmp_path, env_files):
    state = run_with(tmp_path, env_files, source=TWO_OUTPUTS, arrays=XY)

    assert state["outcome"] == "success"
    assert state["benchmark"]["values"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_output_below_threshold_is_the_accuracy_stage(tmp_path, caplog, monkeypatch, env_files):
    def noisy(state):
        return FakeEngineRunner(state, perturb=lambda outs: [outs[0], torch.randn_like(outs[1])])

    monkeypatch.setattr(benchmark_check, "load_engine", noisy)

    state = run_with(tmp_path, env_files, source=TWO_OUTPUTS, arrays=XY, threshold=0.999)

    assert state["outcome"] == "failure"
    assert state["error"] == "accuracy"
    assert "output 1" in state["customer_message"] and "0.999" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["accuracy"]
    assert not (tmp_path / ".artifact" / "benchmark.json").exists()


def test_threshold_from_config_decides(tmp_path, monkeypatch, env_files):
    torch.manual_seed(0)

    def slightly_off(state):  # a few percent of multiplicative noise: cosine well above 0.5, below 0.9999
        return FakeEngineRunner(state, perturb=lambda outs: [o * (1 + 0.05 * torch.randn_like(o)) for o in outs])

    monkeypatch.setattr(benchmark_check, "load_engine", slightly_off)

    passed = run_with(tmp_path, env_files, source=SCALE, arrays=[ONES], threshold=0.5)
    failed = run_with(tmp_path, env_files, source=SCALE, arrays=[ONES], threshold=0.9999)

    assert passed["outcome"] == "success"
    assert 0.99 < passed["benchmark"]["values"][0] < 0.9999  # about 1 - var(noise) / 2
    assert failed["error"] == "accuracy"


@pytest.mark.parametrize(
    "perturb, fragment",
    [
        pytest.param(lambda outs: outs[:1], "1 output", id="fewer_outputs"),
        pytest.param(lambda outs: [o.reshape(-1) for o in outs], "shape", id="shape_mismatch"),
    ],
)
def test_output_structure_mismatch_is_compare_outputs(tmp_path, caplog, monkeypatch, env_files, perturb, fragment):
    monkeypatch.setattr(benchmark_check, "load_engine", lambda state: FakeEngineRunner(state, perturb=perturb))

    state = run_with(tmp_path, env_files, source=TWO_OUTPUTS, arrays=XY)

    assert state["error"] == "compare_outputs"
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["compare_outputs"]


def test_engine_failure_is_benchmark_engine(tmp_path, caplog, monkeypatch, env_files):
    def broken(state):
        raise RuntimeError("deserialize_cuda_engine returned None")

    monkeypatch.setattr(benchmark_check, "load_engine", broken)

    state = run_with(tmp_path, env_files)

    assert state["error"] == "benchmark_engine"
    assert "deserialize" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["benchmark_engine"]


def test_eager_failure_during_timing_is_benchmark_eager(tmp_path, caplog, monkeypatch, good_model, artifacts):
    calls = {"n": 0}
    real_flatten = benchmark_check.flatten_outputs

    def flaky(outputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("eager blew up under repetition")
        return real_flatten(outputs)

    monkeypatch.setattr(benchmark_check, "flatten_outputs", flaky)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "benchmark_eager"
    assert "blew up" in state["error_detail"]
    assert [r.stage for r in error_records(caplog)] == ["benchmark_eager"]


def test_missing_tensorrt_is_a_support_failure(monkeypatch, caplog, good_model, artifacts):
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("tensorrt is installed here; the import stage cannot be exercised")
    monkeypatch.setattr(benchmark_check, "load_engine", benchmark_check.EngineRunner)

    state = eager_check.run(*good_model, artifacts, device="cpu")

    assert state["error"] == "benchmark_runtime"
    assert "tensorrt" in state["error_detail"]

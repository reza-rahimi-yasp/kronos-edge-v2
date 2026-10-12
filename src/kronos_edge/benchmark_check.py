"""Benchmark check: time eager against the TensorRT engine and compare their outputs.

Runs inside the container after the engine build, on the eager model, the
user's inputs, and the engine bytes already in state. Both sides are warmed
up, then run many times; every run's latency is kept and reported as mean,
median, and p95 in milliseconds, with the speedup as the ratio of medians.
Outputs are paired by position (the ONNX output order is the forward return
order), flattened, and compared with the configured metric; every pair must
reach the threshold from config.yaml.

The engine runner needs the GPU and tensorrt; tests replace ``load_engine``.
The TensorRT ``Runtime`` must outlive the engine, and shape-tensor inputs
are read from host memory, so the runner keeps both alive.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from kronos_edge.artifacts import BENCHMARK_FILE
from kronos_edge.log import PACKAGE_LOGGER, get_logger
from kronos_edge.messages import BENCHMARK as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

# Explicit name: imported by eager_check, which runs under ``python -m``.
logger = get_logger(f"{PACKAGE_LOGGER}.benchmark_check")

WARMUP = 20
RUNS = 200
METRIC = "cosine"


def flatten_outputs(outputs) -> list[torch.Tensor]:
    """Model outputs as a flat list of tensors, in the order the ONNX export emits them."""
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (tuple, list)):
        return [t for item in outputs for t in flatten_outputs(item)]
    raise TypeError(f"unsupported output type {type(outputs).__name__}; outputs must be tensors or tuples of tensors")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().flatten().float(), b.detach().flatten().float().to(a.device)
    na, nb = a.norm(), b.norm()
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def timed(fn, on_cuda: bool) -> list[float]:
    """Warm up, then run ``fn`` RUNS times; one latency sample per run, in milliseconds."""
    for _ in range(WARMUP):
        fn()
    if on_cuda:
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(RUNS)]
        for start, end in events:
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) for start, end in events]
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}


class EngineRunner:
    """Executes a serialized TensorRT engine on the user's inputs; ``outputs`` are torch tensors on the GPU."""

    def __init__(self, state: State):
        import tensorrt as trt

        from kronos_edge.tensorrt_check import forwarding_logger

        to_torch = {
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8,
            trt.DataType.UINT8: torch.uint8, trt.DataType.BOOL: torch.bool,
        }
        self.log = forwarding_logger(trt, logger)
        self.runtime = trt.Runtime(self.log)  # must outlive the engine
        self.engine = self.runtime.deserialize_cuda_engine(state["engine"])
        if self.engine is None:
            raise RuntimeError("; ".join(self.log.errors) or "deserialize_cuda_engine returned None")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != len(state["user_inputs"]):
            raise RuntimeError(f"the engine has {len(inputs)} inputs but the model takes {len(state['user_inputs'])}")
        self.buffers = {}
        for name, tensor in zip(inputs, state["user_inputs"]):
            dtype = to_torch[self.engine.get_tensor_dtype(name)]
            # Shape tensors are read on the host; everything else lives on the GPU.
            device = "cpu" if self.engine.is_shape_inference_io(name) else state["device"]
            self.buffers[name] = tensor.detach().to(device=device, dtype=dtype).contiguous()
            self.context.set_input_shape(name, tuple(self.buffers[name].shape))
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self.buffers[name] = torch.empty(shape, dtype=to_torch[self.engine.get_tensor_dtype(name)], device=state["device"])
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            self.outputs.append(self.buffers[name])

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("; ".join(self.log.errors) or "execute_async_v3 returned False")


load_engine = EngineRunner


def benchmark(state: State) -> State:
    stage = "benchmark_eager"
    try:
        model, inputs, threshold = state["model"], state["user_inputs"], state["threshold"]
        on_cuda = str(state.get("device", "cpu")).startswith("cuda")
        logger.info("timing eager: %d warm-up, %d runs", WARMUP, RUNS, extra={"stage": stage})
        with torch.no_grad():
            eager_outputs = flatten_outputs(model(*inputs))
            eager_ms = timed(lambda: model(*inputs), on_cuda)

        stage = "benchmark_engine"
        try:
            runner = load_engine(state)
        except ImportError:
            stage = "benchmark_runtime"
            raise
        logger.info("timing the TensorRT engine: %d warm-up, %d runs", WARMUP, RUNS, extra={"stage": stage})
        runner.run()
        trt_ms = timed(runner.run, on_cuda)
        trt_outputs = list(runner.outputs)

        stage = "compare_outputs"
        if len(trt_outputs) != len(eager_outputs):
            raise ValueError(f"eager returned {len(eager_outputs)} outputs, the engine {len(trt_outputs)} output(s)")
        for i, (e, t) in enumerate(zip(eager_outputs, trt_outputs)):
            if tuple(e.shape) != tuple(t.shape):
                raise ValueError(f"output {i}: eager shape {tuple(e.shape)}, engine shape {tuple(t.shape)}")

        stage = "accuracy"
        values = [cosine(e, t) for e, t in zip(eager_outputs, trt_outputs)]
        for i, value in enumerate(values):
            logger.info("output %d: %s %.10f", i, METRIC, value, extra={"stage": stage})
        failing = [i for i, v in enumerate(values) if v < threshold]
        if failing:
            raise ValueError(
                "; ".join(f"output {i}: {METRIC} similarity {values[i]:.10f} is below the threshold {threshold}" for i in failing)
            )

        result = {
            "eager_ms": stats(eager_ms), "tensorrt_ms": stats(trt_ms),
            "metric": METRIC, "threshold": threshold, "values": values,
        }
        result["speedup"] = result["eager_ms"]["median"] / result["tensorrt_ms"]["median"]
        path = Path(state["artifacts_dir"]) / BENCHMARK_FILE
        path.write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", path, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"benchmark": result}

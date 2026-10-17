"""Benchmark check: time eager against the TensorRT engine and compare their outputs.

Runs in the model-check subprocess after the engine build, on the eager model, the
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
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_leaves

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
    """Model outputs as a flat list of tensors, in the order the ONNX export emits them.

    The exporter flattens with pytree, so tuples, lists, and dicts (insertion order) all match.
    """
    leaves = tree_leaves(outputs)
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            raise TypeError(f"unsupported output type {type(leaf).__name__}; every output must be a tensor")
    return leaves


def clone_inputs(inputs):
    """Fresh copies, so a model that mutates its inputs in place cannot skew a later use of them."""
    return tuple(t.detach().clone() for t in inputs)


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
    """Executes a serialized TensorRT engine on the user's inputs; ``outputs`` are torch tensors on the GPU.

    Outputs whose shape depends on the data (for example ``torch.nonzero``)
    are still ``-1`` after shape inference, so those get a TensorRT output
    allocator that hands out GPU memory during the run and reports the shape
    afterwards. ``outputs`` is read after a run.
    """

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
        self.dtypes = {n: to_torch[self.engine.get_tensor_dtype(n)] for n in self.output_names}
        self.allocators = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                self.allocators[name] = _OutputAllocator(trt, state["device"])
                if not self.context.set_output_allocator(name, self.allocators[name]):
                    raise RuntimeError(f"set_output_allocator failed for data-dependent output {name}")
                logger.info("output %s has a data-dependent shape; allocating at run time", name, extra={"stage": "benchmark_engine"})
            else:
                self.buffers[name] = torch.empty(shape, dtype=self.dtypes[name], device=state["device"])
                self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("; ".join(self.log.errors) or "execute_async_v3 returned False")

    @property
    def outputs(self) -> list[torch.Tensor]:
        torch.cuda.synchronize()
        return [
            self.allocators[n].tensor(self.dtypes[n]) if n in self.allocators else self.buffers[n]
            for n in self.output_names
        ]


def _OutputAllocator(trt, device):
    class OutputAllocator(trt.IOutputAllocator):
        """GPU memory for one data-dependent output, grown on demand; the shape arrives via notify_shape."""

        def __init__(self):
            super().__init__()
            self.memory = torch.empty(0, dtype=torch.uint8, device=device)
            self.shape = None

        def reallocate_output_async(self, name, memory, size, alignment, stream):
            if self.memory.numel() < size:
                self.memory = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
            return self.memory.data_ptr()

        def notify_shape(self, name, shape):
            self.shape = tuple(shape)

        def tensor(self, dtype) -> torch.Tensor:
            if self.shape is None:
                raise RuntimeError("the engine never reported the output's shape")
            numel = math.prod(self.shape)
            return self.memory[: numel * torch.empty(0, dtype=dtype).element_size()].view(dtype).reshape(self.shape)

    return OutputAllocator()


load_engine = EngineRunner


def benchmark(state: State) -> State:
    stage = "benchmark_eager"
    try:
        model, inputs, threshold = state["model"], state["user_inputs"], state["threshold"]
        on_cuda = str(state.get("device", "cpu")).startswith("cuda")
        logger.info("timing eager: %d warm-up, %d runs", WARMUP, RUNS, extra={"stage": stage})
        # The reference, the timing loop, and the engine each get their own copies of the inputs:
        # a model that mutates them in place must not change what the other two see.
        timing_inputs = clone_inputs(inputs)
        with torch.no_grad():
            eager_outputs = flatten_outputs(model(*clone_inputs(inputs)))
            eager_ms = timed(lambda: model(*timing_inputs), on_cuda)

        stage = "benchmark_engine"
        try:
            runner = load_engine({**state, "user_inputs": clone_inputs(inputs)})
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
        problems = []
        for i, value in enumerate(values):
            if not math.isfinite(value):
                problems.append(f"output {i}: {METRIC} similarity is not a finite number ({value}); the engine's output is unusable")
            elif value < threshold:
                problems.append(f"output {i}: {METRIC} similarity {value:.10f} is below the threshold {threshold}")
        if problems:
            raise ValueError("; ".join(problems))

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

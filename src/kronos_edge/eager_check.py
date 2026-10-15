"""The container graph: eager run, ONNX export and simplification, TensorRT engine build, benchmark.

model.py must define ``Model`` (a torch.nn.Module subclass) and
``get_inputs()`` or ``get_fwd_args()`` returning the reference inputs.
``get_init_inputs()`` is optional and supplies constructor arguments.
input.npz holds one array per reference input under ``arr_0 .. arr_n-1``,
which is what ``np.savez("input.npz", *arrays)`` writes.
Packages model.py needs beyond the image arrive as ``--dependency`` and are
pip-installed into this container before model.py is imported, pinned to the
versions already in the venv so they cannot replace the image's packages.
"""

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph

from kronos_edge import benchmark_check, onnx_check, onnxsim_check, tensorrt_check
from kronos_edge.config import PRECISIONS
from kronos_edge.log import PACKAGE_LOGGER, configure, get_logger
from kronos_edge.messages import EAGER as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail, route_on_error

# Explicit name: this module runs under ``python -m`` inside the container.
logger = get_logger(f"{PACKAGE_LOGGER}.eager_check")

INPUT_NAMES = ("get_inputs", "get_fwd_args")



class InvalidUserInputs(ValueError):
    """input.npz does not match the reference inputs from get_inputs()."""


class MissingDependency(ImportError):
    """model.py imports a package the environment does not have."""


def installed_versions() -> list[str]:
    """Every distribution in this interpreter as ``name==version``: the validated runtime, torch's build label included."""
    pins = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower() not in pins:  # first on sys.path is what imports resolve to
            pins[name.lower()] = f"{name}=={dist.version}"
    return sorted(pins.values(), key=str.lower)


def install(packages: list[str]) -> None:
    """pip install into this container's venv. Output is captured: stdout belongs to the user's model.

    Every package already in the venv is passed as a pip constraint, so a requirement, direct or transitive,
    that would replace torch or another image package is a resolver conflict and nothing is installed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "constraints.txt"
        constraints.write_text("".join(f"{pin}\n" for pin in installed_versions()))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--constraint", str(constraints), *packages],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"pip exited {result.returncode}\n{result.stdout}{result.stderr}".rstrip())
    logger.debug("%s", result.stdout.rstrip(), extra={"stage": "install_dependencies"})


def _import_module(path: str):
    spec = importlib.util.spec_from_file_location("kronos_user_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_init_args(model_cls: type) -> list[str]:
    params = list(inspect.signature(model_cls.__init__).parameters.values())[1:]
    return [
        p.name
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def load_model(state: State) -> State:
    stage = "select_device"
    try:
        device = state.get("device", "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device} requested but torch.cuda.is_available() is False")
        logger.info("device %s", device, extra={"stage": stage})

        stage = "install_dependencies"
        dependencies = list(state.get("dependencies", ()))
        if dependencies:
            logger.info("installing %s", ", ".join(dependencies), extra={"stage": stage})
            install(dependencies)

        stage = "import_module"
        logger.info("loading %s", state["model_path"], extra={"stage": stage})
        try:
            module = _import_module(state["model_path"])
        except ModuleNotFoundError as exc:
            stage = "missing_dependency"
            raise MissingDependency(f"model.py imports {exc.name!r}, which is not installed in the environment") from exc

        stage = "find_model_class"
        model_cls = getattr(module, "Model", None)
        if not (isinstance(model_cls, type) and issubclass(model_cls, torch.nn.Module)):
            raise ValueError("does not define a 'Model' class")

        stage = "find_get_inputs"
        get_inputs = next((getattr(module, n) for n in INPUT_NAMES if hasattr(module, n)), None)
        if get_inputs is None:
            raise ValueError("does not define get_inputs() or get_fwd_args()")

        stage = "init_model"
        if hasattr(module, "get_init_inputs"):
            init_inputs = tuple(module.get_init_inputs())
        else:
            required = _required_init_args(model_cls)
            if required:
                raise ValueError(
                    f"Model.__init__ requires arguments ({', '.join(required)}) "
                    "but model.py does not define get_init_inputs()"
                )
            init_inputs = ()
        model = model_cls(*init_inputs).eval().to(device)
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"model": model, "get_inputs": get_inputs}


def validate_inputs(state: State) -> State:
    stage = "get_inputs"
    try:
        reference = tuple(state["get_inputs"]())

        stage = "load_npz"
        with np.load(state["npz_path"]) as data:
            arrays = dict(data)

        stage = "validate_inputs"
        expected_keys = [f"arr_{i}" for i in range(len(reference))]
        if sorted(arrays) != sorted(expected_keys):
            raise InvalidUserInputs(f"expected keys {expected_keys}, found {sorted(arrays)}")
        user_inputs = []
        for key, ref in zip(expected_keys, reference):
            tensor = torch.from_numpy(arrays[key])
            logger.trace("%s shape=%s dtype=%s", key, tuple(tensor.shape), tensor.dtype, extra={"stage": stage})
            if tuple(tensor.shape) != tuple(ref.shape) or tensor.dtype != ref.dtype:
                raise InvalidUserInputs(
                    f"{key}: expected shape {tuple(ref.shape)} dtype {ref.dtype}, "
                    f"found shape {tuple(tensor.shape)} dtype {tensor.dtype}"
                )
            user_inputs.append(tensor.to(state.get("device", "cpu")))
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"user_inputs": tuple(user_inputs)}


def run_eager(state: State) -> State:
    stage = "forward"
    try:
        logger.info("running forward in eager mode", extra={"stage": stage})
        with torch.no_grad():  # on copies: a model that mutates its inputs must not alter what later nodes see
            state["model"](*[t.clone() for t in state["user_inputs"]])
    except Exception as exc:
        return fail(logger, stage, exc)
    return {}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("load_model", load_model)
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("run_eager", run_eager)
    graph.add_node("export_onnx", onnx_check.export_onnx)
    graph.add_node("simplify_onnx", onnxsim_check.simplify_onnx)
    graph.add_node("build_engine", tensorrt_check.build_engine)
    graph.add_node("benchmark", benchmark_check.benchmark)
    graph.add_edge(START, "load_model")
    graph.add_conditional_edges("load_model", route_on_error, {"ok": "validate_inputs", "failure": END})
    graph.add_conditional_edges("validate_inputs", route_on_error, {"ok": "run_eager", "failure": END})
    graph.add_conditional_edges("run_eager", route_on_error, {"ok": "export_onnx", "failure": END})
    graph.add_conditional_edges("export_onnx", route_on_error, {"ok": "simplify_onnx", "failure": END})
    graph.add_conditional_edges("simplify_onnx", route_on_error, {"ok": "build_engine", "failure": END})
    graph.add_conditional_edges("build_engine", route_on_error, {"ok": "benchmark", "failure": END})
    graph.add_edge("benchmark", END)
    return graph.compile()


def run(
    model_path, npz_path, artifacts_dir, precision: str = "fp32", threshold: float = 0.99, device: str = "cpu",
    dependencies: list[str] = (),
) -> State:
    return build_graph().invoke(
        {
            "model_path": str(model_path), "npz_path": str(npz_path), "artifacts_dir": str(artifacts_dir),
            "precision": precision, "threshold": threshold, "device": device, "dependencies": list(dependencies),
        }
    )


def report(state: State) -> str:
    """The JSON document the host reads: the failure stage and detail, or no error and the benchmark."""
    return json.dumps(
        {"error": state.get("error"), "error_detail": state.get("error_detail", ""), "benchmark": state.get("benchmark")}
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the eager check and write a JSON report.")
    parser.add_argument("model_path")
    parser.add_argument("npz_path")
    parser.add_argument("--report", required=True, help="where to write the JSON report")
    parser.add_argument("--artifacts", required=True, help="existing directory for the files the checks produce")
    parser.add_argument("--precision", required=True, choices=PRECISIONS, help="TensorRT engine precision")
    parser.add_argument("--threshold", required=True, type=float, help="minimum metric value for every output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dependency", action="append", default=[], metavar="REQUIREMENT",
        help="pip requirement to install before importing model.py; repeatable",
    )
    args = parser.parse_args(argv)
    configure(args.log_level)
    state = run(
        args.model_path, args.npz_path, args.artifacts,
        precision=args.precision, threshold=args.threshold, device=args.device, dependencies=args.dependency,
    )
    with open(args.report, "w") as f:
        f.write(report(state))
    return 0 if state.get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

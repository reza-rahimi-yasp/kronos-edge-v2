"""Eager-mode check: run a user's model.py against input.npz and report.

model.py must define ``Model`` (a torch.nn.Module subclass) and
``get_inputs()`` or ``get_fwd_args()`` returning the reference inputs.
``get_init_inputs()`` is optional and supplies constructor arguments.
input.npz holds one array per reference input under ``arr_0 .. arr_n-1``,
which is what ``np.savez("input.npz", *arrays)`` writes.
"""

import argparse
import importlib.util
import inspect
import logging
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph

from kronos_edge.log import PACKAGE_LOGGER, ErrorCollector, configure, get_logger

logger = get_logger(f"{PACKAGE_LOGGER}.eager_check")

INPUT_NAMES = ("get_inputs", "get_fwd_args")

CUSTOMER_MESSAGES = {
    "import_module": "model.py could not be loaded.",
    "find_model_class": "model.py does not define a 'Model' class.",
    "find_get_inputs": "model.py does not define get_inputs() or get_fwd_args().",
    "init_model": "Model could not be constructed.",
    "get_inputs": "get_inputs() raised an exception.",
    "load_npz": "input.npz could not be read.",
    "validate_inputs": "input.npz does not match the inputs returned by get_inputs().",
    "forward": "Your model cannot be run through eager mode. Either fix it or contact support.",
}


class InvalidUserInputs(ValueError):
    """input.npz does not match the reference inputs from get_inputs()."""


class State(TypedDict, total=False):
    model_path: str
    npz_path: str
    collector: ErrorCollector
    model: torch.nn.Module
    get_inputs: Callable[[], object]
    user_inputs: tuple[torch.Tensor, ...]
    error: str | None
    outcome: str
    customer_message: str


def _fail(stage: str, exc: Exception) -> State:
    extra = {"stage": stage}
    logger.error("%s: %s", type(exc).__name__, exc, extra=extra)
    logger.debug("traceback", exc_info=exc, extra=extra)
    return {"error": stage}


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
    stage = "import_module"
    try:
        logger.info("loading %s", state["model_path"], extra={"stage": stage})
        module = _import_module(state["model_path"])

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
        model = model_cls(*init_inputs).eval()
    except Exception as exc:
        return _fail(stage, exc)
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
            user_inputs.append(tensor)
    except Exception as exc:
        return _fail(stage, exc)
    return {"user_inputs": tuple(user_inputs)}


def run_eager(state: State) -> State:
    stage = "forward"
    try:
        logger.info("running forward in eager mode", extra={"stage": stage})
        with torch.no_grad():
            state["model"](*state["user_inputs"])
    except Exception as exc:
        return _fail(stage, exc)
    return {}


def success(state: State) -> State:
    logger.info("eager mode run succeeded")
    return {"outcome": "success", "customer_message": "Eager mode run succeeded."}


def failure(state: State) -> State:
    records = [r for r in state["collector"].records if r.levelno >= logging.ERROR]
    stage = records[0].stage if records else state["error"]
    detail = records[0].getMessage() if records else ""
    message = f"{CUSTOMER_MESSAGES[stage]}\n{detail}".rstrip()
    return {"outcome": "failure", "customer_message": message}


def route_on_error(state: State) -> str:
    return "failure" if state.get("error") else "ok"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("load_model", load_model)
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("run_eager", run_eager)
    graph.add_node("success", success)
    graph.add_node("failure", failure)
    graph.add_edge(START, "load_model")
    graph.add_conditional_edges("load_model", route_on_error, {"ok": "validate_inputs", "failure": "failure"})
    graph.add_conditional_edges("validate_inputs", route_on_error, {"ok": "run_eager", "failure": "failure"})
    graph.add_conditional_edges("run_eager", route_on_error, {"ok": "success", "failure": "failure"})
    graph.add_edge("success", END)
    graph.add_edge("failure", END)
    return graph.compile()


def run_check(model_path, npz_path) -> State:
    collector = ErrorCollector()
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    package_logger.addHandler(collector)
    try:
        return build_graph().invoke(
            {"model_path": str(model_path), "npz_path": str(npz_path), "collector": collector}
        )
    finally:
        package_logger.removeHandler(collector)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_path")
    parser.add_argument("npz_path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure(args.log_level)
    state = run_check(args.model_path, args.npz_path)
    print(state["customer_message"])
    return 0 if state["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

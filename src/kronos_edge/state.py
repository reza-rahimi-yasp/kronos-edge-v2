"""Shared workflow state, routing, and the failure helper used by every check."""

import logging
from collections.abc import Callable
from typing import Any, TypedDict

from kronos_edge.config import Config, Environment


class State(TypedDict, total=False):
    config_path: str
    environments_path: str
    config: Config
    environment: Environment
    model_path: str
    npz_path: str
    artifacts_path: str  # the artifact root
    run_path: str  # this run's directory under it, where the checks write
    artifacts_dir: str
    precision: str
    threshold: float
    model: Any
    get_inputs: Callable[[], object]
    user_inputs: tuple[Any, ...]
    onnx_model: Any
    simplified_onnx: Any
    engine: bytes
    benchmark: dict[str, Any]
    kernel_path: str
    kernelgen_report: str  # the kernelgen report as `yasp-toolkit eval inspect` renders it
    device: str
    log_level: str
    error: str | None
    error_detail: str
    # The ONNX/TensorRT branch's failure: kept apart from `error` so the kernelgen branch still runs after it.
    tensorrt_error: str
    tensorrt_error_detail: str
    outcome: str
    customer_message: str


def fail(logger: logging.Logger, stage: str, exc: Exception) -> State:
    """Log the failure under ``stage`` and return the state update that routes to failure."""
    extra = {"stage": stage}
    logger.error("%s: %s", type(exc).__name__, exc, extra=extra)
    logger.debug("traceback", exc_info=exc, extra=extra)
    return {"error": stage, "error_detail": f"{type(exc).__name__}: {exc}"}


def route_on_error(state: State) -> str:
    return "failure" if state.get("error") else "ok"

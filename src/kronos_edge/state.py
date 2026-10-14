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
    artifacts_path: str
    artifacts_dir: str
    precision: str
    threshold: float
    dependencies: list[str]
    model: Any
    get_inputs: Callable[[], object]
    user_inputs: tuple[Any, ...]
    exported_program: Any
    onnx_model: Any
    simplified_onnx: Any
    engine: bytes
    benchmark: dict[str, Any]
    evaluator: Any  # the Gaia worker container ensure_evaluator started, for release_evaluator; None when it started none
    kernel_path: str
    kernelgen_result: dict
    kernelgen_speedup: float
    device: str
    log_level: str
    error: str | None
    error_detail: str
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

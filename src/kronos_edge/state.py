"""Shared workflow state, routing, and the failure helper used by every check."""

import logging
from collections.abc import Callable
from typing import TypedDict

import torch


class State(TypedDict, total=False):
    model_path: str
    npz_path: str
    model: torch.nn.Module
    get_inputs: Callable[[], object]
    user_inputs: tuple[torch.Tensor, ...]
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

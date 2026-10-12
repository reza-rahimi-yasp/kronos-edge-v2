"""Workflow: compose the checks into a LangGraph and expose the CLI."""

import argparse

from langgraph.graph import END, START, StateGraph

from kronos_edge import eager_check
from kronos_edge.log import PACKAGE_LOGGER, configure, get_logger
from kronos_edge.state import State, route_on_error

# Explicit name: under ``python -m`` this module's __name__ is "__main__",
# which would fall outside the package logger hierarchy.
logger = get_logger(f"{PACKAGE_LOGGER}.workflow")

CUSTOMER_MESSAGES = {**eager_check.CUSTOMER_MESSAGES}


def success(state: State) -> State:
    logger.info("eager mode run succeeded")
    return {"outcome": "success", "customer_message": "Eager mode run succeeded."}


def failure(state: State) -> State:
    message = f"{CUSTOMER_MESSAGES[state['error']]}\n{state['error_detail']}"
    return {"outcome": "failure", "customer_message": message}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("load_model", eager_check.load_model)
    graph.add_node("validate_inputs", eager_check.validate_inputs)
    graph.add_node("run_eager", eager_check.run_eager)
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
    return build_graph().invoke({"model_path": str(model_path), "npz_path": str(npz_path)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

"""Workflow: validate the environment on the host, run the eager check in its container."""

import argparse
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from kronos_edge import container, env_check
from kronos_edge.log import PACKAGE_LOGGER, configure, get_logger
from kronos_edge.messages import CUSTOMER_MESSAGES
from kronos_edge.state import State, route_on_error

# Explicit name: under ``python -m`` this module's __name__ is "__main__",
# which would fall outside the package logger hierarchy.
logger = get_logger(f"{PACKAGE_LOGGER}.workflow")

DEFAULT_ENVIRONMENTS = Path(__file__).with_name("environments.yaml")


def success(state: State) -> State:
    logger.info("eager mode run and ONNX export succeeded")
    return {"outcome": "success", "customer_message": "Eager mode run and ONNX export succeeded."}


def failure(state: State) -> State:
    message = f"{CUSTOMER_MESSAGES[state['error']]}\n{state['error_detail']}"
    return {"outcome": "failure", "customer_message": message}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("load_config", env_check.load_config)
    graph.add_node("match_environment", env_check.match_environment)
    graph.add_node("run_in_container", container.run_in_container)
    graph.add_node("success", success)
    graph.add_node("failure", failure)
    graph.add_edge(START, "load_config")
    graph.add_conditional_edges("load_config", route_on_error, {"ok": "match_environment", "failure": "failure"})
    graph.add_conditional_edges("match_environment", route_on_error, {"ok": "run_in_container", "failure": "failure"})
    graph.add_conditional_edges("run_in_container", route_on_error, {"ok": "success", "failure": "failure"})
    graph.add_edge("success", END)
    graph.add_edge("failure", END)
    return graph.compile()


def run_check(config_path, model_path, npz_path, environments_path=DEFAULT_ENVIRONMENTS, log_level="INFO") -> State:
    return build_graph().invoke(
        {
            "config_path": str(config_path),
            "environments_path": str(environments_path),
            "model_path": str(model_path),
            "npz_path": str(npz_path),
            "log_level": log_level,
        }
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("model_path")
    parser.add_argument("npz_path")
    parser.add_argument("--environments", default=DEFAULT_ENVIRONMENTS, help="registry of supported environments")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure(args.log_level)
    state = run_check(
        args.config_path, args.model_path, args.npz_path,
        environments_path=args.environments, log_level=args.log_level,
    )
    print(state["customer_message"])
    return 0 if state["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

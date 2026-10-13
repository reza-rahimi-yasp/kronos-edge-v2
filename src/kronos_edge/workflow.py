"""Workflow: validate the environment on the host, run the checks in the container, then the evaluator and kernelgen on the host."""

import argparse
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from kronos_edge import container, env_check, evaluator_check, kernelgen_check
from kronos_edge.log import PACKAGE_LOGGER, configure, get_logger
from kronos_edge.messages import CUSTOMER_MESSAGES
from kronos_edge.state import State, route_on_error

# Explicit name: under ``python -m`` this module's __name__ is "__main__",
# which would fall outside the package logger hierarchy.
logger = get_logger(f"{PACKAGE_LOGGER}.workflow")

DEFAULT_ENVIRONMENTS = Path(__file__).with_name("environments.yaml")
DEFAULT_ARTIFACTS = Path(".artifact")


def _latency(name: str, ms: dict) -> str:
    return f"Latency per run, {name}: mean {ms['mean']:.3f} ms, median {ms['median']:.3f} ms, p95 {ms['p95']:.3f} ms."


def success(state: State) -> State:
    precision = state["config"].precision
    bench = state["benchmark"]
    speedup = bench["speedup"]
    kernelgen_speedup = state["kernelgen_speedup"]
    comparison = f"{speedup:.1f}x faster" if speedup >= 1 else f"{1 / speedup:.1f}x slower"
    values = ", ".join(f"{v:.4f}" for v in bench["values"])
    logger.info(
        "all checks succeeded at %s; speedup %.2fx; kernelgen speedup %.2fx", precision, speedup, kernelgen_speedup
    )
    return {
        "outcome": "success",
        "customer_message": "\n".join([
            f"Eager run, ONNX export, ONNX simplification, TensorRT engine build ({precision}), and benchmark succeeded.",
            _latency("eager", bench["eager_ms"]),
            _latency("TensorRT", bench["tensorrt_ms"]),
            f"TensorRT is {comparison} than eager (median). "
            f"{bench['metric'].capitalize()} similarity: {values} (threshold {bench['threshold']}).",
            f"The kernelgen optimization and kernelgen evaluation (speedup {kernelgen_speedup:.2f}x) succeeded.",
        ]),
    }


def failure(state: State) -> State:
    message = f"{CUSTOMER_MESSAGES[state['error']]}\n{state['error_detail']}"
    return {"outcome": "failure", "customer_message": message}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("load_config", env_check.load_config)
    graph.add_node("match_environment", env_check.match_environment)
    graph.add_node("run_in_container", container.run_in_container)
    graph.add_node("ensure_evaluator", evaluator_check.ensure_evaluator)
    graph.add_node("kernelgen", kernelgen_check.kernelgen)
    graph.add_node("evaluate_kernelgen", kernelgen_check.evaluate_kernelgen)
    graph.add_node("release_evaluator", evaluator_check.release_evaluator)
    graph.add_node("success", success)
    graph.add_node("failure", failure)
    graph.add_edge(START, "load_config")
    graph.add_conditional_edges("load_config", route_on_error, {"ok": "match_environment", "failure": "failure"})
    graph.add_conditional_edges("match_environment", route_on_error, {"ok": "run_in_container", "failure": "failure"})
    graph.add_conditional_edges("run_in_container", route_on_error, {"ok": "ensure_evaluator", "failure": "failure"})
    # From ensure_evaluator on, every path goes through release_evaluator, so an evaluator this check started is
    # stopped whatever happens after it; release_evaluator keeps the error that brought it there.
    graph.add_conditional_edges("ensure_evaluator", route_on_error, {"ok": "kernelgen", "failure": "release_evaluator"})
    graph.add_conditional_edges("kernelgen", route_on_error, {"ok": "evaluate_kernelgen", "failure": "release_evaluator"})
    graph.add_edge("evaluate_kernelgen", "release_evaluator")
    graph.add_conditional_edges("release_evaluator", route_on_error, {"ok": "success", "failure": "failure"})
    graph.add_edge("success", END)
    graph.add_edge("failure", END)
    return graph.compile()


def run_check(
    config_path, model_path, npz_path,
    environments_path=DEFAULT_ENVIRONMENTS, artifacts_path=DEFAULT_ARTIFACTS, log_level="INFO",
) -> State:
    return build_graph().invoke(
        {
            "config_path": str(config_path),
            "environments_path": str(environments_path),
            "model_path": str(model_path),
            "npz_path": str(npz_path),
            "artifacts_path": str(artifacts_path),
            "log_level": log_level,
        }
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("model_path")
    parser.add_argument("npz_path")
    parser.add_argument("--environments", default=DEFAULT_ENVIRONMENTS, help="registry of supported environments")
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS, help="directory for the files the checks produce")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure(args.log_level)
    state = run_check(
        args.config_path, args.model_path, args.npz_path,
        environments_path=args.environments, artifacts_path=args.artifacts, log_level=args.log_level,
    )
    print(state["customer_message"])
    return 0 if state["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

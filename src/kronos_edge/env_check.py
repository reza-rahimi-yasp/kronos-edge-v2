"""Environment check: validate the user's config.yaml and match it to a registry entry.

The config names the GPU, driver, CUDA, TensorRT, and torch versions the user
wants. It must match one registry entry exactly; the preflight then verifies
that the Python environment the check runs in is that entry.
"""

from kronos_edge.config import CONFIG_FIELDS, EnvironmentConfig, load_environments, parse_config, read_yaml
from kronos_edge.log import get_logger
from kronos_edge.messages import ENV as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)


def _describe(config: EnvironmentConfig) -> str:
    return ", ".join(f"{k}={v}" for k, v in config.model_dump(include=set(CONFIG_FIELDS)).items())


def load_config(state: State) -> State:
    stage = "read_config"
    try:
        logger.info("reading %s", state["config_path"], extra={"stage": stage})
        data = read_yaml(state["config_path"])

        stage = "validate_config"
        config = parse_config(data)
        logger.trace("config: %s", _describe(config), extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"config": config}


def match_environment(state: State) -> State:
    stage = "load_environments"
    try:
        environments = load_environments(state["environments_path"])

        stage = "match_environment"
        config = state["config"]
        environment = next((e for e in environments if e.matches(config)), None)
        if environment is None:
            supported = "; ".join(f"{e.name} ({_describe(e)})" for e in environments)
            raise ValueError(f"requested {_describe(config)}; supported: {supported}")
        logger.info("matched environment %s", environment.name, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"environment": environment}

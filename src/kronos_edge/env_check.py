"""Environment check: validate the user's config.yaml and match it to an image.

The config names the GPU, driver, CUDA, TensorRT, and torch versions the user
wants. It must match one registry entry exactly, that entry must describe the
GPU in this machine, and the entry's Docker image must be built.
"""

import subprocess

from kronos_edge.config import CONFIG_FIELDS, EnvironmentConfig, load_environments, parse_config, read_yaml
from kronos_edge.log import get_logger
from kronos_edge.state import State, fail

logger = get_logger(__name__)

CUSTOMER_MESSAGES = {
    "read_config": "config.yaml could not be read.",
    "validate_config": "config.yaml does not match the expected schema.",
    "load_environments": "The supported environment registry could not be loaded. Contact support.",
    "match_environment": "No supported environment matches config.yaml.",
    "host_gpu": "The GPU in this machine does not match the requested environment.",
    "image_available": "The Docker image for the requested environment is not built.",
}


def image_exists(image: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return result.returncode == 0


def host_gpu() -> tuple[str, str]:
    """Return (gpu name, driver version) of the first GPU, as nvidia-smi reports them."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    name, driver = result.stdout.splitlines()[0].split(", ")
    return name.strip(), driver.strip()


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

        stage = "host_gpu"
        name, driver = host_gpu()
        if (name, driver) != (environment.gpu, environment.driver):
            raise ValueError(
                f"this machine has gpu={name}, driver={driver}; "
                f"environment {environment.name} needs gpu={environment.gpu}, driver={environment.driver}"
            )

        stage = "image_available"
        if not image_exists(environment.image):
            raise ValueError(
                f"image {environment.image} not found; build it with: "
                f"docker build -f {environment.dockerfile} -t {environment.image} ."
            )
    except Exception as exc:
        return fail(logger, stage, exc)
    return {"environment": environment}

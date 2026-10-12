"""Every customer-facing message, keyed by failure stage.

This module has no heavy imports on purpose: the host loads it to turn a
stage reported by the container into a message, without needing torch.
Stage names form one flat namespace across every check, so ``merge``
refuses duplicates.
"""


def merge(*tables: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for table in tables:
        duplicates = merged.keys() & table.keys()
        if duplicates:
            raise ValueError(f"duplicate stage names across checks: {sorted(duplicates)}")
        merged.update(table)
    return merged


ENV = {
    "read_config": "config.yaml could not be read.",
    "validate_config": "config.yaml does not match the expected schema.",
    "load_environments": "The supported environment registry could not be loaded. Contact support.",
    "match_environment": "No supported environment matches config.yaml.",
    "host_gpu": "The GPU in this machine does not match the requested environment.",
    "image_available": "The Docker image for the requested environment is not built.",
}

CONTAINER = {
    "container_run": "The check could not be run in the environment container. Contact support.",
}

EAGER = {
    "select_device": "The GPU could not be accessed inside the environment container. Contact support.",
    "import_module": "model.py could not be loaded.",
    "find_model_class": "model.py does not define a 'Model' class.",
    "find_get_inputs": "model.py does not define get_inputs() or get_fwd_args().",
    "init_model": "Model could not be constructed.",
    "get_inputs": "get_inputs() raised an exception.",
    "load_npz": "input.npz could not be read.",
    "validate_inputs": "input.npz does not match the inputs returned by get_inputs().",
    "forward": "Your model cannot be run through eager mode. Either fix it or contact support.",
}

CUSTOMER_MESSAGES = merge(ENV, CONTAINER, EAGER)

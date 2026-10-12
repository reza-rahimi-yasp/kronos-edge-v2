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

ONNX = {
    "onnx_exporter": "The ONNX exporter is not available inside the environment container. Contact support.",
    "onnx_precision": "Your model could not be cast to the requested precision for export. Either fix it or contact support.",
    "onnx_capture": "Your model could not be traced by torch.export for ONNX export. Fix the construct named below.",
    "onnx_convert": "Your model was traced but an operator has no ONNX translation. Replace the operator named below or contact support.",
    "onnx_save": "The exported ONNX model could not be serialized or written to the artifact directory.",
}

ONNXSIM = {
    "onnx_simplifier": "The ONNX simplifier is not available inside the environment container. Contact support.",
    "onnx_simplify": "The ONNX model could not be simplified by onnxsim. Either fix the model or contact support.",
    "onnx_simplified_invalid": "onnxsim produced a simplified model that does not pass its check or the ONNX checker. Contact support.",
    "onnx_simplified_save": "The simplified ONNX model could not be written to the artifact directory.",
}

TENSORRT = {
    "tensorrt_import": "TensorRT is not available inside the environment container. Contact support.",
    "tensorrt_parse": "TensorRT could not parse the ONNX model. Replace the operator or node named below or contact support.",
    "tensorrt_build": "TensorRT could not build an engine from the model. The errors it reported are below.",
    "engine_save": "The TensorRT engine could not be written to the artifact directory.",
}

BENCHMARK = {
    "benchmark_eager": "Your model failed while being run repeatedly in eager mode for timing. Either fix it or contact support.",
    "benchmark_runtime": "The TensorRT runtime is not available inside the environment container. Contact support.",
    "benchmark_engine": "The TensorRT engine could not be loaded or run. The errors it reported are below.",
    "compare_outputs": "The TensorRT engine's outputs do not have the same structure as the eager outputs. Contact support.",
    "accuracy": "The TensorRT engine's outputs do not match the eager outputs within the threshold in config.yaml.",
}

CUSTOMER_MESSAGES = merge(ENV, CONTAINER, EAGER, ONNX, ONNXSIM, TENSORRT, BENCHMARK)

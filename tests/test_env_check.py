import logging

import pytest

from conftest import CONFIG, ENVIRONMENT, write_yaml
from kronos_edge.workflow import main, run_check


def error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith("kronos_edge")]


def run(config, environments, good_model):
    model, npz = good_model
    return run_check(config, model, npz, environments_path=environments)


@pytest.mark.parametrize(
    "setting, value",
    [("precision", "fp16"), ("metric", "cosine"), ("threshold", 0.5)],
)
def test_settings_are_accepted_and_ignored_by_matching(tmp_path, env_files, good_model, setting, value):
    """The registry has no settings; the config matches on the environment fields alone."""
    _, environments = env_files
    config = write_yaml(tmp_path / "config.yaml", {**CONFIG, setting: value})

    state = run(config, environments, good_model)

    assert state["outcome"] == "success"
    assert getattr(state["config"], setting) == value
    assert state["environment"].name == "rtx5090-cuda130"


def test_matching_config_reaches_eager_check(env_files, good_model):
    config, environments = env_files

    state = run(config, environments, good_model)

    assert state["outcome"] == "success"
    assert state["environment"].name == "rtx5090-cuda130"


@pytest.mark.parametrize(
    "contents, fragment",
    [
        pytest.param(None, "could not be read", id="file_missing"),
        pytest.param("gpu: [unclosed", "could not be read", id="invalid_yaml"),
    ],
)
def test_unreadable_config(tmp_path, caplog, env_files, good_model, contents, fragment):
    _, environments = env_files
    config = tmp_path / "broken.yaml"
    if contents is not None:
        config.write_text(contents)

    state = run(config, environments, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "read_config"
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["read_config"]


def without(key):
    return {k: v for k, v in CONFIG.items() if k != key}


@pytest.mark.parametrize(
    "data, fragment",
    [
        pytest.param(without("tensorrt"), "tensorrt", id="missing_field"),
        pytest.param(without("precision"), "precision", id="missing_precision"),
        pytest.param(without("eval_routing"), "eval_routing", id="missing_eval_routing"),
        pytest.param({**CONFIG, "precision": "int8"}, "fp16", id="unsupported_precision"),
        pytest.param({**CONFIG, "precision": "FP16"}, "fp16", id="precision_case_sensitive"),
        pytest.param(without("metric"), "metric", id="missing_metric"),
        pytest.param({**CONFIG, "metric": "mse"}, "cosine", id="unsupported_metric"),
        pytest.param(without("threshold"), "threshold", id="missing_threshold"),
        pytest.param({**CONFIG, "threshold": 1.5}, "threshold", id="threshold_above_one"),
        pytest.param({**CONFIG, "threshold": "0.99"}, "threshold", id="threshold_not_a_number"),
        pytest.param({**CONFIG, "dependencies": ["timm"]}, "dependencies", id="retired_dependencies_field"),
        pytest.param({**CONFIG, "foo": "bar"}, "foo", id="extra_field"),
        pytest.param({**CONFIG, "cuda": 13.0}, "quoted", id="unquoted_version"),
        pytest.param({**CONFIG, "torch": "latest"}, "torch", id="bad_version_format"),
        pytest.param({**CONFIG, "gpu": ""}, "gpu", id="empty_gpu"),
        pytest.param([CONFIG], "mapping", id="not_a_mapping"),
    ],
)
def test_schema_violations(tmp_path, caplog, env_files, good_model, data, fragment):
    _, environments = env_files
    config = write_yaml(tmp_path / "bad.yaml", data)

    state = run(config, environments, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "validate_config"
    assert "schema" in state["customer_message"]
    assert fragment in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["validate_config"]


def test_broken_registry(tmp_path, caplog, env_files, good_model):
    config, _ = env_files
    environments = write_yaml(tmp_path / "registry.yaml", {"environments": [{"name": "x"}]})

    state = run(config, environments, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "load_environments"
    assert "support" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["load_environments"]


@pytest.mark.parametrize("field", sorted(ENVIRONMENT))
def test_mismatch_on_any_field_is_rejected(tmp_path, caplog, env_files, good_model, field):
    _, environments = env_files
    config = write_yaml(tmp_path / "mismatch.yaml", {**CONFIG, field: "9.9.9"})

    state = run(config, environments, good_model)

    assert state["outcome"] == "failure"
    assert state["error"] == "match_environment"
    assert "rtx5090-cuda130" in state["customer_message"]
    assert "9.9.9" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["match_environment"]


def test_main_accepts_environments_override(env_files, good_model, capsys):
    config, environments = env_files
    model, npz = good_model

    code = main([str(config), str(model), str(npz), "--environments", str(environments)])

    assert code == 0
    assert "succeeded" in capsys.readouterr().out

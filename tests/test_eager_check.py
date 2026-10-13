import importlib.metadata
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import CONFIG, write_yaml
from kronos_edge import eager_check
from kronos_edge.workflow import main, route_on_error, run_check

CONV_RELU = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""

TYPO = CONV_RELU.replace("self.relu(self.conv(x))", "self.relo(self.conv(x))")
FWD_ARGS_ALIAS = CONV_RELU.replace("def get_inputs", "def get_fwd_args")
NO_MODEL_CLASS = CONV_RELU.replace("class Model(", "class Net(")
NO_GET_INPUTS = CONV_RELU.split("def get_inputs")[0]
GET_INPUTS_RAISES = CONV_RELU.replace(
    "return [torch.randn(1, 3, 8, 8)]", 'raise RuntimeError("boom")'
)

MODEL_NOT_MODULE = """
import torch


class Model:
    pass


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""

INIT_INPUTS = """
import torch


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.conv(x)


def get_init_inputs():
    return [3, 4]


def get_inputs():
    return [torch.randn(1, 3, 8, 8)]
"""
INIT_INPUTS_MISSING = INIT_INPUTS.replace("def get_init_inputs():\n    return [3, 4]\n", "")

TWO_INPUTS = """
import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.randn(2, 3), torch.randn(2, 3)]
"""

SYNTAX_ERROR = "def (:\n"
MISSING_PACKAGE = "import kronos_no_such_package\n" + CONV_RELU
USES_DEPENDENCY = "import kronos_fake_dependency\n" + CONV_RELU

X = np.zeros((1, 3, 8, 8), dtype=np.float32)
XY = [np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]


def write_model(tmp_path, source):
    path = tmp_path / "model.py"
    if source is not None:
        path.write_text(source)
    return path


def write_npz(tmp_path, arrays):
    path = tmp_path / "input.npz"
    if isinstance(arrays, dict):
        np.savez(path, **arrays)
    elif arrays is not None:
        np.savez(path, *arrays)
    return path


def error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith("kronos_edge")]


def run(env_files, model, npz):
    config, environments = env_files
    return run_check(config, model, npz, environments_path=environments)


@pytest.mark.parametrize(
    "source, arrays",
    [
        pytest.param(CONV_RELU, [X], id="conv_relu"),
        pytest.param(FWD_ARGS_ALIAS, [X], id="get_fwd_args_alias"),
        pytest.param(INIT_INPUTS, [X], id="get_init_inputs"),
        pytest.param(TWO_INPUTS, XY, id="two_inputs"),
    ],
)
def test_success(tmp_path, caplog, env_files, source, arrays):
    state = run(env_files, write_model(tmp_path, source), write_npz(tmp_path, arrays))

    assert state["outcome"] == "success"
    assert state.get("error") is None
    assert "succeeded" in state["customer_message"]
    assert error_records(caplog) == []


@pytest.mark.parametrize(
    "source, arrays, stage, fragment",
    [
        pytest.param(TYPO, [X], "forward", "cannot be run through eager mode", id="forward_typo"),
        pytest.param(None, [X], "import_module", "could not be loaded", id="model_file_missing"),
        pytest.param(SYNTAX_ERROR, [X], "import_module", "could not be loaded", id="syntax_error"),
        pytest.param(MISSING_PACKAGE, [X], "missing_dependency", "dependencies", id="missing_package"),
        pytest.param(NO_MODEL_CLASS, [X], "find_model_class", "does not define a 'Model' class", id="no_model_class"),
        pytest.param(MODEL_NOT_MODULE, [X], "find_model_class", "does not define a 'Model' class", id="model_not_module"),
        pytest.param(NO_GET_INPUTS, [X], "find_get_inputs", "get_inputs() or get_fwd_args()", id="no_get_inputs"),
        pytest.param(INIT_INPUTS_MISSING, [X], "init_model", "get_init_inputs", id="init_inputs_missing"),
        pytest.param(GET_INPUTS_RAISES, [X], "get_inputs", "boom", id="get_inputs_raises"),
        pytest.param(CONV_RELU, None, "load_npz", "could not be read", id="npz_file_missing"),
        pytest.param(TWO_INPUTS, [X], "validate_inputs", "arr_1", id="npz_wrong_count"),
        pytest.param(CONV_RELU, {"x": X}, "validate_inputs", "arr_0", id="npz_wrong_key"),
        pytest.param(CONV_RELU, [np.zeros((1, 3, 9, 9), dtype=np.float32)], "validate_inputs", "shape", id="shape_mismatch"),
        pytest.param(CONV_RELU, [np.zeros((1, 3, 8, 8), dtype=np.float64)], "validate_inputs", "dtype", id="dtype_mismatch"),
    ],
)
def test_failure(tmp_path, caplog, env_files, source, arrays, stage, fragment):
    state = run(env_files, write_model(tmp_path, source), write_npz(tmp_path, arrays))

    assert state["outcome"] == "failure"
    assert state["error"] == stage
    assert fragment in state["customer_message"]
    records = error_records(caplog)
    assert len(records) == 1
    assert records[0].stage == stage


def test_missing_package_names_the_module(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, MISSING_PACKAGE), write_npz(tmp_path, [X]))

    assert "kronos_no_such_package" in state["customer_message"]
    assert "config.yaml" in state["customer_message"]


def test_dependencies_are_installed_before_model_py_is_imported(tmp_path, monkeypatch, env_files, pip_stub):
    """The model may import a package that only exists once pip has run."""
    config, environments = env_files
    config = write_yaml(tmp_path / "deps.yaml", {**CONFIG, "dependencies": ["kronos-fake-dependency"]})
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(site)

    def fake_install(packages):
        pip_stub.append(list(packages))
        (site / "kronos_fake_dependency.py").write_text("")

    monkeypatch.setattr(eager_check, "install", fake_install)

    state = run_check(config, write_model(tmp_path, USES_DEPENDENCY), write_npz(tmp_path, [X]), environments_path=environments)

    assert state["outcome"] == "success"
    assert pip_stub == [["kronos-fake-dependency"]]


def test_failed_install_is_its_own_stage(tmp_path, caplog, monkeypatch, env_files):
    config, environments = env_files
    config = write_yaml(tmp_path / "deps.yaml", {**CONFIG, "dependencies": ["kronos-fake-dependency"]})

    def failing_install(packages):
        raise RuntimeError("pip exited 1\nERROR: No matching distribution found for kronos-fake-dependency")

    monkeypatch.setattr(eager_check, "install", failing_install)

    state = run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)

    assert state["outcome"] == "failure"
    assert state["error"] == "install_dependencies"
    assert "could not be installed" in state["customer_message"]
    assert "No matching distribution" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["install_dependencies"]


def constraints_in(cmd):
    """The pins pip was handed through --constraint, as {name: version}; read while the file still exists."""
    lines = Path(cmd[cmd.index("--constraint") + 1]).read_text().splitlines()
    return dict(line.split("==") for line in lines)


def resolving_pip(requires):
    """A pip that honors --constraint the way the real resolver does.

    ``requires`` is what the requested package pulls in transitively. A requirement that disagrees with a pin
    is a conflict and installs nothing; without a pin it installs, replacing whatever was there (recorded).
    """
    replaced = []

    def fake_run(cmd, **kwargs):
        pins = constraints_in(cmd) if "--constraint" in cmd else {}
        for name, version in requires.items():
            if name in pins and pins[name] != version:
                stderr = (
                    "ERROR: Cannot install kronos-fake-dependency because these package versions have conflicting "
                    f"dependencies.\nThe user requested (constraint) {name}=={pins[name]}\n"
                    f"kronos-fake-dependency 1.0 depends on {name}=={version}\n"
                )
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)
            replaced.append(f"{name}=={version}")
        return subprocess.CompletedProcess(cmd, 0, stdout="Successfully installed kronos-fake-dependency", stderr="")

    return fake_run, replaced


def test_install_runs_pip_in_this_interpreter(monkeypatch, pip_stub):
    """The real worker: pip of the container's venv, output captured so stdout stays the user's."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="Successfully installed timm", stderr="")

    monkeypatch.setattr(eager_check.subprocess, "run", fake_run)

    pip_stub.real(["timm", "einops==0.8.0"])

    assert seen["cmd"][:5] == [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    assert seen["cmd"][5] == "--constraint"
    assert seen["cmd"][7:] == ["timm", "einops==0.8.0"]
    assert seen["kwargs"].get("capture_output") is True


def test_install_pins_every_runtime_package(monkeypatch, pip_stub):
    """pip is constrained to the venv as it is, torch with its build label included, so none of it can be replaced."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["path"], seen["pins"] = cmd[cmd.index("--constraint") + 1], constraints_in(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(eager_check.subprocess, "run", fake_run)

    pip_stub.real(["timm"])

    assert seen["pins"]["torch"] == torch.__version__  # 2.x+cpu here, 2.x+cu130 in the image
    assert seen["pins"] == {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    assert not Path(seen["path"]).exists()


def test_transitive_dependency_that_would_replace_the_runtime_fails_at_install_dependencies(
    tmp_path, caplog, monkeypatch, env_files, pip_stub
):
    """A requirement the user's package pulls in must not replace torch or anything else the image carries:
    the install is pinned to the venv's versions, so pip reports the conflict and installs nothing."""
    config, environments = env_files
    config = write_yaml(tmp_path / "deps.yaml", {**CONFIG, "dependencies": ["kronos-fake-dependency"]})
    fake_pip, replaced = resolving_pip({"torch": "99.0.0"})
    monkeypatch.setattr(eager_check.subprocess, "run", fake_pip)
    monkeypatch.setattr(eager_check, "install", pip_stub.real)

    state = run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)

    assert replaced == []
    assert state["outcome"] == "failure"
    assert state["error"] == "install_dependencies"
    assert "torch==99.0.0" in state["customer_message"]
    assert f"torch=={torch.__version__}" in state["customer_message"]
    assert [r.stage for r in error_records(caplog)] == ["install_dependencies"]


def test_dependency_compatible_with_the_runtime_installs_under_the_pins(tmp_path, monkeypatch, env_files, pip_stub):
    """The pins only forbid replacing the runtime: a package that needs the installed torch still installs."""
    config, environments = env_files
    config = write_yaml(tmp_path / "deps.yaml", {**CONFIG, "dependencies": ["kronos-fake-dependency"]})
    fake_pip, replaced = resolving_pip({"torch": torch.__version__})
    monkeypatch.setattr(eager_check.subprocess, "run", fake_pip)
    monkeypatch.setattr(eager_check, "install", pip_stub.real)

    state = run_check(config, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X]), environments_path=environments)

    assert state["outcome"] == "success"
    assert replaced == [f"torch=={torch.__version__}"]


def test_install_failure_carries_pips_output(monkeypatch, pip_stub):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: No matching distribution found for nope")

    monkeypatch.setattr(eager_check.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="No matching distribution"):
        pip_stub.real(["nope"])


def test_forward_failure_includes_exception_text(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X]))

    assert "relo" in state["customer_message"]


def test_init_inputs_missing_names_required_arguments(tmp_path, env_files):
    state = run(env_files, write_model(tmp_path, INIT_INPUTS_MISSING), write_npz(tmp_path, [X]))

    assert "in_channels" in state["customer_message"]
    assert "out_channels" in state["customer_message"]


def test_concurrent_runs_do_not_share_failure_details(tmp_path, env_files):
    typo_dir, no_model_dir = tmp_path / "typo", tmp_path / "no_model"
    typo_dir.mkdir()
    no_model_dir.mkdir()
    jobs = [
        (write_model(typo_dir, TYPO), write_npz(typo_dir, [X])),
        (write_model(no_model_dir, NO_MODEL_CLASS), write_npz(no_model_dir, [X])),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        typo_state, no_model_state = list(pool.map(lambda job: run(env_files, *job), jobs))

    assert typo_state["error"] == "forward"
    assert "relo" in typo_state["customer_message"]
    assert no_model_state["error"] == "find_model_class"
    assert "relo" not in no_model_state["customer_message"]


def test_route_on_error():
    assert route_on_error({"error": "forward"}) == "failure"
    assert route_on_error({}) == "ok"


def cli_args(env_files, model, npz):
    config, environments = env_files
    return [str(config), str(model), str(npz), "--environments", str(environments)]


def test_main_success_exit_code(tmp_path, env_files, capsys):
    code = main(cli_args(env_files, write_model(tmp_path, CONV_RELU), write_npz(tmp_path, [X])))

    assert code == 0
    assert "succeeded" in capsys.readouterr().out


def test_main_failure_exit_code(tmp_path, env_files, capsys):
    code = main(cli_args(env_files, write_model(tmp_path, TYPO), write_npz(tmp_path, [X])))

    assert code == 1
    assert "cannot be run through eager mode" in capsys.readouterr().out

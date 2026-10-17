from datetime import datetime, timedelta, timezone

from kronos_edge import artifacts, benchmark_check, kernelgen_check, onnx_check, onnxsim_check, tensorrt_check


def test_managed_names_agree_with_the_nodes_that_write_them():
    assert onnx_check.EXPORTED_FILE in artifacts.MANAGED_FILES
    assert onnxsim_check.SIMPLIFIED_FILE in artifacts.MANAGED_FILES
    assert tensorrt_check.ENGINE_FILE in artifacts.MANAGED_FILES
    assert benchmark_check.BENCHMARK_FILE in artifacts.MANAGED_FILES
    assert {kernelgen_check.PT2_FILE, kernelgen_check.KERNEL_FILE, kernelgen_check.REPORT_FILE} <= set(artifacts.MANAGED_FILES)


def test_a_run_directory_is_named_for_the_utc_time_it_starts(tmp_path, monkeypatch):
    """A clock in another zone is converted: the name is always UTC."""
    started = datetime(2026, 10, 17, 16, 30, 5, tzinfo=timezone(timedelta(hours=2)))
    monkeypatch.setattr(artifacts, "utc_now", lambda: started)

    run = artifacts.create_run_directory(tmp_path / "root")

    assert run == tmp_path / "root" / "20261017T143005Z"
    assert run.is_dir() and list(run.iterdir()) == []


def test_the_clock_is_utc():
    assert artifacts.utc_now().utcoffset() == timedelta(0)


def test_runs_starting_in_the_same_second_get_their_own_directories(tmp_path, monkeypatch):
    """A taken name is never reused, whether an earlier run or something else holds it."""
    monkeypatch.setattr(artifacts, "utc_now", lambda: datetime(2026, 10, 17, 14, 30, 5, tzinfo=timezone.utc))
    first = artifacts.create_run_directory(tmp_path)
    (first / "model.onnx").write_bytes(b"first run")
    (tmp_path / "20261017T143005Z-2").write_text("not a run")

    second = artifacts.create_run_directory(tmp_path)
    third = artifacts.create_run_directory(tmp_path)

    assert (first.name, second.name, third.name) == ("20261017T143005Z", "20261017T143005Z-3", "20261017T143005Z-4")
    assert (first / "model.onnx").read_bytes() == b"first run"
    assert (tmp_path / "20261017T143005Z-2").read_text() == "not a run"
    assert list(second.iterdir()) == [] and list(third.iterdir()) == []


def test_a_missing_root_is_created(tmp_path):
    run = artifacts.create_run_directory(tmp_path / "a" / "b")

    assert run.parent == tmp_path / "a" / "b" and run.is_dir()


def test_nothing_is_cleared_any_more():
    """Every run writes to a directory it just created, so there is nothing stale to delete."""
    assert not hasattr(artifacts, "clear")
    assert not hasattr(artifacts, "RETIRED_FILES")

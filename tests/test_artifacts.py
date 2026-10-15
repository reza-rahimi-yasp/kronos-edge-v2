from kronos_edge import artifacts, benchmark_check, container, kernelgen_check, onnx_check, onnxsim_check, tensorrt_check


def test_managed_names_agree_with_the_nodes_that_write_them():
    assert onnx_check.EXPORTED_FILE in artifacts.MANAGED_FILES
    assert onnxsim_check.SIMPLIFIED_FILE in artifacts.MANAGED_FILES
    assert tensorrt_check.ENGINE_FILE in artifacts.MANAGED_FILES
    assert benchmark_check.BENCHMARK_FILE in artifacts.MANAGED_FILES
    assert container.PT2_FILE in artifacts.MANAGED_FILES  # written by the export container
    assert {kernelgen_check.KERNEL_FILE, kernelgen_check.REPORT_FILE} <= set(artifacts.MANAGED_FILES)


def test_clear_removes_managed_files_and_companions_only(tmp_path):
    for name in artifacts.MANAGED_FILES:
        (tmp_path / name).write_bytes(b"x")
        (tmp_path / f"{name}.data").write_bytes(b"x")
    (tmp_path / "keep.onnx").write_bytes(b"x")
    (tmp_path / "sub").mkdir()

    artifacts.clear(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.onnx", "sub"]


def test_clear_tolerates_a_missing_directory(tmp_path):
    artifacts.clear(tmp_path / "absent")


def test_clear_removes_the_retired_evaluation_report_of_an_older_version(tmp_path):
    """Earlier versions wrote kernelgen_eval.json; an upgraded artifact directory must not keep it next to fresh files."""
    for name in (*artifacts.MANAGED_FILES, "kernelgen_eval.json"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"mine")

    artifacts.clear(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_retired_files_are_cleared_but_never_produced():
    assert "kernelgen_eval.json" in artifacts.RETIRED_FILES
    assert not set(artifacts.RETIRED_FILES) & set(artifacts.MANAGED_FILES)

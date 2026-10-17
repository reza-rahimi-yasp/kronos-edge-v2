"""The Python benchmark's verdict: a failing post-timing check keeps its diagnostic report and fails the run."""

# The exporter modules are plain scripts, not a package: make them importable from this test alone.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))

import json
import re
from pathlib import Path

import pytest

import checks

EXPERIMENT = Path(__file__).resolve().parents[1]

GOOD = {"shape": [4, 256, 34, 60], "finite": True, "cosine": 0.99999988, "max_abs_diff": 1.3e-6, "allclose_rtol1e-4_atol1e-5": True, "bitwise_equal": False}


def _result(after):
    return {
        "setting": "notf32",
        "checks_before": {"gaia_optimized": dict(GOOD), "hybrid_trt": dict(GOOD), "pure_trt": dict(GOOD)},
        "trials": [{"order": ["gaia_optimized", "hybrid_trt", "pure_trt"], "stats_ms": {}, "samples_ms": {}}],
        "checks_after": {"gaia_optimized": dict(GOOD), "hybrid_trt": after, "pure_trt": dict(GOOD)},
    }


@pytest.mark.parametrize(
    "after, problem",
    [
        pytest.param({**GOOD, "cosine": 0.5}, "hybrid_trt.*after.*cosine", id="below_threshold_after_timing"),
        pytest.param({**GOOD, "finite": False, "cosine": float("nan")}, "hybrid_trt.*after.*finite", id="non_finite_after_timing"),
    ],
)
def test_passing_before_and_failing_after_fails_but_keeps_the_report(tmp_path, after, problem):
    report = tmp_path / "benchmark_python_notf32.json"
    result = _result(after)
    passed = checks.finish_report(report, result, threshold=0.99)
    assert passed is False
    written = json.loads(report.read_text())
    assert written["all_checks_passed"] is False
    assert written["checks_after"]["hybrid_trt"]["finite"] == after["finite"]
    assert written["trials"] == result["trials"]  # the timing evidence is preserved alongside the verdict
    assert any(re.search(problem, p) for p in written["check_problems"]), written["check_problems"]


def test_all_passing_is_reported_as_passed(tmp_path):
    report = tmp_path / "benchmark_python_notf32.json"
    assert checks.finish_report(report, _result(dict(GOOD)), threshold=0.99) is True
    written = json.loads(report.read_text())
    assert written["all_checks_passed"] is True and written["check_problems"] == []


def test_benchmark_script_exits_nonzero_on_a_failed_verdict():
    source = (EXPERIMENT / "exporter" / "benchmark_python.py").read_text()
    assert "checks.finish_report(" in source
    verdict = source.index("checks.finish_report(")
    assert re.search(r"raise SystemExit\(", source[verdict:]), "the script must exit nonzero after writing a failed report"

"""Verdict of the Python benchmark's output checks (stdlib only, so the CPU tests can exercise it).

A check is a dict with ``finite`` and ``cosine`` (as benchmark_python.py
records before and after timing). ``problems`` names every subject whose
check is non-finite or below the threshold, in either phase.
``finish_report`` writes the full result, with the verdict and the problems,
then returns whether everything passed: the caller exits nonzero on False
and the report is preserved for diagnosis either way.
"""

import json
import math
from pathlib import Path


def problems(result: dict, threshold: float) -> list[str]:
    out = []
    for phase in ("checks_before", "checks_after"):
        for subject, check in result.get(phase, {}).items():
            cosine = check.get("cosine")
            if not check.get("finite", False) or cosine is None or not math.isfinite(cosine):
                out.append(f"{subject} {phase.removeprefix('checks_')} timing: output is not finite (cosine {cosine})")
            elif cosine < threshold:
                out.append(f"{subject} {phase.removeprefix('checks_')} timing: cosine {cosine:.10f} is below the threshold {threshold}")
    return out


def finish_report(path: Path, result: dict, threshold: float) -> bool:
    found = problems(result, threshold)
    result["check_problems"] = found
    result["all_checks_passed"] = not found
    Path(path).write_text(json.dumps(result, indent=2))
    return not found

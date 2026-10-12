import subprocess
import sys
from pathlib import Path

import pytest

from kronos_edge import container, eager_check, env_check, messages, workflow

SRC = Path(__file__).resolve().parents[1] / "src"


def test_host_imports_without_torch_or_numpy():
    """The host only validates config and launches containers; it must not need the model stack."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['numpy'] = None\n"
        "import kronos_edge.workflow, kronos_edge.container, kronos_edge.env_check, kronos_edge.state\n"
        "print('host ok')\n"
    )

    result = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": str(SRC)}, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "host ok" in result.stdout


def test_merge_rejects_duplicate_stage_names():
    with pytest.raises(ValueError, match="load"):
        messages.merge({"load": "a"}, {"load": "b"})


def test_merged_table_is_the_union_of_every_check():
    tables = (env_check.CUSTOMER_MESSAGES, container.CUSTOMER_MESSAGES, eager_check.CUSTOMER_MESSAGES)

    assert workflow.CUSTOMER_MESSAGES == {k: v for t in tables for k, v in t.items()}
    assert len(workflow.CUSTOMER_MESSAGES) == sum(len(t) for t in tables)


def test_every_stage_used_in_source_has_a_message():
    """Every `stage = "..."` assignment in a check module must be a key of the merged table."""
    import re

    used = set()
    for module in SRC.glob("*/*.py"):
        used.update(re.findall(r'stage = "([a-z_]+)"', module.read_text()))

    assert used, "no stage assignments found"
    assert used <= set(workflow.CUSTOMER_MESSAGES)

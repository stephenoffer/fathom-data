"""Run every example.

Documentation that is not executed rots. These run the scripts as scripts — not by
importing selected functions — so a broken example fails CI the same way broken code
does. Each one asserts its own claims internally; this only checks they complete.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("[0-9]*.py"))


def test_examples_exist():
    assert EXAMPLES, "no examples found; the glob or the directory moved"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script: Path):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{script.name} failed\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), f"{script.name} printed nothing"

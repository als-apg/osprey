"""Every shell script under ``scripts/ci/`` passes shellcheck.

ci.yml's ``run:`` blocks are shellchecked through actionlint; a program moved
out of the workflow into a script would otherwise leave that coverage behind.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_CI_SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "scripts" / "ci").glob("*.sh"))


@pytest.mark.parametrize("script", _CI_SCRIPTS, ids=lambda path: path.name)
def test_shellcheck_passes(script: Path) -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed (uv sync --extra dev installs shellcheck-py)")
    done = subprocess.run([shellcheck, str(script)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout

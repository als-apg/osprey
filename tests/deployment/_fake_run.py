"""Stand-ins for ``subprocess.run`` results in tests that fake the host."""

from __future__ import annotations

import subprocess


def completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    """A finished child's result, for a fake ``subprocess.run`` to return."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)

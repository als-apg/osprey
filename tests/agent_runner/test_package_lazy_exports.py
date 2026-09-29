"""Unit tests for the import cost of the :mod:`osprey.agent_runner` package.

The package is the agent harness adapter. Most of its modules drive an agent
through the agent SDK, but some serve callers that never drive one. The tests
here hold the invariant that importing the package, or any one module of it,
costs only that module: the agent SDK loads only when a name that needs it is
first read.

The import checks run in a fresh interpreter, not on the already-populated
``sys.modules`` of the test session.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _modules_added_by_import(module: str) -> set[str]:
    """Return the module names a fresh import of ``module`` adds to ``sys.modules``.

    Args:
        module: Dotted name to import in a child interpreter.

    Returns:
        Every module name the import added, stdlib included.

    Raises:
        AssertionError: If the child interpreter fails to import the module.
    """
    code = (
        "import json, sys;"
        "before = set(sys.modules);"
        f"import {module};"
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )
    assert result.returncode == 0, f"fresh import of {module} failed:\n{result.stderr}"
    return set(json.loads(result.stdout))


def test_importing_the_package_loads_no_module_of_it_and_no_agent_sdk():
    """The package root resolves its exports on access, so it imports no module."""
    added = _modules_added_by_import("osprey.agent_runner")

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]
    assert not [name for name in added if name.startswith("osprey.agent_runner.")]

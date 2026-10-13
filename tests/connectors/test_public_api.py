"""The connectors package's namespaces hand out what their ``__all__`` declares.

``control_system``, ``archiver``, ``simulation`` and ``ipc`` gather
underscore-private implementation modules behind one ``__init__``. The other
public modules of the package are their own API and list it in their own
``__all__``. Which modules the framework may import is not checked here:
``lint-imports`` holds that (``[tool.importlinter]`` in the root
``pyproject.toml``).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "packages" / "osprey-connectors" / "src"

NAMESPACES = (
    "osprey_connectors.control_system",
    "osprey_connectors.archiver",
    "osprey_connectors.simulation",
    "osprey_connectors.ipc",
)

#: The public modules that are their own API.
MODULES = tuple(
    f"osprey_connectors.{path.stem}"
    for path in sorted((PACKAGE / "osprey_connectors").glob("*.py"))
    if not path.stem.startswith("_")
) + ("osprey_connectors.control_system.write_door",)


@pytest.mark.parametrize("module", NAMESPACES + MODULES)
def test_every_listed_name_is_there(module: str):
    loaded = importlib.import_module(module)

    assert loaded.__all__, f"{module} lists nothing"
    missing = [name for name in loaded.__all__ if not hasattr(loaded, name)]
    assert not missing, f"{module} lists names it does not define: {missing}"


def test_the_lazy_namespace_maps_exactly_what_it_lists():
    ipc = importlib.import_module("osprey_connectors.ipc")

    assert set(ipc.__all__) == set(ipc._LAZY_EXPORTS)
    for name, module in ipc._LAZY_EXPORTS.items():
        assert getattr(ipc, name) is getattr(importlib.import_module(module), name)


def test_importing_the_ipc_namespace_loads_none_of_its_modules():
    # The connector-host child runs as `python -m osprey_connectors.ipc._host`;
    # a package that imported `_host` itself would load it twice.
    code = (
        "import sys, osprey_connectors.ipc;"
        "loaded = sorted(m for m in sys.modules if m.startswith('osprey_connectors.ipc._'));"
        "assert not loaded, loaded;"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(PACKAGE)),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout

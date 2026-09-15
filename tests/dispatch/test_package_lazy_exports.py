"""Unit tests for the import cost of the :mod:`osprey.dispatch` package.

The package gathers four unrelated pieces of the dispatcher — the pool, the
trigger registry, the trigger-configuration reader and the HTTP worker client.
Only the last of those needs an HTTP stack, so re-exporting the four eagerly
would make the cheapest leaf the most expensive import in the tree: reading
``triggers.yml`` would pull ``httpx`` in behind it.

The tests here hold the invariant that importing the package, or any one leaf
of it, costs only that leaf, and that the lazy map behind the public names
cannot drift from the modules that define them.

The import checks run in a fresh interpreter, not on the already-populated
``sys.modules`` of the test session.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import osprey.dispatch

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


def test_reading_trigger_configuration_costs_no_http_stack():
    """``trigger_config`` needs a YAML parser; it must not drag ``httpx`` in.

    The assertion names ``httpx`` rather than demanding that nothing new
    appears at all: the reader legitimately imports ``yaml``.
    """
    added = _modules_added_by_import("osprey.dispatch.trigger_config")

    assert not [name for name in added if name.split(".")[0] == "httpx"]


def test_importing_the_package_pulls_in_neither_the_client_nor_the_pool():
    """The package root resolves its exports on access, so it imports no leaf."""
    added = _modules_added_by_import("osprey.dispatch")

    assert not [name for name in added if name.split(".")[0] == "httpx"]
    assert "osprey.dispatch.pool" not in added


@pytest.mark.parametrize("name", sorted(osprey.dispatch.__all__))
def test_every_public_name_resolves_to_its_defining_module(name):
    """Attribute access yields the object the submodule exports, not a stale copy."""
    from importlib import import_module

    value = getattr(osprey.dispatch, name)
    module = import_module(osprey.dispatch._LAZY_EXPORTS[name], osprey.dispatch.__name__)

    assert value is getattr(module, name)


def test_an_unknown_name_raises_attribute_error():
    """The lazy lookup refuses names the package does not export."""
    with pytest.raises(AttributeError):
        osprey.dispatch.no_such_export

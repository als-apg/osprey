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


@pytest.mark.parametrize("name", sorted(osprey.dispatch._LAZY_EXPORTS))
def test_every_public_name_resolves_to_its_defining_module(name):
    """Attribute access yields the object the submodule exports, not a stale copy."""
    from importlib import import_module

    value = getattr(osprey.dispatch, name)
    module = import_module(osprey.dispatch._LAZY_EXPORTS[name], osprey.dispatch.__name__)

    assert value is getattr(module, name)


def test_public_names_are_the_lazy_map_plus_the_package_constants():
    """``__all__`` advertises exactly the two kinds of name the package resolves.

    A name reaching ``__all__`` without an entry in one of the two sets would
    be advertised and then unreachable; a name in either set that never reaches
    ``__all__`` is resolvable but undiscoverable. Pinning the partition catches
    both directions, which is what keeps the parametrized test above a complete
    check of the lazy half rather than a sample of it.
    """
    assert sorted(osprey.dispatch.__all__) == sorted(
        [*osprey.dispatch._LAZY_EXPORTS, *osprey.dispatch._EAGER_EXPORTS]
    )
    for name in osprey.dispatch._EAGER_EXPORTS:
        assert hasattr(osprey.dispatch, name)


def test_the_dispatcher_mcp_path_is_the_transport_path_of_the_mcp_server():
    """The one spelling of the dispatcher's MCP path, shared by both ends.

    The proxy appends it to the dispatcher's base URL and the dispatcher's
    compose environment hands it to FastMCP as ``FASTMCP_STREAMABLE_HTTP_PATH``.
    The two ends can only meet on a value both read, so the value is pinned
    here and the render test pins the compose spelling against it.
    """
    assert osprey.dispatch.DISPATCHER_MCP_PATH == "/mcp"
    assert osprey.dispatch.DISPATCHER_MCP_PATH.startswith("/")


def test_reading_the_mcp_path_costs_no_leaf_of_the_package():
    """The constant is eager, so reading it must still not load a submodule.

    It is read by the proxy route, which has no business paying for the HTTP
    worker client or the pool to learn a string.
    """
    code = (
        "import json, sys;"
        "import osprey.dispatch as d;"
        "assert d.DISPATCHER_MCP_PATH;"
        "print(json.dumps(sorted(n for n in sys.modules if n.startswith('osprey.dispatch.'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_an_unknown_name_raises_attribute_error():
    """The lazy lookup refuses names the package does not export."""
    with pytest.raises(AttributeError):
        osprey.dispatch.no_such_export

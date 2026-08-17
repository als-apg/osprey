"""Tests for the capabilities MCP tool."""

import json

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.registry import get_registry
from tests.mcp_server.ariel.conftest import get_tool_fn


def _get_capabilities():
    from osprey.mcp_server.ariel.tools.capabilities import capabilities

    return get_tool_fn(capabilities)


def _setup_registry(tmp_path, monkeypatch, search_modules=None):
    """Write a config, initialize the framework registry and the ARIEL context.

    The framework registry must be initialized because ``capabilities`` now
    advertises the modes the registry actually carries, not a hardcoded list.

    Args:
        tmp_path: Temporary working directory fixture.
        monkeypatch: Pytest monkeypatch fixture, used to chdir.
        search_modules: Optional ``search_modules`` config block. Defaults to
            keyword and semantic both enabled.
    """
    monkeypatch.chdir(tmp_path)
    if search_modules is None:
        search_modules = {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": "nomic-embed-text"},
        }
    config = json.dumps(
        {
            "ariel": {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": search_modules,
            }
        }
    )
    (tmp_path / "config.yml").write_text(config)
    get_registry().initialize()
    initialize_ariel_context()


@pytest.mark.unit
async def test_capabilities_returns_modules(tmp_path, monkeypatch):
    """Capabilities returns enabled search modules."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert not data.get("error", False)
    assert "keyword" in data["enabled_search_modules"]
    assert "semantic" in data["enabled_search_modules"]


@pytest.mark.unit
async def test_capabilities_includes_search_modes(tmp_path, monkeypatch):
    """Capabilities advertises every registered, enabled search module."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_omits_sql_query_mode(tmp_path, monkeypatch):
    """``sql_query`` is a tool, not a mode, so it never appears in the mode list."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "sql_query" not in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_omits_disabled_modes(tmp_path, monkeypatch):
    """A registered module that config disables is not advertised as a mode."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        search_modules={
            "keyword": {"enabled": True},
            "semantic": {"enabled": False},
        },
    )

    fn = _get_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" not in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_no_registry_import():
    """Capabilities does NOT import from osprey.registry (main framework)."""
    import ast
    import inspect

    from osprey.mcp_server.ariel.tools import capabilities

    # Read the source file directly rather than via inspect.getsource, whose
    # linecache/bytecode-lineno slicing can drift under a transient .py/.pyc skew.
    source_path = inspect.getsourcefile(capabilities) or inspect.getfile(capabilities)
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("osprey.registry"), (
                    "capabilities must NOT import from osprey.registry"
                )

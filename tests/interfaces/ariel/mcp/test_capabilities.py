"""Tests for the ariel_capabilities MCP tool."""

import json

import pytest
from unittest.mock import patch

from tests.interfaces.ariel.mcp.conftest import get_tool_fn
from osprey.interfaces.ariel.mcp.registry import initialize_ariel_registry


def _get_ariel_capabilities():
    from osprey.interfaces.ariel.mcp.tools.capabilities import ariel_capabilities
    return get_tool_fn(ariel_capabilities)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = json.dumps({
        "ariel": {
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "nomic-embed-text"},
            },
            "pipelines": {
                "rag": {"enabled": True},
                "agent": {"enabled": True},
            },
            "reasoning": {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "max_iterations": 5,
                "temperature": 0.1,
            },
            "default_max_results": 15,
        }
    })
    (tmp_path / "config.yml").write_text(config)
    initialize_ariel_registry()


@pytest.mark.unit
async def test_capabilities_returns_modules(tmp_path, monkeypatch):
    """Capabilities returns enabled modules and pipelines."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_ariel_capabilities()
    result = await fn()

    data = json.loads(result)
    assert not data.get("error", False)
    assert "keyword" in data["enabled_search_modules"]
    assert "semantic" in data["enabled_search_modules"]
    assert "rag" in data["enabled_pipelines"]
    assert "agent" in data["enabled_pipelines"]
    assert data["default_max_results"] == 15


@pytest.mark.unit
async def test_capabilities_includes_search_modes(tmp_path, monkeypatch):
    """Capabilities includes all search mode enum values."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_ariel_capabilities()
    result = await fn()

    data = json.loads(result)
    assert "keyword" in data["search_modes"]
    assert "semantic" in data["search_modes"]
    assert "rag" in data["search_modes"]
    assert "agent" in data["search_modes"]


@pytest.mark.unit
async def test_capabilities_includes_reasoning(tmp_path, monkeypatch):
    """Capabilities includes reasoning configuration."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_ariel_capabilities()
    result = await fn()

    data = json.loads(result)
    assert data["reasoning"]["provider"] == "openai"
    assert data["reasoning"]["model_id"] == "gpt-4o-mini"


@pytest.mark.unit
async def test_capabilities_no_registry_import():
    """Capabilities does NOT import from osprey.registry (main framework)."""
    import ast
    import inspect

    from osprey.interfaces.ariel.mcp.tools import capabilities

    source = inspect.getsource(capabilities)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("osprey.registry"), (
                    "ariel_capabilities must NOT import from osprey.registry"
                )

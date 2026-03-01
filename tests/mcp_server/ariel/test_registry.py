"""Tests for ARIEL MCP registry."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.registry import (
    ARIELMCPRegistry,
    get_ariel_registry,
    initialize_ariel_registry,
    reset_ariel_registry,
)


@pytest.mark.unit
def test_initialize_loads_config(tmp_path, monkeypatch):
    """Registry initialization loads ariel config from config.yml."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {"keyword": {"enabled": True}},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_registry()
    assert registry.config is not None
    assert registry.config.database.uri == "postgresql://localhost:5432/ariel"


@pytest.mark.unit
def test_initialize_connection_string_compat(tmp_path, monkeypatch):
    """Registry maps connection_string to uri for DatabaseConfig compatibility."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"connection_string": "postgresql://localhost:5432/ariel"},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_registry()
    assert registry.config.database.uri == "postgresql://localhost:5432/ariel"


@pytest.mark.unit
def test_initialize_missing_ariel_section(tmp_path, monkeypatch):
    """Missing ariel section warns but doesn't crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system: {}\n")

    registry = initialize_ariel_registry()
    with pytest.raises(RuntimeError, match="ARIEL config not available"):
        _ = registry.config


@pytest.mark.unit
def test_get_registry_before_init():
    """get_ariel_registry raises before initialization."""
    reset_ariel_registry()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_ariel_registry()


@pytest.mark.unit
def test_reset_clears_singleton():
    """reset_ariel_registry clears the singleton."""
    # Initialize first
    with patch.object(ARIELMCPRegistry, "_load_config", return_value={}):
        initialize_ariel_registry()

    # Should work
    get_ariel_registry()

    # Reset
    reset_ariel_registry()

    # Should fail now
    with pytest.raises(RuntimeError):
        get_ariel_registry()


@pytest.mark.unit
async def test_service_caches(tmp_path, monkeypatch):
    """service() creates the service once and caches it."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_registry()

    mock_service = AsyncMock()
    with patch(
        "osprey.services.ariel_search.service.create_ariel_service",
        new=AsyncMock(return_value=mock_service),
    ) as mock_create:
        svc1 = await registry.service()
        svc2 = await registry.service()

    assert svc1 is svc2
    assert mock_create.call_count == 1

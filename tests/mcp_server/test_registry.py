"""Tests for the MCP Server Registry.

Covers: initialization, singleton access, connector caching, invalidation,
config validation warnings, shutdown, channel_finder_config, and dot-path access.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.mcp_server.control_system.registry import (
    MCPRegistry,
    get_mcp_registry,
    initialize_mcp_registry,
    reset_mcp_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config_dict):
    """Write a config.yml to tmp_path and return the path."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(config_dict))
    return config_file


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initialize_loads_config(tmp_path, monkeypatch):
    """Registry reads config.yml and exposes sections."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "control_system": {"type": "mock", "writes_enabled": True},
            "archiver": {"type": "mock_archiver"},
            "channel_finder": {"db_path": "/data/channels.db"},
        },
    )

    registry = initialize_mcp_registry()

    assert registry.config.control_system["type"] == "mock"
    assert registry.config.archiver["type"] == "mock_archiver"
    assert registry.config.channel_finder["db_path"] == "/data/channels.db"
    assert registry.config.writes_enabled is True


@pytest.mark.unit
def test_initialize_missing_config(tmp_path, monkeypatch):
    """Registry initializes with empty config when config.yml is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    registry = initialize_mcp_registry()

    assert registry.config.raw == {}
    assert registry.config.control_system == {}
    assert registry.config.writes_enabled is False


@pytest.mark.unit
def test_initialize_idempotent(tmp_path, monkeypatch):
    """Calling initialize() multiple times is a no-op after the first."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    registry = MCPRegistry()
    registry.initialize()
    registry.initialize()  # Second call should be a no-op

    assert registry.config.control_system["type"] == "mock"


@pytest.mark.unit
def test_config_not_initialized_raises():
    """Accessing config before initialization raises RuntimeError."""
    registry = MCPRegistry()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = registry.config


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_access(tmp_path, monkeypatch):
    """get_mcp_registry() returns the same instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    initialize_mcp_registry()
    r1 = get_mcp_registry()
    r2 = get_mcp_registry()

    assert r1 is r2


@pytest.mark.unit
def test_get_before_initialize_raises():
    """get_mcp_registry() raises before initialize_mcp_registry()."""
    with pytest.raises(RuntimeError, match="not initialized"):
        get_mcp_registry()


@pytest.mark.unit
def test_reset_clears_singleton(tmp_path, monkeypatch):
    """reset_mcp_registry() clears the singleton so get raises again."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    initialize_mcp_registry()
    reset_mcp_registry()

    with pytest.raises(RuntimeError, match="not initialized"):
        get_mcp_registry()


# ---------------------------------------------------------------------------
# Connector caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connector_caching(tmp_path, monkeypatch):
    """Two calls to registry.control_system() return the same instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_mcp_registry()

    mock_connector = AsyncMock()
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        c1 = await registry.control_system()
        c2 = await registry.control_system()

    assert c1 is c2
    # Factory should only be called once (cached after first call)
    assert mock_connector is c1


@pytest.mark.unit
async def test_archiver_connector_caching(tmp_path, monkeypatch):
    """Two calls to registry.archiver() return the same instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"archiver": {"type": "mock_archiver"}})
    registry = initialize_mcp_registry()

    mock_connector = AsyncMock()
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        a1 = await registry.archiver()
        a2 = await registry.archiver()

    assert a1 is a2


# ---------------------------------------------------------------------------
# Connector invalidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connector_invalidation(tmp_path, monkeypatch):
    """After invalidate_connector(), next call creates a fresh instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_mcp_registry()

    mock_c1 = AsyncMock()
    mock_c2 = AsyncMock()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        side_effect=[mock_c1, mock_c2],
    ):
        c1 = await registry.control_system()
        await registry.invalidate_connector("control_system")
        c2 = await registry.control_system()

    assert c1 is not c2
    assert c1 is mock_c1
    assert c2 is mock_c2
    mock_c1.disconnect.assert_called_once()


@pytest.mark.unit
async def test_invalidate_unknown_connector(tmp_path, monkeypatch):
    """Invalidating a non-existent connector is a no-op."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_mcp_registry()

    # Should not raise
    await registry.invalidate_connector("nonexistent")


@pytest.mark.unit
async def test_invalidate_unconnected(tmp_path, monkeypatch):
    """Invalidating a connector that was never created is a no-op."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_mcp_registry()

    # Should not raise (no instance to disconnect)
    await registry.invalidate_connector("control_system")


# ---------------------------------------------------------------------------
# Config validation warnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_validation_warnings_unknown_type(tmp_path, monkeypatch, caplog):
    """Unknown connector types emit warnings."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "control_system": {"type": "unknown_system"},
            "archiver": {"type": "unknown_archiver"},
        },
    )

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.registry"):
        initialize_mcp_registry()

    assert "Unknown control_system.type: unknown_system" in caplog.text
    assert "Unknown archiver.type: unknown_archiver" in caplog.text


@pytest.mark.unit
def test_config_validation_warnings_missing_sections(tmp_path, monkeypatch, caplog):
    """Missing config sections emit warnings."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {})

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.registry"):
        initialize_mcp_registry()

    assert "No control_system section" in caplog.text
    assert "No archiver section" in caplog.text


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_disconnects_all(tmp_path, monkeypatch):
    """shutdown() disconnects all cached connectors."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "control_system": {"type": "mock"},
            "archiver": {"type": "mock_archiver"},
        },
    )
    registry = initialize_mcp_registry()

    mock_cs = AsyncMock()
    mock_arch = AsyncMock()

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_cs,
        ),
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
            new_callable=AsyncMock,
            return_value=mock_arch,
        ),
    ):
        await registry.control_system()
        await registry.archiver()

    await registry.shutdown()

    mock_cs.disconnect.assert_called_once()
    mock_arch.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Channel finder config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_channel_finder_config(tmp_path, monkeypatch):
    """channel_finder_config() returns the correct section."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "channel_finder": {
                "db_path": "/data/channels.db",
                "model_config": {"model": "bge-small"},
            },
        },
    )

    registry = initialize_mcp_registry()
    cf_config = registry.channel_finder_config()

    assert cf_config["db_path"] == "/data/channels.db"
    assert cf_config["model_config"]["model"] == "bge-small"


@pytest.mark.unit
def test_channel_finder_config_empty(tmp_path, monkeypatch):
    """channel_finder_config() returns empty dict when section is missing."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {})

    registry = initialize_mcp_registry()
    assert registry.channel_finder_config() == {}


# ---------------------------------------------------------------------------
# Dot-path access
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dot_path_access(tmp_path, monkeypatch):
    """registry.get('control_system.type') navigates nested config."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "control_system": {"type": "mock", "writes_enabled": True},
        },
    )

    registry = initialize_mcp_registry()

    assert registry.get("control_system.type") == "mock"
    assert registry.get("control_system.writes_enabled") is True
    assert registry.get("nonexistent.key") is None
    assert registry.get("nonexistent.key", "default") == "default"
    assert registry.get("control_system.nonexistent", 42) == 42


# ---------------------------------------------------------------------------
# MCPServerConfig properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_server_config_properties(tmp_path, monkeypatch):
    """MCPServerConfig exposes all expected properties."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {
            "control_system": {"type": "mock", "writes_enabled": False},
            "archiver": {"type": "mock_archiver"},
            "channel_finder": {"db_path": "/test"},
            "ariel": {"api_url": "https://ariel.test"},
            "approval": {"mode": "selective"},
        },
    )

    registry = initialize_mcp_registry()
    cfg = registry.config

    assert cfg.control_system["type"] == "mock"
    assert cfg.archiver["type"] == "mock_archiver"
    assert cfg.channel_finder["db_path"] == "/test"
    assert cfg.ariel["api_url"] == "https://ariel.test"
    assert cfg.approval["mode"] == "selective"
    assert cfg.writes_enabled is False


@pytest.mark.unit
async def test_unknown_connector_raises(tmp_path, monkeypatch):
    """Requesting an unknown connector type raises ValueError."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_mcp_registry()

    with pytest.raises(ValueError, match="Unknown connector"):
        await registry._get_connector("nonexistent")

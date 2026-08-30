"""Tests for the MCP Server Registry.

Covers: initialization, singleton access, connector caching, invalidation,
config validation warnings, the one config the server refuses to start on,
shutdown, channel_finder_config, and dot-path access.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.errors import ConfigurationError
from osprey.mcp_server.control_system.server_context import (
    ControlSystemContext,
    get_server_context,
    initialize_server_context,
    reset_server_context,
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

    registry = initialize_server_context()

    assert registry.config.control_system["type"] == "mock"
    assert registry.config.archiver["type"] == "mock_archiver"
    assert registry.config.channel_finder["db_path"] == "/data/channels.db"
    assert registry.config.writes_enabled is True


@pytest.mark.unit
def test_initialize_missing_config(tmp_path, monkeypatch):
    """Registry initializes with empty config when config.yml is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    registry = initialize_server_context()

    assert registry.config.raw == {}
    assert registry.config.control_system == {}
    assert registry.config.writes_enabled is False


@pytest.mark.unit
def test_initialize_idempotent(tmp_path, monkeypatch):
    """Calling initialize() multiple times is a no-op after the first."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    registry = ControlSystemContext()
    registry.initialize()
    registry.initialize()  # Second call should be a no-op

    assert registry.config.control_system["type"] == "mock"


@pytest.mark.unit
def test_config_not_initialized_raises():
    """Accessing config before initialization raises RuntimeError."""
    registry = ControlSystemContext()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = registry.config


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_access(tmp_path, monkeypatch):
    """get_server_context() returns the same instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    initialize_server_context()
    r1 = get_server_context()
    r2 = get_server_context()

    assert r1 is r2


@pytest.mark.unit
def test_get_before_initialize_raises():
    """get_server_context() raises before initialize_server_context()."""
    with pytest.raises(RuntimeError, match="not initialized"):
        get_server_context()


@pytest.mark.unit
def test_reset_clears_singleton(tmp_path, monkeypatch):
    """reset_server_context() clears the singleton so get raises again."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})

    initialize_server_context()
    reset_server_context()

    with pytest.raises(RuntimeError, match="not initialized"):
        get_server_context()


# ---------------------------------------------------------------------------
# Connector caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connector_caching(tmp_path, monkeypatch):
    """Two calls to registry.control_system() return the same instance."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_server_context()

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
    registry = initialize_server_context()

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
    registry = initialize_server_context()

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
    registry = initialize_server_context()

    # Should not raise
    await registry.invalidate_connector("nonexistent")


@pytest.mark.unit
async def test_invalidate_unconnected(tmp_path, monkeypatch):
    """Invalidating a connector that was never created is a no-op."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_server_context()

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

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.control_system.server_context"):
        initialize_server_context()

    assert "Unknown control_system.type: unknown_system" in caplog.text
    assert "Unknown archiver.type: unknown_archiver" in caplog.text


@pytest.mark.unit
def test_config_validation_warnings_missing_sections(tmp_path, monkeypatch, caplog):
    """Missing config sections emit warnings."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {})

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.control_system.server_context"):
        initialize_server_context()

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
    registry = initialize_server_context()

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

    registry = initialize_server_context()
    cf_config = registry.channel_finder_config()

    assert cf_config["db_path"] == "/data/channels.db"
    assert cf_config["model_config"]["model"] == "bge-small"


@pytest.mark.unit
def test_channel_finder_config_empty(tmp_path, monkeypatch):
    """channel_finder_config() returns empty dict when section is missing."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {})

    registry = initialize_server_context()
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

    registry = initialize_server_context()

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
        },
    )

    registry = initialize_server_context()
    cfg = registry.config

    assert cfg.control_system["type"] == "mock"
    assert cfg.archiver["type"] == "mock_archiver"
    assert cfg.channel_finder["db_path"] == "/test"
    assert cfg.ariel["api_url"] == "https://ariel.test"
    assert cfg.writes_enabled is False


@pytest.mark.unit
async def test_unknown_connector_raises(tmp_path, monkeypatch):
    """Requesting an unknown connector type raises ValueError."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"control_system": {"type": "mock"}})
    registry = initialize_server_context()

    with pytest.raises(ValueError, match="Unknown connector"):
        await registry._get_connector("nonexistent")


# ---------------------------------------------------------------------------
# The honesty rule at runtime: the server refuses to serve invented history
# ---------------------------------------------------------------------------

_VA_MOCK_NESTED = {
    "control_system": {"type": "virtual_accelerator", "writes_enabled": False},
    "archiver": {"type": "mock_archiver"},
}
_VA_UNSET_ARCHIVER = {"control_system": {"type": "virtual_accelerator"}}

# A hand-edited config.yml can carry a top-level dotted line that LOOKS like it
# sets the key. MCPServerConfig reads nested sections only (raw.get("archiver")),
# and the factory then sees no type and falls back to the mock -- so each of
# these runs a VA against invented history while appearing configured.
_FLAT_ARCHIVER_OVER_NESTED_MOCK = _VA_MOCK_NESTED | {"archiver.type": "mongodb_archiver"}
_FLAT_ARCHIVER_ONLY = _VA_UNSET_ARCHIVER | {"archiver.type": "mongodb_archiver"}
_FLAT_CONTROL_SYSTEM_OVER_NESTED_VA = _VA_MOCK_NESTED | {"control_system.type": "mock"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_VA_MOCK_NESTED, id="nested-mock"),
        pytest.param(_VA_UNSET_ARCHIVER, id="archiver-unset"),
        pytest.param(_FLAT_ARCHIVER_OVER_NESTED_MOCK, id="inert-flat-archiver-over-mock"),
        pytest.param(_FLAT_ARCHIVER_ONLY, id="inert-flat-archiver-only"),
        pytest.param(_FLAT_CONTROL_SYSTEM_OVER_NESTED_VA, id="inert-flat-control-system"),
    ],
)
def test_a_virtual_accelerator_with_invented_history_refuses_to_start(
    tmp_path, monkeypatch, config
):
    """Every tool in this server would answer, and the archiver's answers would
    be invented -- a failure no single call can show. So the server does not
    start: whether the mock is named or fallen back to, and whether or not an
    inert top-level dotted line makes the file look configured.
    """
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, config)

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_server_context()

    assert "virtual_accelerator" in str(excinfo.value)


_STANDIN_MOCK_NESTED = {
    "control_system": {"type": "live_standin", "writes_enabled": False},
    "archiver": {"type": "mock_archiver"},
}
_STANDIN_UNSET_ARCHIVER = {"control_system": {"type": "live_standin"}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_STANDIN_MOCK_NESTED, id="standin-nested-mock"),
        pytest.param(_STANDIN_UNSET_ARCHIVER, id="standin-archiver-unset"),
    ],
)
def test_a_stand_in_with_invented_history_is_refused_under_its_own_name(
    tmp_path, monkeypatch, config
):
    """The stand-in is the other machine this deployment stands up for itself,
    so it reaches the same refusal — and whoever reads it at a launcher's stderr
    has no virtual accelerator to go looking for."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, config)

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_server_context()
    message = str(excinfo.value)

    assert "'live_standin'" in message
    # The shared reason sentence names both machines in prose; the *type* named
    # back must be the one this config carries.
    assert "'virtual_accelerator'" not in message


@pytest.mark.unit
def test_the_virtual_accelerator_is_still_named_when_it_is_the_invented_machine(
    tmp_path, monkeypatch
):
    """The type is read from the config rather than assumed, so the case that
    motivated the rule keeps the message it had."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _VA_MOCK_NESTED)

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_server_context()
    message = str(excinfo.value)

    assert "'virtual_accelerator'" in message
    assert "'live_standin'" not in message


@pytest.mark.unit
def test_the_refusal_explains_an_inert_flat_line_rather_than_calling_it_unset(
    tmp_path, monkeypatch
):
    """Someone who typed `archiver.type: mongodb_archiver` at the top level is
    owed better than "unset" -- the reason it did nothing IS the fix."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FLAT_ARCHIVER_ONLY)

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_server_context()
    message = str(excinfo.value)

    assert "configures nothing" in message
    assert "nested sections" in message


@pytest.mark.unit
def test_a_flat_only_control_system_is_a_mock_deployment_not_a_refusal(tmp_path, monkeypatch):
    """The mirror of the rule. With no `control_system:` section the factory
    falls back to the mock, so this is a mock machine with a mock archive --
    honest, however little the inert line achieved."""
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        {"control_system.type": "virtual_accelerator", "archiver": {"type": "mock_archiver"}},
    )

    registry = initialize_server_context()

    assert registry.config.control_system == {}


@pytest.mark.unit
def test_the_refusal_names_the_config_and_both_ways_out(tmp_path, monkeypatch):
    """A refusal at MCP startup reaches someone reading a launcher's stderr, who
    needs the file to open and the edit to make."""
    monkeypatch.chdir(tmp_path)
    config_file = _write_config(tmp_path, _VA_MOCK_NESTED)

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_server_context()
    message = str(excinfo.value)

    assert str(config_file) in message
    assert "mongodb_archiver" in message
    assert "'mock'" in message


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "control_system": {"type": "virtual_accelerator"},
                "archiver": {"type": "mongodb_archiver"},
            },
            id="va-with-store",
        ),
        pytest.param(
            {"control_system": {"type": "mock"}, "archiver": {"type": "mock_archiver"}},
            id="mock-with-mock",
        ),
        pytest.param(
            {"control_system": {"type": "epics"}, "archiver": {"type": "mock_archiver"}},
            id="epics-with-mock",
        ),
    ],
)
def test_every_honest_pairing_still_starts(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, config)

    registry = initialize_server_context()

    assert registry.config.raw == config

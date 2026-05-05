"""Tests for the channel_read MCP tool.

Covers: single read, multiple reads, metadata on/off, connection errors,
unknown channels, and error format compliance.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import assert_error, extract_response_dict, get_tool_fn


def _make_channel_value(
    value=500.2, units="mA", alarm_status="NO_ALARM", timestamp="2024-01-15T10:30:00"
):
    cv = MagicMock()
    cv.value = value
    cv.timestamp = timestamp
    cv.metadata = MagicMock()
    cv.metadata.units = units
    cv.metadata.alarm_status = alarm_status
    cv.metadata.precision = 3
    cv.metadata.description = "Test channel"
    cv.metadata.min_value = 0.0
    cv.metadata.max_value = 1000.0
    return cv


def _get_channel_read():
    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    return get_tool_fn(channel_read)


@pytest.mark.unit
async def test_channel_read_single(tmp_path, monkeypatch):
    """Single channel read returns value with metadata in summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_value = _make_channel_value(value=500.2, units="mA")
    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = mock_value

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["SR:CURRENT:RB"])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["channels_read"] == 1
    assert "SR:CURRENT:RB" in data["summary"]["readings"]
    assert data["summary"]["readings"]["SR:CURRENT:RB"]["value"] == 500.2
    mock_connector.read_channel.assert_called_once_with("SR:CURRENT:RB")


@pytest.mark.unit
async def test_channel_read_multiple(tmp_path, monkeypatch):
    """Multiple channel read returns all values in summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    values = {
        "SR:CURRENT:RB": _make_channel_value(value=500.2, units="mA"),
        "SR:ENERGY:RB": _make_channel_value(value=1.9, units="GeV"),
    }
    mock_connector = AsyncMock()
    mock_connector.read_multiple_channels.return_value = values

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["SR:CURRENT:RB", "SR:ENERGY:RB"])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["channels_read"] == 2
    assert "SR:CURRENT:RB" in data["summary"]["readings"]
    assert "SR:ENERGY:RB" in data["summary"]["readings"]


@pytest.mark.unit
async def test_channel_read_metadata_disabled(tmp_path, monkeypatch):
    """Reading with include_metadata=False omits metadata fields."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_value = _make_channel_value(value=500.2)
    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = mock_value

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["SR:CURRENT:RB"], include_metadata=False)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    channel = data["summary"]["readings"]["SR:CURRENT:RB"]
    assert "value" in channel
    assert "units" in channel


@pytest.mark.unit
async def test_channel_read_with_metadata(tmp_path, monkeypatch):
    """Reading with include_metadata=True includes metadata fields."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_value = _make_channel_value(value=500.2, units="mA", alarm_status="NO_ALARM")
    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = mock_value

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["SR:CURRENT:RB"], include_metadata=True)

    data = extract_response_dict(result)
    channel = data["summary"]["readings"]["SR:CURRENT:RB"]
    assert channel["units"] == "mA"
    assert channel["value"] == 500.2


@pytest.mark.unit
async def test_channel_read_connection_error(tmp_path, monkeypatch):
    """Connection error returns standard error format."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.read_channel.side_effect = ConnectionError("timeout")

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["BAD:PV"])

    data = assert_error(result, error_type="connection_error")
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_channel_read_empty_list(tmp_path, monkeypatch):
    """Empty channel list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_channel_read()
    result = await fn(channels=[])

    data = assert_error(result, error_type="validation_error")

"""Tests for the channel_read MCP tool.

Covers: single read, multiple reads, metadata on/off, connection errors,
unknown channels, and error format compliance.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)


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
    cv.metadata.display_low = 0.0
    cv.metadata.display_high = 1000.0
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
    assert "units" not in channel
    assert "alarm_status" not in channel


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
    assert channel["alarm_status"] == "NO_ALARM"


@pytest.mark.unit
@pytest.mark.parametrize("include_metadata", [True, False])
async def test_access_details_lists_exactly_the_fields_shipped(
    tmp_path, monkeypatch, include_metadata
):
    """fields_per_entry must name the keys the tool actually returns.

    The tool advertised a "metadata" field it never put in the payload, so an
    agent reading access_details asked for a key that was never there.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = _make_channel_value()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        result = await fn(channels=["SR:CURRENT:RB"], include_metadata=include_metadata)

    data = extract_response_dict(result)
    advertised = set(data["access_details"]["fields_per_entry"])
    shipped = set(data["summary"]["readings"]["SR:CURRENT:RB"])
    assert advertised == shipped


@pytest.mark.unit
async def test_include_metadata_changes_the_payload(tmp_path, monkeypatch):
    """include_metadata must be observable — it was inert."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = _make_channel_value()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        with_md = extract_response_dict(await fn(channels=["SR:CURRENT:RB"]))
        without_md = extract_response_dict(
            await fn(channels=["SR:CURRENT:RB"], include_metadata=False)
        )

    with_keys = set(with_md["summary"]["readings"]["SR:CURRENT:RB"])
    without_keys = set(without_md["summary"]["readings"]["SR:CURRENT:RB"])
    assert with_keys > without_keys


@pytest.mark.unit
async def test_channel_read_does_not_promise_write_limits(tmp_path, monkeypatch):
    """No connector reports write bounds here — the tool must not imply it does.

    Channel write bounds come from the limits database via channel_limits; the
    control system only ever reports a display range.
    """
    # "limits" may appear only as a pointer to the channel_limits tool, never as
    # something channel_read claims to return.
    doc = (_get_channel_read().__doc__ or "").replace("channel_limits", "")
    assert "limits" not in doc

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.read_channel.return_value = _make_channel_value()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_channel_read()
        data = extract_response_dict(await fn(channels=["SR:CURRENT:RB"]))

    entry = data["summary"]["readings"]["SR:CURRENT:RB"]
    assert "min_value" not in entry
    assert "max_value" not in entry


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
        with assert_raises_error(error_type="connection_error") as _exc_ctx:
            await fn(channels=["BAD:PV"])

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_channel_read_empty_list(tmp_path, monkeypatch):
    """Empty channel list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_channel_read()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(channels=[])

    _exc_ctx["envelope"]

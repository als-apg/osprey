"""Tests for the channel_write MCP tool.

Covers: successful write with readback, limits violation (inline validator),
verification levels, connection errors, and error format compliance.

Note: writes_enabled check is handled by the PreToolUse hook, not the tool itself.
The tool does its own limits validation via LimitsValidator.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.control_system.registry import initialize_mcp_registry
from tests.mcp_server.conftest import get_tool_fn


def _make_write_result(
    channel="TEST:PV", value=1.0, success=True, error_message=None, verification_level="callback"
):
    result = MagicMock()
    result.channel_address = channel
    result.value_written = value
    result.success = success
    result.error_message = error_message
    result.verification = MagicMock()
    result.verification.level = verification_level
    result.verification.verified = True
    result.verification.readback_value = value
    result.verification.tolerance_used = 0.1
    result.verification.notes = ""
    return result


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return get_tool_fn(channel_write)


@pytest.mark.unit
async def test_channel_write_success(tmp_path, monkeypatch):
    """Successful write returns result with verification in summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_mcp_registry()

    write_result = _make_write_result(channel="TEST:PV", value=42.0)
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "TEST:PV", "value": 42.0}])

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 1
    assert data["summary"]["failed"] == 0
    assert data["summary"]["results"][0]["channel"] == "TEST:PV"


@pytest.mark.unit
async def test_channel_write_with_readback(tmp_path, monkeypatch):
    """Write with readback verification returns verification details in summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_mcp_registry()

    write_result = _make_write_result(channel="TEST:PV", value=42.0, verification_level="readback")
    write_result.verification.readback_value = 42.01
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        fn = _get_channel_write()
        result = await fn(
            operations=[{"channel": "TEST:PV", "value": 42.0}],
            verification_level="readback",
        )

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["results"][0]["verification"]["level"] == "readback"
    assert data["summary"]["results"][0]["verification"]["readback_value"] == 42.01


@pytest.mark.unit
async def test_channel_write_limits_violation(tmp_path, monkeypatch):
    """Write exceeding channel limits (via inline validator) returns error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = ValueError("Value 9999.0 exceeds max limit 100.0")

    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=mock_validator,
    ):
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "TEST:PV", "value": 9999.0}])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "limits_violation"


@pytest.mark.unit
async def test_channel_write_connection_error(tmp_path, monkeypatch):
    """Connection error during write returns standard error format."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_mcp_registry()

    mock_connector = AsyncMock()
    mock_connector.write_channel.side_effect = ConnectionError("IOC unreachable")

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "TEST:PV", "value": 1.0}])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "connection_error"
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_channel_write_multiple_operations(tmp_path, monkeypatch):
    """Multiple write operations are all processed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_mcp_registry()

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_channel.side_effect = results

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        fn = _get_channel_write()
        result = await fn(
            operations=[
                {"channel": "PV:A", "value": 1.0},
                {"channel": "PV:B", "value": 2.0},
            ]
        )

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 2
    assert data["summary"]["failed"] == 0


@pytest.mark.unit
async def test_channel_write_empty_operations(tmp_path, monkeypatch):
    """Empty operations list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_channel_write()
    result = await fn(operations=[])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_channel_write_missing_channel_key(tmp_path, monkeypatch):
    """Operation missing 'channel' key returns validation error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=None,
    ):
        fn = _get_channel_write()
        result = await fn(operations=[{"value": 42.0}])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"

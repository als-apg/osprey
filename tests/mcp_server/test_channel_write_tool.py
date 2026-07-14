"""Tests for the channel_write MCP tool.

Covers: successful write with readback, limits violation (inline validator),
verification levels, connection errors, and error format compliance.

Note: writes_enabled check is handled by the PreToolUse hook, not the tool itself.
The tool does its own limits validation via LimitsValidator.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)


def _make_write_result(
    channel="TEST:PV",
    value=1.0,
    success=True,
    error_message=None,
    verification_level="callback",
    blocked=False,
    refusal_reason=None,
):
    result = MagicMock()
    result.channel_address = channel
    result.value_written = value
    result.success = success
    result.error_message = error_message
    result.blocked = blocked
    result.refusal_reason = refusal_reason
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
    initialize_server_context()

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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 1
    assert data["summary"]["failed"] == 0
    assert data["summary"]["results"][0]["channel"] == "TEST:PV"


@pytest.mark.unit
async def test_channel_write_with_readback(tmp_path, monkeypatch):
    """Write with readback verification returns verification details in summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["results"][0]["verification"]["level"] == "readback"
    assert data["summary"]["results"][0]["verification"]["readback_value"] == 42.01


@pytest.mark.unit
async def test_channel_write_limits_violation(tmp_path, monkeypatch):
    """Write exceeding channel limits (via inline validator) returns structured error."""
    from osprey.errors import ChannelLimitsViolationError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=9999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 9999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=mock_validator,
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="limits_violation") as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 9999.0}])

    data = _exc_ctx["envelope"]
    # Error message includes the channel, value, reason, and allowed range
    assert "TEST:PV" in data["error_message"]
    assert "9999.0" in data["error_message"]
    assert "100.0" in data["error_message"]
    # Structured details include machine-readable limits
    assert "details" in data
    details = data["details"]
    assert details[0]["channel"] == "TEST:PV"
    assert details[0]["min_value"] == 0.0
    assert details[0]["max_value"] == 100.0
    assert details[0]["violation_type"] == "MAX_EXCEEDED"
    # Suggestions are actionable guidance, not the violation banner
    assert any("Do NOT" in s for s in data["suggestions"])


@pytest.mark.unit
async def test_channel_write_connection_error(tmp_path, monkeypatch):
    """Connection error during write returns standard error format."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

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
        with assert_raises_error(error_type="connection_error") as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 1.0}])

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_channel_write_multiple_operations(tmp_path, monkeypatch):
    """Multiple write operations are all processed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(channel="PV:A", value=1.0),
        _make_write_result(channel="PV:B", value=2.0),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 2
    assert data["summary"]["failed"] == 0


@pytest.mark.unit
async def test_channel_write_connector_limits_violation(tmp_path, monkeypatch):
    """ChannelLimitsViolationError from connector is classified as limits_violation with structured details."""
    from osprey.errors import ChannelLimitsViolationError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.write_channel.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

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
        with assert_raises_error() as _exc_ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 999.0}])

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "limits_violation", (
        f"Expected limits_violation but got {data['error_type']} — "
        "ChannelLimitsViolationError from connector must not be misclassified as internal_error"
    )
    # Connector-level catch should also provide structured details
    assert "100.0" in data["error_message"]
    assert "details" in data
    assert data["details"]["channel"] == "TEST:PV"
    assert data["details"]["max_value"] == 100.0


@pytest.mark.unit
async def test_channel_write_empty_operations(tmp_path, monkeypatch):
    """Empty operations list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_channel_write()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(operations=[])

    _exc_ctx["envelope"]


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
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(operations=[{"value": 42.0}])

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_channel_write_all_refused_is_write_refused(tmp_path, monkeypatch):
    """All-refused batch (every op blocked) yields a typed write_refused envelope.

    The connector refuses every write (blocked=True, never sent to the control
    system). The tool must raise ChannelWriteBlockedError so the error handler
    classifies it as write_refused — NOT the generic internal_error that a bare
    RuntimeError would produce.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            success=False,
            error_message="Write to 'PV:A' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="Write to 'PV:B' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

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
        with assert_raises_error() as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "write_refused", (
        f"Expected write_refused but got {data['error_type']} — an all-refused "
        "batch is a policy refusal, not an internal_error"
    )
    assert data["error_type"] != "internal_error"
    # Both refused channels are named in the summary message.
    assert "PV:A" in data["error_message"]
    assert "PV:B" in data["error_message"]
    # Structured details carry the refusal reason discriminator.
    assert data["details"]["reason"] == "WRITES_DISABLED"


@pytest.mark.unit
async def test_channel_write_partial_refusal_reports_per_op(tmp_path, monkeypatch):
    """Partial-rejection batch returns success JSON with per-op refusal reporting.

    One op succeeds, one is refused. The tool must NOT raise (writes did happen);
    it returns a success-status result whose per-op entries carry blocked /
    refusal_reason and whose summary reports refused == 1, successful == 1.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(channel="PV:A", value=1.0, success=True),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="Write to 'PV:B' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    summary = data["summary"]
    assert summary["successful"] == 1
    assert summary["refused"] == 1
    assert summary["failed"] == 1
    by_channel = {r["channel"]: r for r in summary["results"]}
    assert by_channel["PV:A"]["success"] is True
    assert by_channel["PV:A"]["blocked"] is False
    assert by_channel["PV:B"]["success"] is False
    assert by_channel["PV:B"]["blocked"] is True
    assert by_channel["PV:B"]["refusal_reason"] == "WRITES_DISABLED"


@pytest.mark.unit
async def test_channel_write_all_failed_caput_is_internal_error(tmp_path, monkeypatch):
    """All-failed caput (attempted, not refused) preserves internal_error.

    The writes were sent to the control system and failed (blocked=False). This
    is an I/O failure, not a policy refusal, so it must keep the RuntimeError ->
    internal_error classification rather than becoming write_refused.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            success=False,
            error_message="caput failed: timeout",
            blocked=False,
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="caput failed: no connection",
            blocked=False,
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

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
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    data = _exc_ctx["envelope"]
    assert data["error_type"] == "internal_error"
    assert data["error_type"] != "write_refused"

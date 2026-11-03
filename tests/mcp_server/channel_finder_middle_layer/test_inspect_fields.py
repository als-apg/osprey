"""Tests for inspect_fields tool."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.usefixtures("cf_ml_context")


def test_inspect_fields_returns_fields():
    """Happy path: returns field structure for a system/family."""
    mock_db = MagicMock()
    mock_db.inspect_fields.return_value = {
        "Monitor": {"type": "leaf", "description": "Read-only values"},
        "Setpoint": {"type": "leaf", "description": "Writable values"},
    }
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.inspect_fields import (
            inspect_fields,
        )

        fn = get_tool_fn(inspect_fields)
        result = fn(system="SR", family="BPM")

    data = extract_response_dict(result)
    assert "Monitor" in data["fields"]
    assert "Setpoint" in data["fields"]
    mock_db.inspect_fields.assert_called_once_with("SR", "BPM", None)


def test_inspect_fields_with_field_drilldown():
    """Passing a specific field drills down into subfields."""
    mock_db = MagicMock()
    mock_db.inspect_fields.return_value = {
        "X": {"type": "leaf", "description": "Horizontal position"},
        "Y": {"type": "leaf", "description": "Vertical position"},
    }
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.inspect_fields import (
            inspect_fields,
        )

        fn = get_tool_fn(inspect_fields)
        result = fn(system="SR", family="BPM", field="Monitor")

    data = extract_response_dict(result)
    assert "X" in data["fields"]
    assert "Y" in data["fields"]
    mock_db.inspect_fields.assert_called_once_with("SR", "BPM", "Monitor")


def test_inspect_fields_validation_error():
    """ValueError from database returns validation_error envelope."""
    mock_db = MagicMock()
    mock_db.inspect_fields.side_effect = ValueError("Unknown family 'XYZ' in system 'SR'")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.inspect_fields import (
            inspect_fields,
        )

        fn = get_tool_fn(inspect_fields)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(system="SR", family="XYZ")

    data = _exc_ctx["envelope"]
    assert "Unknown family" in data["error_message"]


def test_inspect_fields_internal_error():
    """Internal error returns standard error envelope."""
    mock_db = MagicMock()
    mock_db.inspect_fields.side_effect = Exception("Unexpected failure")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.inspect_fields import (
            inspect_fields,
        )

        fn = get_tool_fn(inspect_fields)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(system="SR", family="BPM")

    data = _exc_ctx["envelope"]
    assert "Unexpected failure" in data["error_message"]

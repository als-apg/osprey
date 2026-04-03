"""Tests for inspect_fields tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.mcp_server.channel_finder_middle_layer.server_context import (
    initialize_cf_ml_context,
)
from tests.mcp_server.channel_finder_middle_layer.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()


@pytest.mark.unit
def test_inspect_fields_returns_fields(tmp_path, monkeypatch):
    """Happy path: returns field structure for a system/family."""
    _setup(tmp_path, monkeypatch)
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

    data = json.loads(result)
    assert "Monitor" in data["fields"]
    assert "Setpoint" in data["fields"]
    mock_db.inspect_fields.assert_called_once_with("SR", "BPM", None)


@pytest.mark.unit
def test_inspect_fields_with_field_drilldown(tmp_path, monkeypatch):
    """Passing a specific field drills down into subfields."""
    _setup(tmp_path, monkeypatch)
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

    data = json.loads(result)
    assert "X" in data["fields"]
    assert "Y" in data["fields"]
    mock_db.inspect_fields.assert_called_once_with("SR", "BPM", "Monitor")


@pytest.mark.unit
def test_inspect_fields_validation_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error envelope."""
    _setup(tmp_path, monkeypatch)
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
        result = fn(system="SR", family="XYZ")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Unknown family" in data["error_message"]


@pytest.mark.unit
def test_inspect_fields_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
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
        result = fn(system="SR", family="BPM")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Unexpected failure" in data["error_message"]

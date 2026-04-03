"""Tests for list_channels tool."""

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
def test_list_channels_returns_channels(tmp_path, monkeypatch):
    """Happy path: returns channel names for a system/family/field path."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.return_value = [
        "SR:C01-MG:BPM1:X",
        "SR:C01-MG:BPM2:X",
        "SR:C02-MG:BPM1:X",
    ]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(system="SR", family="BPM", field="Monitor")

    data = json.loads(result)
    assert data["total"] == 3
    assert "SR:C01-MG:BPM1:X" in data["channels"]
    mock_db.list_channel_names.assert_called_once_with("SR", "BPM", "Monitor", None, None, None)


@pytest.mark.unit
def test_list_channels_with_subfield_and_filters(tmp_path, monkeypatch):
    """Subfield and sector/device filters are passed to database."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.return_value = ["SR:C01-MG:BPM1:X"]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(
            system="SR",
            family="BPM",
            field="Monitor",
            subfield="X",
            sectors=[1, 2],
            devices=[1],
        )

    data = json.loads(result)
    assert data["total"] == 1
    assert "SR:C01-MG:BPM1:X" in data["channels"]
    mock_db.list_channel_names.assert_called_once_with("SR", "BPM", "Monitor", "X", [1, 2], [1])


@pytest.mark.unit
def test_list_channels_validation_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.side_effect = ValueError("Unknown field 'Bad' in 'SR:BPM'")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(system="SR", family="BPM", field="Bad")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Unknown field" in data["error_message"]


@pytest.mark.unit
def test_list_channels_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.side_effect = Exception("Segfault")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(system="SR", family="BPM", field="Monitor")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Segfault" in data["error_message"]

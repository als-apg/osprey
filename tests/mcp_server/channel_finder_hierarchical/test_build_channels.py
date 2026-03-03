"""Tests for build_channels tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.mcp_server.channel_finder_hierarchical.server_context import (
    initialize_cf_hier_context,
)
from tests.mcp_server.channel_finder_hierarchical.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_context()


@pytest.mark.unit
def test_build_channels_happy_path(tmp_path, monkeypatch):
    """Returns list of constructed channel addresses."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.build_channels_from_selections.return_value = [
        "SR:BPM:01:X",
        "SR:BPM:01:Y",
        "SR:BPM:02:X",
    ]
    mock_db.validate_channel.return_value = True
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        selections = {"system": "SR", "family": "BPM", "device": ["01", "02"]}
        result = fn(selections=selections)
    data = json.loads(result)
    assert data["total"] == 3
    assert "SR:BPM:01:X" in data["channels"]
    assert "SR:BPM:02:X" in data["channels"]
    assert data["valid_count"] == 3
    assert data["invalid_count"] == 0
    assert data["valid"] == ["SR:BPM:01:X", "SR:BPM:01:Y", "SR:BPM:02:X"]
    assert data["invalid"] == []
    mock_db.build_channels_from_selections.assert_called_once_with(selections)


@pytest.mark.unit
def test_build_channels_with_invalid_channels(tmp_path, monkeypatch):
    """Inline validation separates valid and invalid channels."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.build_channels_from_selections.return_value = [
        "SR:BPM:01:X",
        "SR:BPM:99:X",
        "SR:BPM:01:Y",
    ]
    mock_db.validate_channel.side_effect = lambda ch: ch != "SR:BPM:99:X"
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        result = fn(selections={"system": "SR", "family": "BPM"})
    data = json.loads(result)
    assert data["total"] == 3
    assert data["valid_count"] == 2
    assert data["invalid_count"] == 1
    assert data["valid"] == ["SR:BPM:01:X", "SR:BPM:01:Y"]
    assert data["invalid"] == ["SR:BPM:99:X"]


@pytest.mark.unit
def test_build_channels_value_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.build_channels_from_selections.side_effect = ValueError(
        "Missing required level: 'system'"
    )
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        result = fn(selections={})
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Missing required level" in data["error_message"]


@pytest.mark.unit
def test_build_channels_internal_error(tmp_path, monkeypatch):
    """Unexpected exception returns internal_error."""
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db crashed"),
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        result = fn(selections={"system": "SR"})
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "db crashed" in data["error_message"]

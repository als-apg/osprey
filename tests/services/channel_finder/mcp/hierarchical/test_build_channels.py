"""Tests for cf_hier_build_channels tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.services.channel_finder.mcp.hierarchical.registry import (
    initialize_cf_hier_registry,
)
from tests.services.channel_finder.mcp.hierarchical.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_registry()


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
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.build_channels import (
            cf_hier_build_channels,
        )

        fn = get_tool_fn(cf_hier_build_channels)
        selections = {"system": "SR", "family": "BPM", "device": ["01", "02"]}
        result = fn(selections=selections)
    data = json.loads(result)
    assert data["total"] == 3
    assert "SR:BPM:01:X" in data["channels"]
    assert "SR:BPM:02:X" in data["channels"]
    mock_db.build_channels_from_selections.assert_called_once_with(selections)


@pytest.mark.unit
def test_build_channels_value_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.build_channels_from_selections.side_effect = ValueError(
        "Missing required level: 'system'"
    )
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.build_channels import (
            cf_hier_build_channels,
        )

        fn = get_tool_fn(cf_hier_build_channels)
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
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db crashed"),
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.build_channels import (
            cf_hier_build_channels,
        )

        fn = get_tool_fn(cf_hier_build_channels)
        result = fn(selections={"system": "SR"})
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "db crashed" in data["error_message"]

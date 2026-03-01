"""Tests for get_options tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.mcp_server.channel_finder_hierarchical.registry import (
    initialize_cf_hier_registry,
)
from tests.mcp_server.channel_finder_hierarchical.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_registry()


@pytest.mark.unit
def test_get_options_happy_path(tmp_path, monkeypatch):
    """Returns options list for a valid level and selections."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_options_at_level.return_value = [
        {"name": "SR", "description": "Storage Ring"},
        {"name": "BR", "description": "Booster Ring"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="system", selections=None)
    data = json.loads(result)
    assert data["level"] == "system"
    assert len(data["options"]) == 2
    assert data["total"] == 2
    assert data["options"][0]["name"] == "SR"
    mock_db.get_options_at_level.assert_called_once_with("system", {})


@pytest.mark.unit
def test_get_options_with_selections(tmp_path, monkeypatch):
    """Passes selections dict through to database."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_options_at_level.return_value = [
        {"name": "BPM", "description": "Beam Position Monitor"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="family", selections={"system": "SR"})
    data = json.loads(result)
    assert data["level"] == "family"
    assert data["total"] == 1
    mock_db.get_options_at_level.assert_called_once_with("family", {"system": "SR"})


@pytest.mark.unit
def test_get_options_value_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_options_at_level.side_effect = ValueError("Unknown level: 'bogus'")
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="bogus", selections=None)
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "bogus" in data["error_message"]


@pytest.mark.unit
def test_get_options_internal_error(tmp_path, monkeypatch):
    """Unexpected exception returns internal_error."""
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db exploded"),
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="system", selections=None)
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "db exploded" in data["error_message"]

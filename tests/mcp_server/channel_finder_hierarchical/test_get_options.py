"""Tests for get_options tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.usefixtures("cf_hier_context")

_DATABASE = (
    "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database"
)


@pytest.fixture
def ring_db(tmp_path) -> HierarchicalChannelDatabase:
    """Two sectors of two correctors, each device under its own sector only."""
    document = {
        "hierarchy": {
            "levels": [{"name": name, "type": "tree"} for name in ("sector", "device", "leaf")],
            "naming_pattern": "{sector}:{device}:{leaf}",
        },
        "tree": {
            sector: {f"{sector}/HCM{magnet}": {"setpoint": {}} for magnet in (1, 2)}
            for sector in ("S01", "S02")
        },
    }
    path = tmp_path / "ring.json"
    path.write_text(json.dumps(document))
    return HierarchicalChannelDatabase(str(path))


def test_get_options_happy_path():
    """Returns options list for a valid level and selections."""
    mock_db = MagicMock()
    mock_db.get_options_at_level.return_value = [
        {"name": "SR", "description": "Storage Ring"},
        {"name": "BR", "description": "Booster Ring"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="system", selections=None)
    data = extract_response_dict(result)
    assert data["level"] == "system"
    assert len(data["options"]) == 2
    assert data["total"] == 2
    assert data["options"][0]["name"] == "SR"
    mock_db.get_options_at_level.assert_called_once_with("system", {})


def test_get_options_with_selections():
    """Passes selections dict through to database."""
    mock_db = MagicMock()
    mock_db.get_options_at_level.return_value = [
        {"name": "BPM", "description": "Beam Position Monitor"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        result = fn(level="family", selections={"system": "SR"})
    data = extract_response_dict(result)
    assert data["level"] == "family"
    assert data["total"] == 1
    mock_db.get_options_at_level.assert_called_once_with("family", {"system": "SR"})


def test_get_options_value_error():
    """ValueError from database returns validation_error."""
    mock_db = MagicMock()
    mock_db.get_options_at_level.side_effect = ValueError("Unknown level: 'bogus'")
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(level="bogus", selections=None)
    data = _exc_ctx["envelope"]
    assert "bogus" in data["error_message"]


def test_get_options_internal_error():
    """Unexpected exception returns internal_error."""
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db exploded"),
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(level="system", selections=None)
    data = _exc_ctx["envelope"]
    assert "db exploded" in data["error_message"]


@pytest.mark.parametrize(
    ("level", "selections", "name"),
    [
        ("sectr", None, "sectr"),
        ("device", {"bogus": "X"}, "bogus"),
        ("leaf", {"sector": "S01", "device": "S02/HCM1"}, "S02/HCM1"),
    ],
    ids=["undefined-level", "unknown-key", "unknown-value"],
)
def test_get_options_refuses_a_selection_mistake(ring_db, level, selections, name):
    """A real database's refusal reaches the agent as validation_error naming the mistake."""
    with patch(_DATABASE, new_callable=PropertyMock, return_value=ring_db):
        from osprey.mcp_server.channel_finder_hierarchical.tools.get_options import (
            get_options,
        )

        fn = get_tool_fn(get_options)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(level=level, selections=selections)
    data = _exc_ctx["envelope"]
    assert repr(name) in data["error_message"]

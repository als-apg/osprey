"""Tests for validate tool."""

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
def test_validate_mixed_valid_invalid(tmp_path, monkeypatch):
    """Returns per-channel results with valid/invalid counts."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    # First channel valid, second invalid, third valid
    mock_db.validate_channel.side_effect = [True, False, True]
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["SR:BPM:01:X", "FAKE:CHANNEL", "SR:BPM:02:Y"])
    data = json.loads(result)
    assert data["total"] == 3
    assert data["valid_count"] == 2
    assert data["invalid_count"] == 1
    assert len(data["results"]) == 3
    assert data["results"][0] == {"channel": "SR:BPM:01:X", "valid": True}
    assert data["results"][1] == {"channel": "FAKE:CHANNEL", "valid": False}
    assert data["results"][2] == {"channel": "SR:BPM:02:Y", "valid": True}


@pytest.mark.unit
def test_validate_all_valid(tmp_path, monkeypatch):
    """All channels valid."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.validate_channel.return_value = True
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["SR:BPM:01:X"])
    data = json.loads(result)
    assert data["valid_count"] == 1
    assert data["invalid_count"] == 0


@pytest.mark.unit
def test_validate_empty_list(tmp_path, monkeypatch):
    """Empty channel list returns zero counts."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=[])
    data = json.loads(result)
    assert data["total"] == 0
    assert data["valid_count"] == 0
    assert data["invalid_count"] == 0
    assert data["results"] == []


@pytest.mark.unit
def test_validate_internal_error(tmp_path, monkeypatch):
    """Unexpected exception returns internal_error."""
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db unreachable"),
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["SR:BPM:01:X"])
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "db unreachable" in data["error_message"]

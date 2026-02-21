"""Tests for statistics tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.services.channel_finder.mcp.middle_layer.registry import (
    initialize_cf_ml_registry,
)
from tests.services.channel_finder.mcp.middle_layer.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_registry()


@pytest.mark.unit
def test_statistics_returns_stats(tmp_path, monkeypatch):
    """Happy path: returns database statistics."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.return_value = {
        "total_channels": 1500,
        "total_systems": 3,
        "total_families": 25,
    }
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()

    data = json.loads(result)
    assert data["total_channels"] == 1500
    assert data["total_systems"] == 3
    assert data["total_families"] == 25


@pytest.mark.unit
def test_statistics_empty_database(tmp_path, monkeypatch):
    """Returns zero counts for empty database."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.return_value = {
        "total_channels": 0,
        "total_systems": 0,
        "total_families": 0,
    }
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()

    data = json.loads(result)
    assert data["total_channels"] == 0
    assert data["total_systems"] == 0


@pytest.mark.unit
def test_statistics_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.side_effect = Exception("Stats computation failed")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Stats computation failed" in data["error_message"]

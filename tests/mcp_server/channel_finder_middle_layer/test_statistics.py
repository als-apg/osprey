"""Tests for statistics tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch
from tests.mcp_server.conftest import assert_error, assert_raises_error, extract_response_dict

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
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()

    data = extract_response_dict(result)
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
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()

    data = extract_response_dict(result)
    assert data["total_channels"] == 0
    assert data["total_systems"] == 0


@pytest.mark.unit
def test_statistics_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.side_effect = Exception("Stats computation failed")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn()

    data = _exc_ctx["envelope"]
    assert "Stats computation failed" in data["error_message"]

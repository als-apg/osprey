"""Tests for statistics tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.mcp_server.channel_finder_in_context.server_context import (
    initialize_cf_ic_context,
)
from tests.mcp_server.channel_finder_in_context.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ic_context()


@pytest.mark.unit
def test_statistics_happy_path(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.return_value = {
        "total_channels": 100,
        "format": "flat",
    }
    mock_db.chunk_database.return_value = [
        [{"channel": f"CH{i}"} for i in range(50)],
        [{"channel": f"CH{i}"} for i in range(50, 100)],
    ]
    with patch(
        "osprey.mcp_server.channel_finder_in_context.server_context.ChannelFinderICContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_in_context.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()
    data = json.loads(result)
    assert data["total_channels"] == 100
    assert data["format"] == "flat"
    assert data["total_chunks_at_50"] == 2
    assert "facility_name" in data
    mock_db.chunk_database.assert_called_once_with(50)


@pytest.mark.unit
def test_statistics_internal_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.side_effect = RuntimeError("Stats unavailable")
    with patch(
        "osprey.mcp_server.channel_finder_in_context.server_context.ChannelFinderICContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_in_context.tools.statistics import (
            statistics,
        )

        fn = get_tool_fn(statistics)
        result = fn()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Stats unavailable" in data["error_message"]

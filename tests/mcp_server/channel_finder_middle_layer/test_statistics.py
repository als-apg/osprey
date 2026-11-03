"""Tests for statistics tool."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.usefixtures("cf_ml_context")


def test_statistics_returns_stats():
    """Happy path: returns database statistics."""
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


def test_statistics_empty_database():
    """Returns zero counts for empty database."""
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


def test_statistics_internal_error():
    """Internal error returns standard error envelope."""
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

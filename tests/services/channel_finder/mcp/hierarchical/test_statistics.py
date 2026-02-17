"""Tests for cf_hier_statistics tool."""

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
def test_statistics_happy_path(tmp_path, monkeypatch):
    """Returns statistics dict from database."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_statistics.return_value = {
        "total_channels": 1500,
        "hierarchy_levels": ["system", "family", "device", "signal"],
        "systems": {
            "SR": {"channels": 1200},
            "BR": {"channels": 300},
        },
    }
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.statistics import (
            cf_hier_statistics,
        )

        fn = get_tool_fn(cf_hier_statistics)
        result = fn()
    data = json.loads(result)
    assert data["total_channels"] == 1500
    assert "SR" in data["systems"]
    assert data["systems"]["SR"]["channels"] == 1200
    mock_db.get_statistics.assert_called_once()


@pytest.mark.unit
def test_statistics_internal_error(tmp_path, monkeypatch):
    """Unexpected exception returns internal_error."""
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("db not available"),
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.statistics import (
            cf_hier_statistics,
        )

        fn = get_tool_fn(cf_hier_statistics)
        result = fn()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "db not available" in data["error_message"]

"""Tests for hierarchy_info tool."""

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
def test_hierarchy_info_returns_structure(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.hierarchy_levels = ["system", "device", "signal"]
    mock_db.hierarchy_config = {
        "levels": {
            "system": {"type": "tree"},
            "device": {"type": "instances"},
            "signal": {"type": "tree"},
        }
    }
    mock_db.naming_pattern = "{system}:{device}:{signal}"
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.hierarchy_info import (
            hierarchy_info,
        )

        fn = get_tool_fn(hierarchy_info)
        result = fn()
    data = json.loads(result)
    assert data["hierarchy_levels"] == ["system", "device", "signal"]
    assert "{system}" in data["naming_pattern"]
    assert data["facility_name"] == "control system"


@pytest.mark.unit
def test_hierarchy_info_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.services.channel_finder.mcp.hierarchical.registry."
        "ChannelFinderHierRegistry.database",
        new_callable=PropertyMock,
        side_effect=RuntimeError("not configured"),
    ):
        from osprey.services.channel_finder.mcp.hierarchical.tools.hierarchy_info import (
            hierarchy_info,
        )

        fn = get_tool_fn(hierarchy_info)
        result = fn()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "not configured" in data["error_message"]

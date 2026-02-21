"""Tests for list_systems tool."""

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
def test_list_systems_returns_systems(tmp_path, monkeypatch):
    """Happy path: returns list of systems with count."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_systems.return_value = [
        {"name": "SR", "description": "Storage Ring"},
        {"name": "BR", "description": "Booster Ring"},
    ]
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        result = fn()

    data = json.loads(result)
    assert data["total"] == 2
    assert data["systems"][0]["name"] == "SR"
    assert data["systems"][1]["name"] == "BR"


@pytest.mark.unit
def test_list_systems_empty(tmp_path, monkeypatch):
    """Returns empty list when no systems exist."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_systems.return_value = []
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        result = fn()

    data = json.loads(result)
    assert data["total"] == 0
    assert data["systems"] == []


@pytest.mark.unit
def test_list_systems_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_systems.side_effect = Exception("DB broke")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        result = fn()

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB broke" in data["error_message"]

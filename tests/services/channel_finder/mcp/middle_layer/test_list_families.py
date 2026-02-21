"""Tests for list_families tool."""

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
def test_list_families_returns_families(tmp_path, monkeypatch):
    """Happy path: returns list of families for a system."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_families.return_value = [
        {"name": "BPM", "description": "Beam Position Monitor"},
        {"name": "QF", "description": "Focusing Quadrupole"},
    ]
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        result = fn(system="SR")

    data = json.loads(result)
    assert data["total"] == 2
    assert data["families"][0]["name"] == "BPM"
    assert data["families"][1]["name"] == "QF"
    mock_db.list_families.assert_called_once_with("SR")


@pytest.mark.unit
def test_list_families_validation_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_families.side_effect = ValueError("Unknown system 'XX'")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        result = fn(system="XX")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Unknown system" in data["error_message"]


@pytest.mark.unit
def test_list_families_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_families.side_effect = Exception("DB broke")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        result = fn(system="SR")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB broke" in data["error_message"]

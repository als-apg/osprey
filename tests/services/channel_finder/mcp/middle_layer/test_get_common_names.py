"""Tests for get_common_names tool."""

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
def test_get_common_names_returns_names(tmp_path, monkeypatch):
    """Happy path: returns common names for a family."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_common_names.return_value = ["BPM 1", "BPM 2", "BPM 3"]
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        result = fn(system="SR", family="BPM")

    data = json.loads(result)
    assert data["common_names"] == ["BPM 1", "BPM 2", "BPM 3"]
    mock_db.get_common_names.assert_called_once_with("SR", "BPM")


@pytest.mark.unit
def test_get_common_names_returns_none(tmp_path, monkeypatch):
    """Returns null common_names with message when not available."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_common_names.return_value = None
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        result = fn(system="SR", family="QF")

    data = json.loads(result)
    assert data["common_names"] is None
    assert "message" in data
    assert "No common names" in data["message"]


@pytest.mark.unit
def test_get_common_names_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_common_names.side_effect = Exception("DB connection lost")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        result = fn(system="SR", family="BPM")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB connection lost" in data["error_message"]

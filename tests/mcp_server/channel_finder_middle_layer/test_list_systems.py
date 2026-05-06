"""Tests for list_systems tool."""

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
def test_list_systems_returns_systems(tmp_path, monkeypatch):
    """Happy path: returns list of systems with count."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_systems.return_value = [
        {"name": "SR", "description": "Storage Ring"},
        {"name": "BR", "description": "Booster Ring"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        result = fn()

    data = extract_response_dict(result)
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
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        result = fn()

    data = extract_response_dict(result)
    assert data["total"] == 0
    assert data["systems"] == []


@pytest.mark.unit
def test_list_systems_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_systems.side_effect = Exception("DB broke")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_systems import (
            list_systems,
        )

        fn = get_tool_fn(list_systems)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn()

    data = _exc_ctx["envelope"]
    assert "DB broke" in data["error_message"]

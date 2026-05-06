"""Tests for list_families tool."""

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
def test_list_families_returns_families(tmp_path, monkeypatch):
    """Happy path: returns list of families for a system."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_families.return_value = [
        {"name": "BPM", "description": "Beam Position Monitor"},
        {"name": "QF", "description": "Focusing Quadrupole"},
    ]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        result = fn(system="SR")

    data = extract_response_dict(result)
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
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(system="XX")

    data = _exc_ctx["envelope"]
    assert "Unknown system" in data["error_message"]


@pytest.mark.unit
def test_list_families_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_families.side_effect = Exception("DB broke")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_families import (
            list_families,
        )

        fn = get_tool_fn(list_families)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(system="SR")

    data = _exc_ctx["envelope"]
    assert "DB broke" in data["error_message"]

"""Tests for get_common_names tool."""

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
def test_get_common_names_returns_names(tmp_path, monkeypatch):
    """Happy path: returns common names for a family."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_common_names.return_value = ["BPM 1", "BPM 2", "BPM 3"]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        result = fn(system="SR", family="BPM")

    data = extract_response_dict(result)
    assert data["common_names"] == ["BPM 1", "BPM 2", "BPM 3"]
    mock_db.get_common_names.assert_called_once_with("SR", "BPM")


@pytest.mark.unit
def test_get_common_names_returns_none(tmp_path, monkeypatch):
    """Returns null common_names with message when not available."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_common_names.return_value = None
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        result = fn(system="SR", family="QF")

    data = extract_response_dict(result)
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
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.get_common_names import (
            get_common_names,
        )

        fn = get_tool_fn(get_common_names)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(system="SR", family="BPM")

    data = _exc_ctx["envelope"]
    assert "DB connection lost" in data["error_message"]

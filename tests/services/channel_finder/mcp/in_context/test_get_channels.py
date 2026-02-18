"""Tests for cf_ic_get_channels tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.services.channel_finder.mcp.in_context.registry import (
    initialize_cf_ic_registry,
)
from tests.services.channel_finder.mcp.in_context.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ic_registry()


@pytest.mark.unit
def test_get_all_channels(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_all_channels.return_value = [
        {"channel": "CH1", "address": "PV:CH1"},
        {"channel": "CH2", "address": "PV:CH2"},
    ]
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.get_channels import (
            cf_ic_get_channels,
        )

        fn = get_tool_fn(cf_ic_get_channels)
        result = fn()
    data = json.loads(result)
    assert data["total"] == 2
    assert len(data["channels"]) == 2


@pytest.mark.unit
def test_get_channels_chunked(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.chunk_database.return_value = [
        [{"channel": "CH1"}],
        [{"channel": "CH2"}],
    ]
    mock_db.format_chunk_for_prompt.return_value = "- CH1"
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.get_channels import (
            cf_ic_get_channels,
        )

        fn = get_tool_fn(cf_ic_get_channels)
        result = fn(chunk_idx=0, chunk_size=1)
    data = json.loads(result)
    assert data["chunk_idx"] == 0
    assert data["total_chunks"] == 2
    assert "CH1" in data["formatted"]


@pytest.mark.unit
def test_get_channels_invalid_chunk(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.chunk_database.return_value = [[{"channel": "CH1"}]]
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.get_channels import (
            cf_ic_get_channels,
        )

        fn = get_tool_fn(cf_ic_get_channels)
        result = fn(chunk_idx=5, chunk_size=1)
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
def test_get_channels_internal_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_all_channels.side_effect = RuntimeError("DB exploded")
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.get_channels import (
            cf_ic_get_channels,
        )

        fn = get_tool_fn(cf_ic_get_channels)
        result = fn()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB exploded" in data["error_message"]

"""Tests for validate tool."""

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
def test_validate_valid_and_invalid(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.validate_channels.return_value = [
        {"channel": "CH1", "valid": True},
        {"channel": "FAKE", "valid": False},
    ]
    mock_db.get_valid_channels.return_value = ["CH1"]
    mock_db.get_invalid_channels.return_value = ["FAKE"]
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["CH1", "FAKE"])
    data = json.loads(result)
    assert data["total"] == 2
    assert data["valid_count"] == 1
    assert data["invalid_count"] == 1
    assert "CH1" in data["valid_channels"]
    assert "FAKE" in data["invalid_channels"]


@pytest.mark.unit
def test_validate_empty_list(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from osprey.services.channel_finder.mcp.in_context.tools.validate import (
        validate,
    )

    fn = get_tool_fn(validate)
    result = fn(channels=[])
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Empty" in data["error_message"]


@pytest.mark.unit
def test_validate_internal_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.validate_channels.side_effect = RuntimeError("DB connection lost")
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["CH1"])
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB connection lost" in data["error_message"]

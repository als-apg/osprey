"""Tests for cf_ml_validate tool."""

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
def test_validate_returns_results(tmp_path, monkeypatch):
    """Happy path: validates channel names and returns results."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.validate_channel.side_effect = [True, False]
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.validate import (
            cf_ml_validate,
        )

        fn = get_tool_fn(cf_ml_validate)
        result = fn(channels=["SR:BPM1:X", "INVALID:PV"])

    data = json.loads(result)
    assert data["total"] == 2
    assert data["results"][0]["channel"] == "SR:BPM1:X"
    assert data["results"][0]["valid"] is True
    assert data["results"][1]["channel"] == "INVALID:PV"
    assert data["results"][1]["valid"] is False


@pytest.mark.unit
def test_validate_empty_list(tmp_path, monkeypatch):
    """Empty channel list returns validation_error envelope."""
    _setup(tmp_path, monkeypatch)
    from osprey.services.channel_finder.mcp.middle_layer.tools.validate import (
        cf_ml_validate,
    )

    fn = get_tool_fn(cf_ml_validate)
    result = fn(channels=[])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "Empty channel list" in data["error_message"]


@pytest.mark.unit
def test_validate_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.validate_channel.side_effect = Exception("Corrupted index")
    with patch(
        "osprey.services.channel_finder.mcp.middle_layer.registry.ChannelFinderMLRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.middle_layer.tools.validate import (
            cf_ml_validate,
        )

        fn = get_tool_fn(cf_ml_validate)
        result = fn(channels=["SR:BPM1:X"])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Corrupted index" in data["error_message"]

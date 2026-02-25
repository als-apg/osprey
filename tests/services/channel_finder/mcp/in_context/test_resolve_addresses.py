"""Tests for resolve_addresses tool."""

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
def test_resolve_addresses_happy_path(tmp_path, monkeypatch):
    """All channels resolve — correct counts."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_channel.side_effect = lambda name: {
        "TubeGunPressure": {"address": "GunPressure"},
        "TubeVacuumStatus": {"address": "VacStatus"},
    }.get(name)
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.resolve_addresses import (
            resolve_addresses,
        )

        fn = get_tool_fn(resolve_addresses)
        result = fn(channels=["TubeGunPressure", "TubeVacuumStatus"])
    data = json.loads(result)
    assert data["total"] == 2
    assert data["valid_count"] == 2
    assert data["invalid_count"] == 0
    assert data["addresses"] == ["GunPressure", "VacStatus"]
    assert data["unresolved"] == []


@pytest.mark.unit
def test_resolve_addresses_partial_unresolved(tmp_path, monkeypatch):
    """Some channels unresolved — correct valid/invalid counts."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_channel.side_effect = lambda name: (
        {"address": "GunPressure"} if name == "TubeGunPressure" else None
    )
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.resolve_addresses import (
            resolve_addresses,
        )

        fn = get_tool_fn(resolve_addresses)
        result = fn(channels=["TubeGunPressure", "NoSuchChannel"])
    data = json.loads(result)
    assert data["total"] == 2
    assert data["valid_count"] == 1
    assert data["invalid_count"] == 1
    assert data["addresses"] == ["GunPressure"]
    assert data["unresolved"] == ["NoSuchChannel"]


@pytest.mark.unit
def test_resolve_addresses_all_unresolved(tmp_path, monkeypatch):
    """No channels resolve — all listed as unresolved."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.get_channel.return_value = None
    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.ChannelFinderICRegistry.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.services.channel_finder.mcp.in_context.tools.resolve_addresses import (
            resolve_addresses,
        )

        fn = get_tool_fn(resolve_addresses)
        result = fn(channels=["BadA", "BadB"])
    data = json.loads(result)
    assert data["total"] == 2
    assert data["valid_count"] == 0
    assert data["invalid_count"] == 2
    assert data["addresses"] == []
    assert data["unresolved"] == ["BadA", "BadB"]

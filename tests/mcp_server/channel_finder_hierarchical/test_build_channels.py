"""Tests for build_channels tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.usefixtures("cf_hier_context")

_DATABASE = (
    "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.database"
)

_SECTORS = [f"S{sector:02d}" for sector in range(1, 13)]
_DEVICES = [f"S{sector:02d}/HCM{magnet}" for sector in range(1, 13) for magnet in range(1, 7)]


@pytest.fixture
def ring_db(tmp_path) -> HierarchicalChannelDatabase:
    """Twelve sectors of six correctors, each device under its own sector only."""
    tree = {
        sector: {
            f"{sector}/HCM{magnet}": {"setpoint": {}, "readback": {}} for magnet in range(1, 7)
        }
        for sector in _SECTORS
    }
    document = {
        "hierarchy": {
            "levels": [{"name": name, "type": "tree"} for name in ("sector", "device", "leaf")],
            "naming_pattern": "{sector}:{device}:{leaf}",
        },
        "tree": tree,
    }
    path = tmp_path / "ring.json"
    path.write_text(json.dumps(document))
    return HierarchicalChannelDatabase(str(path))


def _build(selections: dict, database) -> dict:
    with patch(_DATABASE, new_callable=PropertyMock, return_value=database):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        return extract_response_dict(get_tool_fn(build_channels)(selections=selections))


def test_build_channels_lists_each_existing_address_once(ring_db):
    """A multi-sector selection answers with the real addresses, not the product."""
    data = _build(
        {"sector": _SECTORS, "device": _DEVICES, "leaf": ["setpoint", "readback"]}, ring_db
    )

    assert data["total"] == 144
    assert len(data["channels"]) == len(set(data["channels"])) == 144
    assert all(ring_db.validate_channel(channel) for channel in data["channels"])
    assert data["channels"][:2] == ["S01:S01/HCM1:setpoint", "S01:S01/HCM1:readback"]
    # 12 sectors x 72 devices x 2 leaves, less the 144 that exist.
    assert data["invalid_count"] == 1584
    assert data["invalid_examples"][0] == {
        "sector": "S01",
        "device": "S02/HCM1",
        "leaf": "setpoint",
    }
    assert len(data["invalid_examples"]) == 10
    assert set(data) == {"channels", "total", "invalid_count", "invalid_examples"}


def test_build_channels_with_no_existing_address(ring_db):
    data = _build({"sector": "S01", "device": "S01/HCM1", "leaf": "nope"}, ring_db)

    assert data == {
        "channels": [],
        "total": 0,
        "invalid_count": 1,
        "invalid_examples": [{"sector": "S01", "device": "S01/HCM1", "leaf": "nope"}],
    }


def test_build_channels_value_error(ring_db):
    """A missing required level returns validation_error."""
    with patch(_DATABASE, new_callable=PropertyMock, return_value=ring_db):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(selections={"sector": "S01"})
    data = _exc_ctx["envelope"]
    assert "Required level 'device' is missing" in data["error_message"]


def test_build_channels_internal_error():
    """Unexpected exception returns internal_error."""
    with patch(_DATABASE, new_callable=PropertyMock, side_effect=RuntimeError("db crashed")):
        from osprey.mcp_server.channel_finder_hierarchical.tools.build_channels import (
            build_channels,
        )

        fn = get_tool_fn(build_channels)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(selections={"system": "SR"})
    data = _exc_ctx["envelope"]
    assert "db crashed" in data["error_message"]


def test_build_channels_answers_with_the_service_helper():
    """The tool returns the database's answer unchanged."""
    answer = {"channels": ["A"], "total": 1, "invalid_count": 0, "invalid_examples": []}
    mock_db = MagicMock(spec=HierarchicalChannelDatabase)
    mock_db.build_existing_channels.return_value = answer

    assert _build({"x": "y"}, mock_db) == answer
    mock_db.build_existing_channels.assert_called_once_with({"x": "y"})

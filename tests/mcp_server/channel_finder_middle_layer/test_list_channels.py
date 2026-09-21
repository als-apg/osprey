"""Tests for list_channels tool."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

from osprey.mcp_server.channel_finder_middle_layer.server_context import (
    initialize_cf_ml_context,
)
from tests.mcp_server.channel_finder_middle_layer.conftest import get_tool_fn
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()


def test_list_channels_returns_channels(tmp_path, monkeypatch):
    """Happy path: returns channel names for a system/family/field path."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.return_value = [
        "SR:C01-MG:BPM1:X",
        "SR:C01-MG:BPM2:X",
        "SR:C02-MG:BPM1:X",
    ]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(system="SR", family="BPM", field="Monitor")

    data = extract_response_dict(result)
    assert data["total"] == 3
    assert "SR:C01-MG:BPM1:X" in data["channels"]
    mock_db.list_channel_names.assert_called_once_with("SR", "BPM", "Monitor", None, None, None)


def test_list_channels_with_subfield_and_filters(tmp_path, monkeypatch):
    """Subfield and sector/device filters are passed to database."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.return_value = ["SR:C01-MG:BPM1:X"]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        result = fn(
            system="SR",
            family="BPM",
            field="Monitor",
            subfield="X",
            sectors=[1, 2],
            devices=[1],
        )

    data = extract_response_dict(result)
    assert data["total"] == 1
    assert "SR:C01-MG:BPM1:X" in data["channels"]
    mock_db.list_channel_names.assert_called_once_with("SR", "BPM", "Monitor", "X", [1, 2], [1])


def test_list_channels_validation_error(tmp_path, monkeypatch):
    """ValueError from database returns validation_error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.side_effect = ValueError("Unknown field 'Bad' in 'SR:BPM'")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            fn(system="SR", family="BPM", field="Bad")

    data = _exc_ctx["envelope"]
    assert "Unknown field" in data["error_message"]


def test_list_channels_internal_error(tmp_path, monkeypatch):
    """Internal error returns standard error envelope."""
    _setup(tmp_path, monkeypatch)
    mock_db = MagicMock()
    mock_db.list_channel_names.side_effect = Exception("Segfault")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        fn = get_tool_fn(list_channels)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(system="SR", family="BPM", field="Monitor")

    data = _exc_ctx["envelope"]
    assert "Segfault" in data["error_message"]


#: A dual-key field (CA and Tango names) and a Tango-only field in one family.
_PROTOCOL_DB = {
    "RING": {
        "KICK": {
            "Voltage": {
                "ChannelNames": ["K1:V", "K2:V"],
                "TangoNames": ["ring/kick/1/v", "ring/kick/2/v"],
            },
            "Current": {"TangoNames": ["ring/kick/1/i", "ring/kick/2/i"]},
            "setup": {"DeviceList": [[1, 1], [1, 2]]},
        }
    }
}


def _real_database(tmp_path):
    from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(_PROTOCOL_DB), encoding="utf-8")
    return MiddleLayerDatabase(str(path))


def _call_on(database, **kwargs):
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=database,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.list_channels import (
            list_channels,
        )

        return get_tool_fn(list_channels)(**kwargs)


def test_list_channels_schema_offers_protocol_enum():
    """``protocol`` is an optional schema property restricted to ``ca``/``tango``."""
    import asyncio

    import osprey.mcp_server.channel_finder_middle_layer.tools.list_channels  # noqa: F401
    from osprey.mcp_server.channel_finder_middle_layer.server import mcp

    schema = asyncio.run(mcp.get_tool("list_channels")).parameters
    assert "protocol" not in schema.get("required", [])
    prop = json.dumps(schema["properties"]["protocol"])
    assert '"ca"' in prop
    assert '"tango"' in prop


def test_list_channels_protocol_tango_on_dual_key(tmp_path, monkeypatch):
    """``protocol='tango'`` returns the TangoNames list of a dual-key field."""
    _setup(tmp_path, monkeypatch)
    database = _real_database(tmp_path)

    result = _call_on(database, system="RING", family="KICK", field="Voltage", protocol="tango")

    data = extract_response_dict(result)
    assert data == {"channels": ["ring/kick/1/v", "ring/kick/2/v"], "total": 2}


def test_list_channels_protocol_absent_is_validation_error(tmp_path, monkeypatch):
    """Asking a Tango-only field for ``ca`` names the keys the field carries."""
    _setup(tmp_path, monkeypatch)
    database = _real_database(tmp_path)

    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        _call_on(database, system="RING", family="KICK", field="Current", protocol="ca")

    message = _exc_ctx["envelope"]["error_message"]
    assert "ChannelNames" in message
    assert "TangoNames" in message

"""The phoebus server's half of ``phoebus.agent_access``.

Under ``read`` (the default) the server leaves ``phoebus_drive`` out of
``tools/list`` and refuses a call to it, naming the key. Under ``read_write``
the drive is listed and reaches the bridge.
"""

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.phoebus.server import mcp
from osprey.mcp_server.phoebus.tools import bridge_tools
from osprey.phoebus_agent_access import agent_access, drive_offered
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
    registered_tool_names,
)

_MOD = "osprey.mcp_server.phoebus.tools.bridge_tools"

_READING_TOOLS = [
    "phoebus_list_displays",
    "phoebus_perceive",
    "phoebus_perceive_region",
    "phoebus_snapshot",
    "phoebus_open_panel",
    "phoebus_open_databrowser",
    "phoebus_panel_lookup",
]


def _drive():
    return get_tool_fn(bridge_tools.phoebus_drive)


def _config(tmp_path, monkeypatch, phoebus):
    """Write ``{"phoebus": phoebus}`` to a temp config.yml and point OSPREY_CONFIG at it."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"phoebus": phoebus}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")
    return config_file


# ── the parser ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("config", [{}, {"phoebus": None}, {"phoebus": {}}])
def test_agent_access_defaults_to_read(config):
    assert agent_access(config) == "read"


def test_agent_access_accepts_both_values():
    assert agent_access({"phoebus": {"agent_access": "read"}}) == "read"
    assert agent_access({"phoebus": {"agent_access": "read_write"}}) == "read_write"


@pytest.mark.parametrize("value", ["write", "READ", True, 1, None])
def test_agent_access_refuses_an_unknown_value_by_name(value):
    with pytest.raises(ValueError) as exc_info:
        agent_access({"phoebus": {"agent_access": value}})
    message = str(exc_info.value)
    assert "phoebus.agent_access" in message
    assert repr(value) in message


def test_drive_offered_fails_closed_on_an_unknown_value():
    assert drive_offered({"phoebus": {"agent_access": "write"}}) is False
    assert drive_offered({"phoebus": {"agent_access": "read"}}) is False
    assert drive_offered({"phoebus": {"agent_access": "read_write"}}) is True


# ── the tool list ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("phoebus", [{}, {"agent_access": "read"}])
def test_read_leaves_phoebus_drive_out_of_the_tool_list(phoebus, tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, phoebus)
    names = registered_tool_names(mcp)
    assert "phoebus_drive" not in names
    for tool in _READING_TOOLS:
        assert tool in names


def test_read_write_lists_phoebus_drive(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {"agent_access": "read_write"})
    assert "phoebus_drive" in registered_tool_names(mcp)


def test_unknown_value_leaves_phoebus_drive_out_of_the_tool_list(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {"agent_access": "write"})
    assert "phoebus_drive" not in registered_tool_names(mcp)


# ── the drive refusal ──────────────────────────────────────────────────────
async def test_read_refuses_a_drive_naming_the_key(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {})
    with patch(f"{_MOD}._http_post_drive") as post:
        with assert_raises_error(error_type="not_supported") as ctx:
            await _drive()(widget="SetButton", verb="click")
    assert "phoebus.agent_access" in ctx["envelope"]["error_message"]
    post.assert_not_called()


async def test_unknown_value_refuses_a_drive_naming_the_key_and_value(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {"agent_access": "write"})
    with patch(f"{_MOD}._http_post_drive") as post:
        with assert_raises_error(error_type="configuration_error") as ctx:
            await _drive()(widget="SetButton", verb="click")
    message = ctx["envelope"]["error_message"]
    assert "phoebus.agent_access" in message
    assert "'write'" in message
    post.assert_not_called()


async def test_read_refusal_precedes_argument_validation(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {"agent_access": "read"})
    with assert_raises_error(error_type="not_supported"):
        await _drive()(widget="0", verb="frobnicate")


async def test_read_write_drive_reaches_the_bridge(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, {"agent_access": "read_write"})
    with patch(f"{_MOD}._http_post_drive", return_value=(200, {"fired": True, "detail": "ok"})):
        result = await _drive()(widget="SetButton", verb="click")
    assert extract_response_dict(result)["status"] == "success"

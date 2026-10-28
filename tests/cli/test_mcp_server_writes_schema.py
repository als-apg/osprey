"""``mcp_servers.<n>.{hooks,writes}``: loaded, validated and persisted.

A facility server states the tools that write as ``writes:``, a list of bare
tool names. The build validates the list as written (it never asks the server
for its tool list) and persists it, with the server's ``hooks:``, into the
rendered ``claude_code.servers.<n>`` block the registry reads back. A server
spelled both as a profile ``mcp_servers`` entry and as a ``claude_code.servers``
definition in ``config:`` is refused, and an unknown per-server key warns until
the write-posture sunset instead of vanishing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_load
from osprey.cli.build_persistence import _persist_mcp_servers
from osprey.cli.build_profile_load import RETIRED_WRITE_KEYS_REFUSED_FROM, _parse_profile
from osprey.cli.build_profile_schema import McpServerDef
from osprey_connectors.errors import BuildProfileError

_LOADER_LOGGER = "osprey.cli.build_profile_load"


@pytest.fixture(autouse=True)
def _fresh_warnings():
    build_profile_load._WARNED_MCP_SERVER_KEYS.clear()
    yield
    build_profile_load._WARNED_MCP_SERVER_KEYS.clear()


def _raw(server: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {"name": "t", "mcp_servers": {"facility": server}}
    if config is not None:
        raw["config"] = config
    return raw


def _server(**extra: Any) -> dict[str, Any]:
    return {"url": "http://facility:9000/mcp", **extra}


# ── loading ──────────────────────────────────────────────────────────────


def test_writes_and_hooks_load_into_the_server_definition():
    profile = _parse_profile(
        _raw(_server(writes=["set_magnet", "ramp-rf"], hooks={"pre_tool_use": ["approval"]}))
    )
    server = profile.mcp_servers["facility"]
    assert server.writes == ["set_magnet", "ramp-rf"]
    assert server.hooks == {"pre_tool_use": ["approval"]}


def test_a_server_without_writes_states_none():
    server = _parse_profile(_raw(_server())).mcp_servers["facility"]
    assert server.writes is None
    assert server.hooks == {}


def test_an_empty_writes_list_is_a_declaration_of_no_write_tools():
    assert _parse_profile(_raw(_server(writes=[]))).mcp_servers["facility"].writes == []


def test_the_definition_defaults_keep_existing_constructors_working():
    server = McpServerDef(command="uvx")
    assert server.writes is None
    assert server.hooks == {}


# ── CC-5: malformed writes ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("writes", "fragment"),
    [
        ("set_magnet", "must be a list"),
        ({"set_magnet": True}, "must be a list"),
        ([42], "not a string"),
        ([None], "not a string"),
        (["set_*"], "bare tool name"),
        (["set_.*"], "bare tool name"),
        (["set_(a|b)"], "bare tool name"),
        (["^set$"], "bare tool name"),
        ([""], "bare tool name"),
        (["set_magnet", "set_magnet"], "more than once"),
        (["mcp__facility__set_magnet"], "mcp__"),
    ],
)
def test_malformed_writes_is_a_build_error_naming_the_server(writes, fragment):
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(_raw(_server(writes=writes)))
    message = str(excinfo.value)
    assert "'facility'" in message
    assert fragment in message


@pytest.mark.parametrize(
    "hooks",
    [
        ["approval"],
        {"pre_tool_use": "approval"},
        {"pre_tool_use": [1]},
        {"post_tool_use": ["approval"]},
    ],
)
def test_malformed_hooks_is_a_build_error_naming_the_server(hooks):
    with pytest.raises(BuildProfileError, match="'facility'"):
        _parse_profile(_raw(_server(hooks=hooks)))


# ── both spellings ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config",
    [
        {"claude_code": {"servers": {"facility": {"url": "http://x:1/mcp"}}}},
        {"claude_code.servers.facility.writes": ["set_magnet"]},
        {"claude_code.servers.facility": {"command": "uvx"}},
        {"claude_code.servers.facility.hooks": {"pre_tool_use": ["approval"]}},
    ],
)
def test_a_server_spelled_under_both_keys_is_a_build_error(config):
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(_raw(_server(), config=config))
    message = str(excinfo.value)
    assert "'facility'" in message
    assert "mcp_servers" in message
    assert "claude_code.servers" in message


def test_an_enable_switch_beside_a_profile_server_is_not_a_second_spelling():
    profile = _parse_profile(
        _raw(_server(), config={"claude_code.servers.facility.enabled": False})
    )
    assert "facility" in profile.mcp_servers


def test_a_config_definition_of_another_server_is_not_a_clash():
    profile = _parse_profile(
        _raw(_server(), config={"claude_code.servers.other.url": "http://o:1/mcp"})
    )
    assert "facility" in profile.mcp_servers


# ── unknown per-server keys ───────────────────────────────────────────────


def test_an_unknown_per_server_key_warns_naming_the_server_and_the_sunset(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
        _parse_profile(_raw(_server(write=["set_magnet"])))
    warnings = [r.getMessage() for r in caplog.records if r.name == _LOADER_LOGGER]
    assert len(warnings) == 1
    assert "'facility'" in warnings[0]
    assert "'write'" in warnings[0]
    assert "writes" in warnings[0]  # the closest known spelling
    assert RETIRED_WRITE_KEYS_REFUSED_FROM in warnings[0]


def test_the_unknown_key_warning_is_given_once_per_server_and_key(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
        _parse_profile(_raw(_server(colour="red")))
        _parse_profile(_raw(_server(colour="red")))
    assert len([r for r in caplog.records if r.name == _LOADER_LOGGER]) == 1


def test_known_keys_draw_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
        _parse_profile(
            _raw(
                _server(
                    transport="http",
                    port=9000,
                    permissions={"allow": ["status"]},
                    writes=["set_magnet"],
                    hooks={"pre_tool_use": ["approval"]},
                )
            )
        )
    assert [r for r in caplog.records if r.name == _LOADER_LOGGER] == []


# ── persistence ───────────────────────────────────────────────────────────


def _persisted(tmp_path: Path, servers: dict[str, McpServerDef]) -> dict[str, Any]:
    (tmp_path / "config.yml").write_text("facility_name: test\n")
    _persist_mcp_servers(tmp_path, servers)
    return yaml.safe_load((tmp_path / "config.yml").read_text())["claude_code"]["servers"]


def test_writes_and_hooks_persist_into_claude_code_servers(tmp_path):
    servers = _persisted(
        tmp_path,
        {
            "facility": McpServerDef(
                url="http://facility:9000/mcp",
                writes=["set_magnet"],
                hooks={"pre_tool_use": ["approval"]},
            )
        },
    )
    assert servers["facility"]["writes"] == ["set_magnet"]
    assert servers["facility"]["hooks"] == {"pre_tool_use": ["approval"]}


def test_an_empty_writes_list_persists_as_a_declaration(tmp_path):
    servers = _persisted(tmp_path, {"facility": McpServerDef(command="uvx", writes=[])})
    assert servers["facility"]["writes"] == []


def test_a_server_without_writes_or_hooks_persists_as_before(tmp_path):
    servers = _persisted(tmp_path, {"facility": McpServerDef(command="uvx")})
    assert servers["facility"] == {"command": "uvx"}


def test_the_loaded_profile_round_trips_to_the_rendered_spec(tmp_path):
    profile = _parse_profile(
        _raw(_server(writes=["set_magnet"], hooks={"pre_tool_use": ["limits"]}))
    )
    servers = _persisted(tmp_path, profile.mcp_servers)
    assert servers["facility"]["writes"] == ["set_magnet"]
    assert servers["facility"]["hooks"] == {"pre_tool_use": ["limits"]}

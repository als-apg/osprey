"""The facility write lints: undeclared write-looking tools and ``control_system.write_tools``.

A facility server that declares ``writes:`` names its write tools itself. A
tool on such a server that asks for permission, or whose name says it destroys
state, and that the list leaves out, is gated as a write anyway; the build
warns so the list gets fixed before the sunset refuses the profile (PR-5). A
server with no list is not judged: it keeps today's whole-server gate.

``control_system.write_tools`` is the old way to name extra write tools; a
non-empty list warns and names the per-server key that replaces it (PR-5a).
Its gating is unchanged until the sunset.

Both ride the deploy-aware warnings path, so ``osprey build`` and
``osprey profile validate`` print the same lines.
"""

from __future__ import annotations

from typing import Any

import pytest
from packaging.version import Version

from osprey.cli.build_profile_deploy import (
    deploy_aware_config_errors,
    deploy_aware_config_warnings,
    facility_write_warnings,
)
from osprey.cli.build_profile_load import RETIRED_WRITE_KEYS_REFUSED_FROM
from osprey.cli.build_profile_schema import McpServerDef
from osprey.version import get_release_version


def _undeclared(warnings: list[str]) -> list[str]:
    return [w for w in warnings if "not in its `writes:` list" in w]


def _write_tools_key(warnings: list[str]) -> list[str]:
    return [w for w in warnings if "control_system.write_tools" in w]


def _server(*, ask=(), allow=(), writes=None) -> McpServerDef:
    perms: dict[str, list[str]] = {}
    if ask:
        perms["ask"] = list(ask)
    if allow:
        perms["allow"] = list(allow)
    return McpServerDef(command="run-it", permissions=perms, writes=writes)


# --- PR-5: undeclared write-looking tools on a declaring server ---------------


def test_an_asked_tool_missing_from_writes_warns() -> None:
    servers = {"magnets": _server(ask=["mcp__magnets__ramp"], writes=["set_value"])}

    (warning,) = _undeclared(deploy_aware_config_warnings(None, {}, mcp_servers=servers))

    assert "mcp_servers.magnets" in warning
    assert "`ramp`" in warning
    assert "permissions.ask" in warning
    assert "mcp_servers.magnets.writes" in warning
    assert f"refused from OSPREY {RETIRED_WRITE_KEYS_REFUSED_FROM}" in warning


@pytest.mark.parametrize("tool", ["delete_entry", "ClearCache", "purge"])
def test_a_destructive_name_missing_from_writes_warns(tool: str) -> None:
    servers = {"logbook": _server(allow=[f"mcp__logbook__{tool}"], writes=[])}

    (warning,) = _undeclared(facility_write_warnings({}, servers))

    assert f"`{tool}`" in warning
    assert "destructive" in warning
    assert "mcp_servers.logbook" in warning


def test_a_bare_permission_entry_is_judged_the_same() -> None:
    servers = {"magnets": _server(ask=["ramp"], writes=["set_value"])}

    (warning,) = _undeclared(facility_write_warnings({}, servers))

    assert "`ramp`" in warning


def test_a_declared_tool_is_not_flagged() -> None:
    servers = {
        "magnets": _server(
            ask=["mcp__magnets__set_value"],
            allow=["mcp__magnets__delete_setpoint", "mcp__magnets__get_value"],
            writes=["set_value", "delete_setpoint"],
        )
    }

    assert _undeclared(facility_write_warnings({}, servers)) == []


def test_an_allowed_plain_read_is_not_flagged() -> None:
    servers = {"magnets": _server(allow=["mcp__magnets__get_value"], writes=["set_value"])}

    assert _undeclared(facility_write_warnings({}, servers)) == []


def test_a_wildcard_permission_is_not_a_tool_name() -> None:
    servers = {"magnets": _server(ask=["mcp__magnets__.*"], writes=["set_value"])}

    assert _undeclared(facility_write_warnings({}, servers)) == []


def test_a_list_less_server_with_asked_tools_draws_no_pr5_warning() -> None:
    servers = {
        "magnets": _server(ask=["mcp__magnets__ramp", "mcp__magnets__delete_all"], writes=None)
    }

    warnings = deploy_aware_config_warnings(None, {}, mcp_servers=servers)

    assert _undeclared(warnings) == []
    assert deploy_aware_config_errors(None, {}) == []


def test_one_warning_per_undeclared_tool() -> None:
    servers = {
        "magnets": _server(
            ask=["mcp__magnets__ramp", "mcp__magnets__wipe_log"], writes=["set_value"]
        )
    }

    warnings = _undeclared(facility_write_warnings({}, servers))

    assert len(warnings) == 2
    assert any("`ramp`" in w for w in warnings)
    assert any("`wipe_log`" in w for w in warnings)


def test_a_dict_server_definition_is_read_the_same() -> None:
    servers: dict[str, Any] = {
        "magnets": {"permissions": {"ask": ["mcp__magnets__ramp"]}, "writes": ["set_value"]}
    }

    (warning,) = _undeclared(facility_write_warnings({}, servers))

    assert "`ramp`" in warning


def test_persisted_claude_code_servers_are_linted_too() -> None:
    config = {
        "claude_code": {
            "servers": {
                "magnets": {
                    "permissions": {"ask": ["mcp__magnets__ramp"]},
                    "writes": ["set_value"],
                }
            }
        }
    }

    (warning,) = _undeclared(deploy_aware_config_warnings(None, config))

    assert "magnets" in warning
    assert "`ramp`" in warning


def test_dotted_claude_code_servers_are_linted_too() -> None:
    config = {
        "claude_code.servers.magnets.permissions": {"ask": ["mcp__magnets__ramp"]},
        "claude_code.servers.magnets.writes": ["set_value"],
    }

    (warning,) = _undeclared(deploy_aware_config_warnings(None, config))

    assert "`ramp`" in warning


def test_a_server_in_both_places_warns_once() -> None:
    servers = {"magnets": _server(ask=["mcp__magnets__ramp"], writes=["set_value"])}
    config = {
        "claude_code": {
            "servers": {
                "magnets": {
                    "permissions": {"ask": ["mcp__magnets__ramp"]},
                    "writes": ["set_value"],
                }
            }
        }
    }

    assert len(_undeclared(facility_write_warnings(config, servers))) == 1


def test_a_disabled_persisted_server_is_not_linted() -> None:
    config = {
        "claude_code": {
            "servers": {
                "magnets": {
                    "enabled": False,
                    "permissions": {"ask": ["mcp__magnets__ramp"]},
                    "writes": ["set_value"],
                }
            }
        }
    }

    assert _undeclared(facility_write_warnings(config)) == []


def test_another_servers_prefix_is_not_this_servers_tool() -> None:
    servers = {"magnets": _server(ask=["mcp__other__ramp"], writes=["set_value"])}

    (warning,) = _undeclared(facility_write_warnings({}, servers))

    # The entry is judged as written: it is not this server's prefix, so the
    # whole entry is the name the warning quotes.
    assert "`mcp__other__ramp`" in warning


# --- PR-5a: control_system.write_tools ---------------------------------------


def test_a_non_empty_write_tools_list_warns() -> None:
    config = {"control_system": {"write_tools": ["mcp__magnets__ramp"]}}

    (warning,) = _write_tools_key(deploy_aware_config_warnings(None, config))

    assert "use `mcp_servers.<server>.writes`" in warning
    assert f"refused from OSPREY {RETIRED_WRITE_KEYS_REFUSED_FROM}" in warning
    assert "mcp__magnets__ramp" in warning


def test_a_dotted_write_tools_list_warns_the_same() -> None:
    config = {"control_system.write_tools": ["mcp__magnets__ramp"]}

    (warning,) = _write_tools_key(deploy_aware_config_warnings(None, config))

    assert "use `mcp_servers.<server>.writes`" in warning


@pytest.mark.parametrize("value", [[], None])
def test_an_empty_or_absent_write_tools_list_is_silent(value: Any) -> None:
    config: dict[str, Any] = {"control_system": {"type": "epics"}}
    if value is not None:
        config["control_system"]["write_tools"] = value

    assert _write_tools_key(facility_write_warnings(config)) == []


def test_neither_lint_refuses_inside_the_window() -> None:
    servers = {"magnets": _server(ask=["mcp__magnets__ramp"], writes=["set_value"])}
    config = {"control_system": {"write_tools": ["mcp__magnets__ramp"]}}

    assert deploy_aware_config_errors(None, config) == []
    assert len(facility_write_warnings(config, servers)) == 2


# --- the sunset pin -----------------------------------------------------------


def test_the_facility_write_window_is_still_open() -> None:
    """Fails on the sunset release: both lints must become refusals there.

    The release that reaches the sunset moves :func:`facility_write_warnings`
    from the deploy-aware warnings to the deploy-aware errors, and removes this
    test.
    """
    assert Version(get_release_version()) < Version(RETIRED_WRITE_KEYS_REFUSED_FROM)

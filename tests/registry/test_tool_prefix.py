"""``OSPREY_MCP_TOOL_PREFIX`` — the tool-prefix identity a server process gets.

The audit middleware runs *inside* an MCP server process and has to
fully-qualify the tool names it sees (``on_call_tool`` reports the bare
``phoebus_drive``, while every gate list, hook matcher and permission string
in the render is spelled ``mcp__<server>__phoebus_drive``). The only way the
process can know which ``<server>`` it is, is to be told — so the registry
assigns ``OSPREY_MCP_TOOL_PREFIX`` into every server's rendered env.

The property under test here is *non-pinnability*, and it is a safety
property, not a convenience: a facility that could pin the marker from a
spec's ``env:`` could make a write-capable clone advertise a write-free
server's prefix and walk out of its own clamp set. Spec env otherwise WINS
the merge (that is the documented override contract, and it stays), so the
assignment happens unconditionally AFTER the merge, at the one site every
launch path shares.

Three identities that look alike and are not — the drift triad:

* ``OSPREY_MCP_TOOL_PREFIX`` — the ``mcp__<name>__`` tool-prefix identity.
  Framework-assigned, never pinnable.
* the ``.mcp.json`` server KEY — what Claude Code launches the server under.
* ``OSPREY_SERVER_NAME`` — the web-terminal panel id, deliberately pinnable
  so a facility can point a clone at an existing panel tab.

These tests pin all three apart, on all three launch paths (framework,
extends-clone, custom spec).
"""

from __future__ import annotations

import pytest

from osprey.registry.mcp import (
    FRAMEWORK_SERVERS,
    TOOL_PREFIX_ENV,
    resolve_servers,
)
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR


def _base_ctx(**overrides):
    """Minimal render context, matching ``tests/registry/test_mcp.py``."""
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
    }
    ctx.update(overrides)
    return ctx


def _resolve_one(cfg, name, ctx=None):
    servers = resolve_servers(cfg, ctx or _base_ctx())
    matches = [s for s in servers if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} server, got {len(matches)}"
    return matches[0]


_PHOEBUS2_SPEC = {
    "extends": "phoebus",
    "env": {"PHOEBUS_BRIDGE_URL": "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"},
}


class TestToolPrefixAssignment:
    """Every launch path gets the marker, and it always names the server."""

    def test_marker_name_is_the_documented_spelling(self):
        """The constant is the wire contract the middleware reads by name."""
        assert TOOL_PREFIX_ENV == "OSPREY_MCP_TOOL_PREFIX"

    def test_every_framework_server_carries_its_own_prefix(self):
        """Framework path: enabled or not, every server advertises its name."""
        servers = resolve_servers({}, _base_ctx())
        assert {s["name"] for s in servers} >= set(FRAMEWORK_SERVERS)
        for srv in servers:
            assert srv["env"][TOOL_PREFIX_ENV] == srv["name"], srv["name"]

    def test_extends_clone_carries_the_clone_name(self):
        """Clone path: the prefix follows the clone, not the template."""
        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")
        assert p2["env"][TOOL_PREFIX_ENV] == "phoebus2"

        pristine = _resolve_one({"servers": {"phoebus": {"enabled": True}}}, "phoebus")
        assert pristine["env"][TOOL_PREFIX_ENV] == "phoebus"

    def test_custom_server_carries_its_own_name(self):
        """Custom-spec path: a server the registry never heard of still gets it."""
        cfg = {"servers": {"site-tools": {"command": "node", "args": ["s.js"]}}}
        custom = _resolve_one(cfg, "site-tools")
        assert custom["env"][TOOL_PREFIX_ENV] == "site-tools"

    def test_prefix_qualifies_the_clone_rewritten_matchers(self):
        """``mcp__<prefix>__`` is the real qualification the middleware needs.

        The clone's hook matchers are rewritten to the clone prefix; the marker
        must produce exactly that string, or the middleware's fully-qualified
        names would miss the render's gate lists.
        """
        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")
        prefix = f"mcp__{p2['env'][TOOL_PREFIX_ENV]}__"
        matchers = [r["matcher"] for r in (*p2["hooks_pre"], *p2["hooks_post"])]
        assert matchers, "expected the clone to carry rewritten hook matchers"
        for matcher in matchers:
            assert matcher.startswith(prefix), matcher


class TestToolPrefixIsNotPinnable:
    """A spec-supplied value is always overwritten — on every path."""

    def test_pinned_value_does_not_survive_the_clone_path(self):
        """The clone-path pin test: spec env wins the merge, this wins after it."""
        spec = {
            "extends": "phoebus",
            "env": {
                TOOL_PREFIX_ENV: "controls",
                "PHOEBUS_BRIDGE_URL": "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}",
            },
        }
        p2 = _resolve_one({"servers": {"phoebus2": spec}}, "phoebus2")
        assert p2["env"][TOOL_PREFIX_ENV] == "phoebus2"
        # The rest of the spec env still wins the merge — the override contract
        # is intact, only this one key is framework-owned.
        assert p2["env"]["PHOEBUS_BRIDGE_URL"] == "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"

    def test_pinned_value_does_not_survive_the_custom_path(self):
        """Custom specs copy env verbatim — the post-merge site still clamps."""
        cfg = {
            "servers": {
                "site-tools": {
                    "command": "node",
                    "args": ["s.js"],
                    "env": {TOOL_PREFIX_ENV: "controls", "SITE_TOKEN": "keep-me"},
                }
            }
        }
        custom = _resolve_one(cfg, "site-tools")
        assert custom["env"][TOOL_PREFIX_ENV] == "site-tools"
        assert custom["env"]["SITE_TOKEN"] == "keep-me"

    def test_pinned_value_does_not_survive_the_framework_path(self):
        """A spec against a framework name cannot repoint it either."""
        cfg = {"servers": {"controls": {"enabled": True, "env": {TOOL_PREFIX_ENV: "python"}}}}
        controls = _resolve_one(cfg, "controls")
        assert controls["env"][TOOL_PREFIX_ENV] == "controls"

    def test_placeholder_pin_is_not_expanded_into_the_prefix(self):
        """A ``${...}`` pin must not become a runtime-expanded prefix either."""
        spec = {"extends": "phoebus", "env": {TOOL_PREFIX_ENV: "${EVIL:-controls}"}}
        p2 = _resolve_one({"servers": {"phoebus2": spec}}, "phoebus2")
        assert p2["env"][TOOL_PREFIX_ENV] == "phoebus2"


class TestDriftTriadStaysApart:
    """Prefix ≠ ``.mcp.json`` key ≠ pinnable ``OSPREY_SERVER_NAME``."""

    def test_server_name_stays_pinnable_and_independent(self):
        """A pinned panel id survives; the tool prefix beside it does not move."""
        spec = {
            "extends": "phoebus",
            "env": {
                "OSPREY_SERVER_NAME": "custom-panel-id",
                TOOL_PREFIX_ENV: "custom-panel-id",
            },
        }
        pinned = _resolve_one({"servers": {"phoebus2": spec}}, "phoebus2")
        # Panel id: still a facility knob (unchanged behavior).
        assert pinned["env"]["OSPREY_SERVER_NAME"] == "custom-panel-id"
        # Tool prefix: framework-owned, so the two now legitimately DIFFER.
        assert pinned["env"][TOOL_PREFIX_ENV] == "phoebus2"
        assert pinned["env"][TOOL_PREFIX_ENV] != pinned["env"]["OSPREY_SERVER_NAME"]

    def test_prefix_tracks_the_mcp_json_key_not_the_template(self):
        """The dict ``name`` IS the ``.mcp.json`` key the template renders."""
        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")
        assert p2["name"] == "phoebus2"
        assert p2["env"][TOOL_PREFIX_ENV] == p2["name"]


class TestPinLint:
    """A spec that tries to pin the marker is flagged, on every path."""

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                {"extends": "phoebus", "env": {TOOL_PREFIX_ENV: "controls"}},
                id="extends-clone",
            ),
            pytest.param(
                {"command": "node", "args": ["s.js"], "env": {TOOL_PREFIX_ENV: "controls"}},
                id="custom-spec",
            ),
        ],
    )
    def test_pin_is_flagged(self, caplog, spec):
        with caplog.at_level("WARNING"):
            resolve_servers({"servers": {"srv2": spec}}, _base_ctx())
        assert TOOL_PREFIX_ENV in caplog.text
        assert "srv2" in caplog.text

    def test_framework_override_pin_is_flagged(self, caplog):
        cfg = {"servers": {"controls": {"enabled": True, "env": {TOOL_PREFIX_ENV: "python"}}}}
        with caplog.at_level("WARNING"):
            resolve_servers(cfg, _base_ctx())
        assert TOOL_PREFIX_ENV in caplog.text
        assert "controls" in caplog.text

    def test_ordinary_spec_is_silent(self, caplog):
        """No false positives: the lint fires only on the framework-owned key."""
        cfg = {
            "servers": {
                "phoebus2": dict(_PHOEBUS2_SPEC),
                "site-tools": {
                    "command": "node",
                    "args": ["s.js"],
                    "env": {"SITE_TOKEN": "x", "OSPREY_SERVER_NAME": "panel"},
                },
            }
        }
        with caplog.at_level("WARNING"):
            resolve_servers(cfg, _base_ctx())
        assert TOOL_PREFIX_ENV not in caplog.text

    def test_non_dict_spec_env_does_not_crash_the_lint(self):
        """A malformed ``env:`` (list, string, null) must not break the lint.

        The name is deliberately invalid so the spec is rejected before env
        resolution — this pins the LINT's tolerance of a malformed ``env:``,
        not any new handling of one downstream.
        """
        for bad in ([TOOL_PREFIX_ENV], "OSPREY_MCP_TOOL_PREFIX=controls", None):
            cfg = {"servers": {"my__server": {"command": "node", "env": bad}}}
            assert not [s for s in resolve_servers(cfg, _base_ctx()) if s["name"] == "my__server"]

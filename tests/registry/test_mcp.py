"""Tests for the unified server and agent registry."""

import json
import logging

import pytest

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.registry.mcp import HOOK_PRESETS, resolve_agents, resolve_servers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_ctx(**overrides):
    """Build a minimal template context for testing."""
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Server resolution tests
# ---------------------------------------------------------------------------


class TestResolveServers:
    """Tests for resolve_servers()."""

    def test_resolve_default_config(self):
        """No overrides → core servers enabled, optional servers disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        enabled = {s["name"] for s in servers if s["enabled"]}
        disabled = {s["name"] for s in servers if not s["enabled"]}

        # Core servers always on
        assert {"controls", "osprey_workspace", "ariel"} <= enabled
        # Conditional servers off (conditions not in ctx); opt-in servers off by default
        assert {"channel-finder", "health"} <= disabled

    def test_resolve_disable_framework_server(self):
        """New format: servers: {ariel: {enabled: false}}."""
        ctx = _base_ctx()
        servers = resolve_servers({"servers": {"ariel": {"enabled": False}}}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["ariel"] is False
        # Other core servers still enabled
        assert names["controls"] is True

    def test_resolve_custom_server(self):
        """Custom server in new format appears with correct structure."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"MY_KEY": "val"},
                    "permissions": {"allow": ["read_data"], "ask": ["write_data"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        custom = [s for s in servers if s["name"] == "my-server"]
        assert len(custom) == 1
        s = custom[0]
        assert s["enabled"] is True
        assert s["command"] == "node"
        assert s["args"] == ["server.js"]
        assert s["env"] == {"MY_KEY": "val"}
        assert s["permissions_allow"] == ["read_data"]
        assert s["permissions_ask"] == ["write_data"]
        assert s["is_custom"] is True

    def test_resolve_custom_server_url_transport(self):
        """Custom URL/SSE server resolves with url field, empty command."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "remote-api": {
                    "url": "http://remote:8001/sse",
                    "permissions": {"allow": ["search"], "ask": []},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        remote = [s for s in servers if s["name"] == "remote-api"]
        assert len(remote) == 1
        s = remote[0]
        assert s["url"] == "http://remote:8001/sse"
        assert s["command"] == ""
        assert s["args"] == []
        assert s["permissions_allow"] == ["search"]
        assert s["is_custom"] is True

    def test_resolve_conditional_server_enabled(self):
        """channel_finder_pipeline in ctx → channel-finder server enabled."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["channel-finder"] is True

    def test_resolve_conditional_server_disabled(self):
        """channel_finder_pipeline not in ctx → channel-finder server disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["channel-finder"] is False

    def test_env_resolution(self):
        """Placeholder {project_root} in env values is resolved."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        controls = [s for s in servers if s["name"] == "controls"][0]
        assert controls["env"]["OSPREY_CONFIG"] == "/tmp/test-project/config.yml"
        # Shell variables ${...} are preserved
        assert controls["env"]["EPICS_CA_ADDR_LIST"] == "${EPICS_CA_ADDR_LIST:-}"

    def test_all_python_servers_set_config_file(self):
        """Every framework python MCP server must set CONFIG_FILE, not just OSPREY_CONFIG.

        osprey.utils.config reads CONFIG_FILE (OSPREY_CONFIG is only used to locate
        .env). When a server subprocess is launched with a CWD other than the
        project dir (e.g. the dispatch worker's /app WORKDIR), a missing CONFIG_FILE
        makes config resolution fall back to CWD/config.yml and fail. Regression
        guard: osprey_workspace / ariel / channel-finder used to omit it.
        """
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        expected = "/tmp/test-project/config.yml"
        for name in ("controls", "python", "osprey_workspace", "ariel", "health", "channel-finder"):
            srv = [s for s in servers if s["name"] == name][0]
            assert srv["env"].get("CONFIG_FILE") == expected, (
                f"{name} must set CONFIG_FILE={expected!r}, got {srv['env'].get('CONFIG_FILE')!r}"
            )

    def test_health_server_entry(self):
        """The health server is an opt-in, read-only server.

        Off by default (opt-in via claude_code.servers.health.enabled), module runs as
        python -m osprey.mcp_server.health, and its allow/ask split puts health_check on
        silent-allow while health_check_full is approval-gated. Being read-only (tools
        take only a categories filter, no channel/URL/probe params), it carries NO
        pre-tool-use write/approval hooks — only the standard PostToolUse error guidance.
        """
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        health = [s for s in servers if s["name"] == "health"][0]
        # Opt-in: off by default.
        assert health["enabled"] is False
        # Module path renders as python -m osprey.mcp_server.health.
        assert health["args"] == ["-m", "osprey.mcp_server.health"]
        # Allow/ask split.
        assert health["permissions_allow"] == ["health_check"]
        assert health["permissions_ask"] == ["health_check_full"]
        # Read-only: no PreToolUse (write/approval) hooks at all.
        assert health["hooks_pre"] == []
        # Only the standard PostToolUse error-guidance hook.
        assert [r["matcher"] for r in health["hooks_post"]] == ["mcp__health__.*"]

    def test_health_server_enabled_via_config_override(self):
        """control-assistant opts the health server in via the dotted override
        ``claude_code.servers.health.enabled: true`` (renders to this config).

        Mirrors the bluesky enablement pattern: the preset's ``claude_code``
        config flips the off-by-default server on.
        """
        ctx = _base_ctx()
        servers = resolve_servers({"servers": {"health": {"enabled": True}}}, ctx)
        health = [s for s in servers if s["name"] == "health"][0]
        assert health["enabled"] is True

    def test_health_server_disabled_without_override(self):
        """hello-world ships no ``claude_code.servers.health`` block, so the
        server stays disabled (default_enabled=False) — off by default."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        health = [s for s in servers if s["name"] == "health"][0]
        assert health["enabled"] is False

    def test_controls_hook_structure(self):
        """controls server has 3 distinct PreToolUse hook rules."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        controls = [s for s in servers if s["name"] == "controls"][0]
        pre_matchers = [r["matcher"] for r in controls["hooks_pre"]]
        assert "mcp__controls__channel_write" in pre_matchers
        assert "mcp__controls__channel_read" in pre_matchers
        assert "mcp__controls__archiver_read" in pre_matchers
        # channel_write has 3 hooks
        cw = [r for r in controls["hooks_pre"] if r["matcher"] == "mcp__controls__channel_write"][0]
        assert len(cw["hooks"]) == 3

    def test_channel_finder_pipeline_resolution(self):
        """channel-finder server resolves {channel_finder_pipeline} in module."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        cf = [s for s in servers if s["name"] == "channel-finder"][0]
        assert cf["enabled"] is True
        assert cf["args"] == ["-m", "osprey.mcp_server.channel_finder_hierarchical"]

    def test_custom_server_with_approval_preset(self):
        """Custom server with hooks.pre_tool_use: [approval] gets approval hook."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        assert plc["enabled"] is True
        assert len(plc["hooks_pre"]) == 1
        rule = plc["hooks_pre"][0]
        assert rule["matcher"] == "mcp__my-plc__.*"
        assert len(rule["hooks"]) == 1
        assert "osprey_approval.py" in rule["hooks"][0]["command"]

    def test_custom_server_with_multiple_presets(self):
        """Custom server with multiple hook presets gets all hooks in one rule."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval", "writes_check"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        assert len(plc["hooks_pre"]) == 1
        rule = plc["hooks_pre"][0]
        assert len(rule["hooks"]) == 2
        commands = [h["command"] for h in rule["hooks"]]
        assert any("osprey_approval.py" in c for c in commands)
        assert any("osprey_writes_check.py" in c for c in commands)

    def test_custom_server_invalid_preset_warns(self, caplog):
        """Unknown hook preset logs a warning and is skipped."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval", "bogus"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        # Only the valid preset should be in the hooks
        assert len(plc["hooks_pre"]) == 1
        assert len(plc["hooks_pre"][0]["hooks"]) == 1
        assert "osprey_approval.py" in plc["hooks_pre"][0]["hooks"][0]["command"]
        # Warning should have been logged
        assert any("bogus" in r.message for r in caplog.records)

    def test_custom_server_no_hooks_key(self):
        """Custom server without hooks key gets no pre-tool-use hooks."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "permissions": {"allow": ["read_data"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        custom = [s for s in servers if s["name"] == "my-server"][0]
        assert custom["hooks_pre"] == []

    def test_hook_presets_dict_contains_expected_keys(self):
        """HOOK_PRESETS has the three expected preset names."""
        assert set(HOOK_PRESETS.keys()) == {"approval", "writes_check", "limits"}


# ---------------------------------------------------------------------------
# Extends (second framework-server instance) tests
# ---------------------------------------------------------------------------

_PHOEBUS2_SPEC = {
    "extends": "phoebus",
    "env": {"PHOEBUS_BRIDGE_URL": "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"},
}

_PHOEBUS_ALLOW = [
    "phoebus_list_displays",
    "phoebus_perceive",
    "phoebus_perceive_region",
    "phoebus_snapshot",
    "phoebus_open_panel",
    "phoebus_open_databrowser",
]

# Only drive actuates hardware-facing controls; open_panel touches no PVs
# and is plain-allowed — see the registry entry.
_PHOEBUS_ASK = ["phoebus_drive"]


def _resolve_one(cfg, name, ctx=None):
    servers = resolve_servers(cfg, ctx or _base_ctx())
    matches = [s for s in servers if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} server, got {len(matches)}"
    return matches[0]


class TestExtendsServers:
    """Tests for extends clones (claude_code.servers.<name>.extends)."""

    def test_golden_parity_with_deleted_framework_entry(self):
        """The extends clone reproduces the old hand-written framework phoebus2
        entry field-for-field (golden capture taken BEFORE the deletion)."""
        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")

        assert p2["enabled"] is True  # despite template default_enabled=False
        assert p2["command"] == "/usr/bin/python3"
        assert p2["args"] == ["-m", "osprey.mcp_server.phoebus"]
        # Spec env wins, ${...} NOT expanded; template keys survive with
        # {project_root} resolved.
        # OSPREY_SERVER_NAME is a deliberate post-golden addition (instance
        # identity for UI signals, e.g. open_panel → panel_focus) — auto-set
        # to the clone's own name, not inherited from the template.
        assert p2["env"] == {
            "OSPREY_CONFIG": "/tmp/test-project/config.yml",
            "CONFIG_FILE": "/tmp/test-project/config.yml",
            "PHOEBUS_BRIDGE_URL": "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}",
            "OSPREY_SERVER_NAME": "phoebus2",
        }
        # Permissions inherited as bare names (unrewritten).
        assert p2["permissions_allow"] == _PHOEBUS_ALLOW
        assert p2["permissions_ask"] == _PHOEBUS_ASK
        # Hook matchers rewritten with the anchored prefix.
        assert len(p2["hooks_pre"]) == 1
        assert p2["hooks_pre"][0]["matcher"] == "mcp__phoebus2__phoebus_drive"
        assert len(p2["hooks_pre"][0]["hooks"]) == 1
        assert "osprey_approval.py" in p2["hooks_pre"][0]["hooks"][0]["command"]
        assert [r["matcher"] for r in p2["hooks_post"]] == ["mcp__phoebus2__.*"]
        assert p2["is_custom"] is False
        assert p2["url"] is None

    def test_server_name_env_identity(self):
        """Each instance's env advertises its own name via OSPREY_SERVER_NAME:
        the template keeps 'phoebus', a clone is auto-rewritten to the clone
        name, and an explicit spec env pin wins over the auto-rewrite."""
        pristine = _resolve_one({"servers": {"phoebus": {"enabled": True}}}, "phoebus")
        assert pristine["env"]["OSPREY_SERVER_NAME"] == "phoebus"

        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")
        assert p2["env"]["OSPREY_SERVER_NAME"] == "phoebus2"

        pinned_spec = {
            "extends": "phoebus",
            "env": {"OSPREY_SERVER_NAME": "custom-panel-id"},
        }
        pinned = _resolve_one({"servers": {"phoebus2": pinned_spec}}, "phoebus2")
        assert pinned["env"]["OSPREY_SERVER_NAME"] == "custom-panel-id"

    def test_no_bare_name_rewrite(self):
        """The matcher rewrite is an anchored prefix splice — 'phoebus2_drive'
        (the bare-name-replace bug class) must appear NOWHERE in any matcher or
        permission string."""
        p2 = _resolve_one({"servers": {"phoebus2": dict(_PHOEBUS2_SPEC)}}, "phoebus2")

        strings = [r["matcher"] for r in (*p2["hooks_pre"], *p2["hooks_post"])]
        strings += p2["permissions_allow"] + p2["permissions_ask"]
        strings += p2["fixed_allow"] + p2["fixed_ask"]
        for s in strings:
            assert "phoebus2_drive" not in s, f"bare-name rewrite corruption in {s!r}"

    def test_template_isolation(self):
        """Resolving an extends spec must not mutate the phoebus template —
        neither in the same pass nor in a subsequent resolve_servers() call."""
        cfg = {"servers": {"phoebus": {"enabled": True}, "phoebus2": dict(_PHOEBUS2_SPEC)}}
        servers = resolve_servers(cfg, _base_ctx())
        phoebus = [s for s in servers if s["name"] == "phoebus"][0]
        assert phoebus["hooks_pre"][0]["matcher"] == "mcp__phoebus__phoebus_drive"
        assert (
            phoebus["env"]["PHOEBUS_BRIDGE_URL"] == "${PHOEBUS_BRIDGE_URL:-http://127.0.0.1:7979}"
        )

        # A fresh resolve with no overrides yields the pristine template.
        pristine = _resolve_one({}, "phoebus")
        assert pristine["hooks_pre"][0]["matcher"] == "mcp__phoebus__phoebus_drive"
        assert pristine["hooks_post"][0]["matcher"] == "mcp__phoebus__.*"

    def test_enabled_semantics(self):
        """Declared ⇒ enabled (template default_enabled=False must not leak);
        enabled: false ⇒ absent; independent of the template's own enablement."""
        # No 'enabled' key → enabled, even with the template explicitly disabled.
        cfg = {"servers": {"phoebus": {"enabled": False}, "phoebus2": dict(_PHOEBUS2_SPEC)}}
        servers = resolve_servers(cfg, _base_ctx())
        by_name = {s["name"]: s for s in servers}
        assert by_name["phoebus2"]["enabled"] is True
        assert by_name["phoebus"]["enabled"] is False

        # enabled: false → clone absent entirely.
        cfg = {"servers": {"phoebus2": {**_PHOEBUS2_SPEC, "enabled": False}}}
        servers = resolve_servers(cfg, _base_ctx())
        assert "phoebus2" not in {s["name"] for s in servers}

    def test_order_independence(self):
        """Clone output is identical regardless of YAML declaration order
        relative to a template override (copy-from-pristine-FRAMEWORK_SERVERS)."""
        cfg_a = {"servers": {"phoebus": {"enabled": True}, "phoebus2": dict(_PHOEBUS2_SPEC)}}
        cfg_b = {"servers": {"phoebus2": dict(_PHOEBUS2_SPEC), "phoebus": {"enabled": True}}}
        p2_a = _resolve_one(cfg_a, "phoebus2")
        p2_b = _resolve_one(cfg_b, "phoebus2")
        assert p2_a == p2_b
        # Both servers enabled; template env/permissions unchanged either way.
        phoebus = _resolve_one(cfg_b, "phoebus")
        assert phoebus["enabled"] is True
        assert (
            phoebus["env"]["PHOEBUS_BRIDGE_URL"] == "${PHOEBUS_BRIDGE_URL:-http://127.0.0.1:7979}"
        )

    def test_unknown_target_warns_and_skips(self, caplog):
        """Unknown extends target → warning, no server emitted."""
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers(
                {"servers": {"phoebus2": {"extends": "phobeus"}}}, _base_ctx()
            )
        assert "phoebus2" not in {s["name"] for s in servers}
        assert any("phobeus" in r.message for r in caplog.records)

    @pytest.mark.parametrize("order", ["a_first", "b_first"])
    def test_chaining_warns_and_skips(self, caplog, order):
        """Extends of an extends/custom server (chaining) → warn + skip the
        chained server, regardless of declaration order; the base clone survives."""
        specs = [("a", {"extends": "phoebus"}), ("b", {"extends": "a"})]
        if order == "b_first":
            specs.reverse()
        cfg = {"servers": dict(specs)}
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers(cfg, _base_ctx())
        names = {s["name"] for s in servers}
        assert "a" in names
        assert "b" not in names
        assert any("'a'" in r.message or '"a"' in r.message for r in caplog.records)

    def test_framework_name_shadowing_warns_keeps_framework_definition(self, caplog):
        """servers.phoebus: {extends: controls} must not rebase phoebus — the
        extends key is rejected loudly, only 'enabled' applies."""
        cfg = {"servers": {"phoebus": {"extends": "controls", "enabled": True}}}
        with caplog.at_level(logging.WARNING):
            phoebus = _resolve_one(cfg, "phoebus")
        # Framework definition intact: module, matchers, permissions.
        assert phoebus["args"] == ["-m", "osprey.mcp_server.phoebus"]
        assert phoebus["hooks_pre"][0]["matcher"] == "mcp__phoebus__phoebus_drive"
        assert "channel_write" not in phoebus["permissions_ask"]
        # The enabled toggle still applies; the extends key warned.
        assert phoebus["enabled"] is True
        assert any("extends" in r.message and "phoebus" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "bad_name", ["phoebus(2", "phoebus.2", "phoe__bus2", "-phoebus2", "controls_"]
    )
    def test_invalid_clone_name_rejected(self, caplog, bad_name):
        """Clone names are spliced into regexes and prefix matches — reject
        anything outside [A-Za-z0-9][A-Za-z0-9_-]*, containing '__', or ending
        in '_' ('controls_' → prefix 'mcp__controls___' startswith-collides
        with 'mcp__controls__' and corrupts approval short-name extraction)."""
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers({"servers": {bad_name: {"extends": "phoebus"}}}, _base_ctx())
        assert bad_name not in {s["name"] for s in servers}
        assert any("Invalid extends server name" in r.message for r in caplog.records)

    @pytest.mark.parametrize("good_name", ["phoebus2", "phoebus-2"])
    def test_valid_clone_name_accepted(self, good_name):
        """The tightened name validation still accepts ordinary clone names
        (trailing digit, trailing hyphen-digit)."""
        clone = _resolve_one({"servers": {good_name: {"extends": "phoebus"}}}, good_name)
        assert clone["enabled"] is True
        assert clone["hooks_pre"][0]["matcher"] == f"mcp__{good_name}__phoebus_drive"

    def test_non_string_extends_target_warns_and_skips(self, caplog):
        """A non-string extends value (e.g. ``extends: [phoebus]``) must
        warn+skip like an unknown target, not TypeError on the dict lookup."""
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers(
                {"servers": {"phoebus2": {"extends": ["phoebus"]}}}, _base_ctx()
            )
        assert "phoebus2" not in {s["name"] for s in servers}
        assert any("Unknown extends target" in r.message for r in caplog.records)

    def test_duplicate_allow_override_cannot_defeat_ask_union(self, caplog):
        """Duplicated entries in an override's permissions.allow must not defeat
        the single-.remove() ask-union guard: drive ends ONLY in ask."""
        cfg = {
            "servers": {
                "phoebus2": {
                    "extends": "phoebus",
                    "permissions": {"allow": ["phoebus_drive", "phoebus_drive"], "ask": []},
                }
            }
        }
        with caplog.at_level(logging.WARNING):
            p2 = _resolve_one(cfg, "phoebus2")
        assert "phoebus_drive" in p2["permissions_ask"]
        assert "phoebus_drive" not in p2["permissions_allow"]

    def test_fixed_lists_rewritten(self, monkeypatch):
        """fixed_allow/fixed_ask hold fully-qualified mcp__<server>__ strings —
        the clone must get them rewritten with the anchored prefix (latent
        today: no framework server sets them, so use a synthetic template)."""
        from osprey.registry.mcp import (
            FRAMEWORK_SERVERS,
            ServerDefinition,
            build_extended_server,
        )

        synthetic = ServerDefinition(
            name="synth",
            module="osprey.mcp_server.synth",
            fixed_allow=["mcp__synth__widget_read", "Read(_agent_data/**)"],
            fixed_ask=["mcp__synth__widget_write"],
        )
        monkeypatch.setitem(FRAMEWORK_SERVERS, "synth", synthetic)

        clone = build_extended_server("synth2", {"extends": "synth"})
        assert clone is not None
        assert clone.fixed_allow == ["mcp__synth2__widget_read", "Read(_agent_data/**)"]
        assert clone.fixed_ask == ["mcp__synth2__widget_write"]
        # Template untouched.
        assert synthetic.fixed_ask == ["mcp__synth__widget_write"]

    @pytest.mark.parametrize("with_pipeline", [False, True])
    def test_conditioned_template_skipped(self, caplog, with_pipeline):
        """Extends of a conditioned/dynamic template (channel-finder) is not
        supported: warn + skip, and no rendered dict may carry an unresolved
        '{channel_finder_pipeline}' placeholder."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical") if with_pipeline else _base_ctx()
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers({"servers": {"cf2": {"extends": "channel-finder"}}}, ctx)
        assert "cf2" not in {s["name"] for s in servers}
        assert any("conditioned" in r.message for r in caplog.records)
        # No ENABLED server (i.e. rendered into .mcp.json) may carry the broken
        # 'osprey.mcp_server.channel_finder_' module or an unresolved placeholder.
        for s in servers:
            if not s["enabled"]:
                continue
            for a in s["args"]:
                assert "{channel_finder_pipeline}" not in a
                assert not a.endswith("channel_finder_")

    def test_legacy_phoebus2_enabled_spec_warns_and_skips(self, caplog):
        """The old form {'phoebus2': {'enabled': True}} (framework entry now
        deleted; no extends/command/url) → warn + skip; no command=='' server
        may ever be emitted (the broken-.mcp.json regression)."""
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers({"servers": {"phoebus2": {"enabled": True}}}, _base_ctx())
        assert "phoebus2" not in {s["name"] for s in servers}
        assert any("phoebus2" in r.message and "extends" in r.message for r in caplog.records)
        for s in servers:
            assert not (s["url"] is None and s["command"] == ""), (
                f"server {s['name']!r} rendered with an empty command"
            )

    def test_ask_override_cannot_narrow(self, caplog):
        """An override that promotes phoebus_drive out of ask into allow is
        corrected: drive stays in ask (union invariant) and leaves allow."""
        cfg = {
            "servers": {
                "phoebus2": {
                    "extends": "phoebus",
                    "permissions": {"allow": ["phoebus_drive", "phoebus_perceive"], "ask": []},
                }
            }
        }
        with caplog.at_level(logging.WARNING):
            p2 = _resolve_one(cfg, "phoebus2")
        assert "phoebus_drive" in p2["permissions_ask"]
        assert "phoebus_drive" not in p2["permissions_allow"]
        assert "phoebus_perceive" in p2["permissions_allow"]
        assert any("phoebus_drive" in r.message for r in caplog.records)

    def test_ask_override_can_add(self):
        """Overrides may ADD approval-gated tools beyond the template's; the
        template's own ask set is re-added by the union invariant."""
        cfg = {
            "servers": {
                "phoebus2": {
                    "extends": "phoebus",
                    "permissions": {"ask": ["phoebus_extra_tool"]},
                }
            }
        }
        p2 = _resolve_one(cfg, "phoebus2")
        assert "phoebus_extra_tool" in p2["permissions_ask"]
        for tool in _PHOEBUS_ASK:
            assert tool in p2["permissions_ask"]

    def test_third_instance_scales(self):
        """Two extends clones (phoebus2 + phoebus3) render side by side with
        distinct prefixes — N-instance scaling without framework changes."""
        cfg = {
            "servers": {
                "phoebus2": dict(_PHOEBUS2_SPEC),
                "phoebus3": {
                    "extends": "phoebus",
                    "env": {"PHOEBUS_BRIDGE_URL": "${PHOEBUS3_BRIDGE_URL:-http://127.0.0.1:7981}"},
                },
            }
        }
        servers = resolve_servers(cfg, _base_ctx())
        by_name = {s["name"]: s for s in servers}
        for name in ("phoebus2", "phoebus3"):
            assert by_name[name]["enabled"] is True
            assert by_name[name]["hooks_pre"][0]["matcher"] == f"mcp__{name}__phoebus_drive"
            assert by_name[name]["permissions_ask"] == _PHOEBUS_ASK
        assert by_name["phoebus3"]["env"]["PHOEBUS_BRIDGE_URL"] == (
            "${PHOEBUS3_BRIDGE_URL:-http://127.0.0.1:7981}"
        )


# ---------------------------------------------------------------------------
# Agent resolution tests
# ---------------------------------------------------------------------------


class TestResolveAgents:
    """Tests for resolve_agents()."""

    def test_default_agents(self):
        """Default config → core agents enabled, conditional ones disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        enabled = {a["name"] for a in agents if a["enabled"]}
        disabled = {a["name"] for a in agents if not a["enabled"]}

        assert {"logbook-search", "logbook-deep-research"} <= enabled
        assert {"channel-finder"} <= disabled
        assert "data-visualizer" in enabled

    def test_disable_framework_agent(self):
        """New format: agents: {logbook-search: {enabled: false}}."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"logbook-search": {"enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["logbook-search"] is False

    def test_custom_agent_from_config(self):
        """Custom agent defined in config appears with correct fields."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"my-db-agent": {"description": "Searches the facility DB."}}},
            ctx,
            resolved_servers=servers,
        )
        custom = [a for a in agents if a["name"] == "my-db-agent"]
        assert len(custom) == 1
        assert custom[0]["enabled"] is True
        assert custom[0]["is_custom"] is True
        assert custom[0]["description"] == "Searches the facility DB."

    def test_custom_agent_with_condition_disabled(self):
        """Custom agent with condition key not in ctx → disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "cond-agent": {"description": "Needs feature.", "condition": "my_feature"}
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["cond-agent"] is False

    def test_custom_agent_with_condition_enabled(self):
        """Custom agent with condition key in ctx → enabled."""
        ctx = _base_ctx(my_feature=True)
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "cond-agent": {"description": "Needs feature.", "condition": "my_feature"}
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["cond-agent"] is True

    def test_custom_agent_with_server_dependency_disabled(self):
        """Custom agent with server_dependency on inactive server → disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "dep-agent": {
                        "description": "Depends on missing server.",
                        "server_dependency": "nonexistent-server",
                    }
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["dep-agent"] is False

    def test_custom_agent_with_server_dependency_enabled(self):
        """Custom agent with server_dependency on active server → enabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        # "controls" is enabled by default
        agents = resolve_agents(
            {
                "agents": {
                    "dep-agent": {
                        "description": "Depends on controls.",
                        "server_dependency": "controls",
                    }
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["dep-agent"] is True

    def test_custom_agent_explicitly_disabled(self):
        """Custom agent with enabled: false in config → not created."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"skip-agent": {"description": "Should not appear.", "enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"] for a in agents}
        assert "skip-agent" not in names

    def test_auto_discovery(self, tmp_path):
        """Custom agent .md file in .claude/agents/ appears in resolved list."""
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "my-agent.md").write_text(
            '---\nname: my-agent\ndescription: "A custom agent"\n---\n\n# My Agent\n'
        )
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, project_dir=tmp_path, resolved_servers=servers)
        custom = [a for a in agents if a["name"] == "my-agent"]
        assert len(custom) == 1
        assert custom[0]["is_custom"] is True
        assert custom[0]["enabled"] is True
        assert custom[0]["description"] == "A custom agent"


# ---------------------------------------------------------------------------
# Template rendering tests
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Tests that templates render valid JSON with the new data-driven approach."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _render(self, tm, template_path, ctx):
        template = tm.jinja_env.get_template(template_path)
        return template.render(**ctx)

    def _full_ctx(self, **overrides):
        """Build a context with servers and agents resolved."""
        ctx = _base_ctx(**overrides)
        claude_code_config = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers(claude_code_config, ctx)
        ctx["agents"] = resolve_agents(claude_code_config, ctx, resolved_servers=ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
        return ctx

    def test_render_mcp_json(self, template_manager):
        """Rendered mcp.json is valid JSON with expected servers."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "mcpServers" in data
        assert "controls" in data["mcpServers"]
        assert "osprey_workspace" in data["mcpServers"]
        # Conditional servers not present
        assert "channel-finder" not in data["mcpServers"]

    def test_render_mcp_json_with_conditional(self, template_manager):
        """Conditional servers appear when conditions are met."""
        ctx = self._full_ctx(
            channel_finder_pipeline="hierarchical",
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "channel-finder" in data["mcpServers"]

    def test_render_mcp_json_url_server(self, template_manager):
        """URL servers render as {type: http, url: ...} entries."""
        ctx = self._full_ctx(
            _claude_code_config={
                "servers": {
                    "remote-api": {
                        "url": "http://remote:8001/sse",
                    }
                }
            }
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "remote-api" in data["mcpServers"]
        remote = data["mcpServers"]["remote-api"]
        assert remote == {"type": "http", "url": "http://remote:8001/sse"}
        # Framework servers still have command/args, no type
        controls = data["mcpServers"]["controls"]
        assert "command" in controls
        assert "type" not in controls

    def test_topology_default_leaves_custom_url_server_untouched(self, template_manager):
        """Regression guard (web-terminals Task 2.5): the `modules.web_terminals.
        mcp.topology` fail-closed gate lives entirely in
        `osprey.deployment.web_terminals.render`, a module that never reads
        `claude_code.servers` and is never invoked by this project-level `.mcp.json`
        pipeline. A project's own custom `url`-transport server must keep rendering
        its `{type: "http", url: ...}` entry exactly as before, both on its own
        (this assertion) and with a sibling facility config that carries the
        default (or omitted) web-terminals topology present in the same overall
        config (the `render_web_terminals()` call below, which must not raise and
        must not consult `claude_code.servers` at all)."""
        # `.mcp.json` side: the custom url server still resolves and renders
        # exactly as it does without the topology key existing at all.
        ctx = self._full_ctx(
            _claude_code_config={
                "servers": {
                    "remote-api": {
                        "url": "http://remote:8001/sse",
                    }
                }
            }
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert data["mcpServers"]["remote-api"] == {
            "type": "http",
            "url": "http://remote:8001/sse",
        }

        # web-terminals side: a sibling facility config with the default topology
        # (here, simply omitted) renders clean, and the topology gate does not
        # implicate this claude_code.servers stanza in any way.
        facility_config = {
            "facility": {"name": "Demo Light Source", "prefix": "dls"},
            "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
            "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
            "modules": {
                "web_terminals": {
                    "enabled": True,
                    "nginx_port": 9080,
                    "web_base_port": 9091,
                    "artifact_base_port": 9291,
                    "ariel_base_port": 9391,
                    "lattice_base_port": 9491,
                    "users": ["alice"],
                }
            },
            # Same shape as the project-level config exercised above — present
            # here only to prove render_web_terminals() never looks at it.
            "claude_code": {"servers": {"remote-api": {"url": "http://remote:8001/sse"}}},
        }
        artifacts = render_web_terminals(facility_config)
        assert "docker-compose.web.yml" in artifacts

    def test_render_settings_json(self, template_manager):
        """Rendered settings.json is valid JSON with permissions and hooks."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)
        assert "permissions" in data
        assert "allow" in data["permissions"]
        assert "deny" in data["permissions"]
        assert "ask" in data["permissions"]
        # Check a sample allow entry
        allow = data["permissions"]["allow"]
        assert '"Read(_agent_data/**)"' not in allow  # Not double-quoted
        assert "Read(_agent_data/**)" in allow
        assert "mcp__osprey_workspace__data_read" in allow
        # Check ask entries
        ask = data["permissions"]["ask"]
        assert "mcp__controls__channel_write" in ask
        # Check hooks structure
        assert "hooks" in data
        assert "PreToolUse" in data["hooks"]
        assert "PostToolUse" in data["hooks"]

    def test_render_settings_json_hooks(self, template_manager):
        """Hook rules from controls server and framework hooks appear in rendered output."""
        ctx = self._full_ctx()
        # Supply framework hook rules (data-driven from selected_hooks)
        from osprey.cli.templates.claude_code import _build_framework_hook_rules

        fw_pre, fw_post = _build_framework_hook_rules(["memory-guard", "notebook-update"])
        ctx["framework_pre_hooks"] = fw_pre
        ctx["framework_post_hooks"] = fw_post

        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)
        pre_matchers = [r["matcher"] for r in data["hooks"]["PreToolUse"]]
        assert "Write" in pre_matchers  # Framework standalone hook (memory-guard)
        assert "mcp__controls__channel_write" in pre_matchers
        post_matchers = [r["matcher"] for r in data["hooks"]["PostToolUse"]]
        assert "NotebookEdit" in post_matchers  # Framework standalone hook (notebook-update)
        assert "mcp__controls__.*" in post_matchers

    def test_render_claude_md(self, template_manager):
        """Rendered CLAUDE.md includes enabled agents."""
        ctx = self._full_ctx(facility_name="Test Facility")
        rendered = self._render(template_manager, "claude_code/CLAUDE.md.j2", ctx)
        assert "logbook-search" in rendered
        assert "logbook-deep-research" in rendered
        # Logbook guidance sentence should appear
        assert "simple lookups" in rendered

    def test_render_hook_config_json(self, template_manager):
        """hook_config.json.j2 renders valid JSON with server/approval prefixes."""
        ctx = self._full_ctx()
        rendered = self._render(
            template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx
        )
        data = json.loads(rendered)
        assert "server_prefixes" in data
        assert "approval_prefixes" in data
        # Core servers should be in server_prefixes
        assert "mcp__controls__" in data["server_prefixes"]
        assert "mcp__osprey_workspace__" in data["server_prefixes"]
        # Controls, workspace, and ariel all have approval hooks
        assert "mcp__controls__" in data["approval_prefixes"]
        assert "mcp__osprey_workspace__" in data["approval_prefixes"]
        assert "mcp__ariel__" in data["approval_prefixes"]

    def test_render_hook_config_json_with_custom_server(self, template_manager):
        """Custom server with approval preset appears in both prefix lists."""
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc"],
                    "hooks": {"pre_tool_use": ["approval"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        ctx = self._full_ctx(_claude_code_config=cfg)
        rendered = self._render(
            template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx
        )
        data = json.loads(rendered)
        assert "mcp__my-plc__" in data["server_prefixes"]
        assert "mcp__my-plc__" in data["approval_prefixes"]

    _EXTENDS_CFG = {
        "servers": {
            "phoebus": {"enabled": True},
            "phoebus2": {
                "extends": "phoebus",
                "env": {"PHOEBUS_BRIDGE_URL": "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"},
            },
        }
    }

    def test_render_settings_json_with_extends(self, template_manager):
        """settings.json: extends clone yields mcp__phoebus2__ permissions and a
        drive-only PreToolUse approval hook (no wildcard, no stale template rule)."""
        ctx = self._full_ctx(_claude_code_config=self._EXTENDS_CFG)
        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)

        allow = set(data["permissions"]["allow"])
        for tool in _PHOEBUS_ALLOW:
            assert f"mcp__phoebus2__{tool}" in allow
        assert "mcp__phoebus2__phoebus_drive" not in allow
        assert set(data["permissions"]["ask"]) >= {
            "mcp__phoebus__phoebus_drive",
            "mcp__phoebus2__phoebus_drive",
        }

        pre = data["hooks"]["PreToolUse"]
        drive_rules = [r for r in pre if r["matcher"] == "mcp__phoebus2__phoebus_drive"]
        assert len(drive_rules) == 1
        assert any("osprey_approval.py" in h["command"] for h in drive_rules[0]["hooks"])
        # Drive-only gating: no wildcard pre rule for the clone.
        assert not any(r["matcher"] == "mcp__phoebus2__.*" for r in pre)
        # The clone contributes no duplicate template-prefixed rule.
        assert sum(1 for r in pre if r["matcher"] == "mcp__phoebus__phoebus_drive") == 1

    def test_render_mcp_json_with_extends(self, template_manager):
        """.mcp.json: clone block equals the old framework rendering — python
        -m osprey.mcp_server.phoebus with ${...} env preserved literally."""
        ctx = self._full_ctx(_claude_code_config=self._EXTENDS_CFG)
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        p2 = data["mcpServers"]["phoebus2"]
        assert p2["command"] == "/usr/bin/python3"
        assert p2["args"] == ["-m", "osprey.mcp_server.phoebus"]
        assert p2["env"]["PHOEBUS_BRIDGE_URL"] == "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"
        assert p2["env"]["OSPREY_CONFIG"] == "/tmp/test-project/config.yml"

    def test_render_hook_config_json_with_extends(self, template_manager):
        """hook_config.json: both phoebus prefixes land in server_prefixes and
        approval_prefixes (landmine D — templates are generic over s.name)."""
        ctx = self._full_ctx(_claude_code_config=self._EXTENDS_CFG)
        rendered = self._render(
            template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx
        )
        data = json.loads(rendered)
        for prefix in ("mcp__phoebus__", "mcp__phoebus2__"):
            assert prefix in data["server_prefixes"]
            assert prefix in data["approval_prefixes"]

    def test_render_settings_json_extends_remove_ask_interplay(self, template_manager):
        """facility remove_ask drops the clone's ask entry but the PreToolUse
        approval matcher survives (the hook is independent of the ask list)."""
        ctx = self._full_ctx(_claude_code_config=self._EXTENDS_CFG)
        ctx["facility_permissions"] = {"remove_ask": ["mcp__phoebus2__phoebus_drive"]}
        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)
        assert "mcp__phoebus2__phoebus_drive" not in data["permissions"]["ask"]
        assert "mcp__phoebus__phoebus_drive" in data["permissions"]["ask"]
        pre_matchers = [r["matcher"] for r in data["hooks"]["PreToolUse"]]
        assert "mcp__phoebus2__phoebus_drive" in pre_matchers

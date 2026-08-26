"""Render tests for ``.claude/hooks/hook_config.json``.

This file is the rendered contract between the build and the two runtime
consumers that gate control-system writes: the ``osprey_writes_check.py``
PreToolUse hook, and the MCP audit middleware. Both read it as data and
classify nothing themselves, so what the template emits *is* the safety
policy — and every key it emits must therefore hold under both writes states,
on both render paths, and when a consumer's context key is missing.

Three properties are pinned here:

1. **All keys are unconditional.** ``mixed_read_write_tools`` and
   ``write_servers`` describe properties of a *tool* and of a *server*, not of
   the current posture. They must not inherit the condition of the writes-off
   ``killswitch_deny`` derivation, whose block only runs when writes are
   disabled: the hook and the middleware read this file at run time under
   either posture.

2. **Both render paths agree.** ``TemplateManager.create_project`` renders
   this file with its own context and never calls
   ``build_claude_code_context``; ``regenerate_claude_code`` uses the latter.
   A key set on one path only ships a safety file whose write coverage
   depends on whether the project was built or rebuilt — with no error to say
   so, since the Jinja environment is not strict.

3. **Absence is empty, never missing.** A context key that is merely absent
   renders as nothing at all in this non-strict environment. Consumers
   distinguish "parsed, key absent" (degrade, warn) from "empty list"
   (nothing to cover), so the template must always emit the key.
"""

import json

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager

_HOOK_CONFIG_TEMPLATE = "claude_code/claude/hooks/hook_config.json.j2"

_PROJECT_COUNTER = 0

# The full key set of the rendered file. Consumers key off exactly these; a
# rename is a runtime contract break, not a cosmetic edit.
_EXPECTED_KEYS = {
    "server_prefixes",
    "approval_prefixes",
    "write_tools",
    "write_servers",
    "mixed_read_write_tools",
}


def _create_project(tmp_path):
    """Build a real project on disk through the create_project render path."""
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=f"hook-config-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    return manager, project_dir


def _hook_config(project_dir):
    return json.loads((project_dir / ".claude" / "hooks" / "hook_config.json").read_text())


def _reconfigure(project_dir, *, writes_enabled=True, claude_code_servers=None):
    """Rewrite the project's config.yml, the input both render paths read."""
    config_file = project_dir / "config.yml"
    config = yaml.safe_load(config_file.read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    if claude_code_servers is not None:
        config.setdefault("claude_code", {})["servers"] = claude_code_servers
    config_file.write_text(yaml.dump(config))
    return config


def _regenerate(manager, project_dir):
    """Re-render through build_claude_code_context, the second render path."""
    manager.regenerate_claude_code(project_dir)
    return _hook_config(project_dir)


# A representative multi-server render: a framework write server whose
# matchers are exact tool names (controls), a mixed read/write server
# (python), an `extends` clone of that mixed server (python2), a second
# framework write server (bluesky), and a facility-custom server whose
# writes_check matcher is the registry's per-server REGEX rather than an exact
# tool name (sitectl) — the case `write_servers` exists to cover.
_MULTI_SERVER = {
    "bluesky": {"enabled": True},
    "python2": {"extends": "python", "enabled": True},
    "sitectl": {
        "command": "node",
        "args": ["sitectl.js"],
        "permissions": {"ask": ["set_point"]},
        "hooks": {"pre_tool_use": ["writes_check", "approval"]},
    },
}


# ---------------------------------------------------------------------------
# Key presence, on both render paths
# ---------------------------------------------------------------------------


def test_create_project_path_emits_every_key(tmp_path):
    """The build path renders the full key set, populated."""
    _, project_dir = _create_project(tmp_path)
    config = _hook_config(project_dir)

    assert set(config) == _EXPECTED_KEYS
    assert "mcp__python__" in config["server_prefixes"]
    assert config["mixed_read_write_tools"] == ["mcp__python__execute"]
    assert config["write_servers"] == []


def test_regenerate_path_emits_every_key(tmp_path):
    """The regen path renders the same full key set, populated."""
    manager, project_dir = _create_project(tmp_path)
    config = _regenerate(manager, project_dir)

    assert set(config) == _EXPECTED_KEYS
    assert "mcp__python__" in config["server_prefixes"]
    assert config["mixed_read_write_tools"] == ["mcp__python__execute"]


def test_both_render_paths_render_the_same_file(tmp_path):
    """Byte-identical output from build and rebuild, for one config.

    The drift this pins is not hypothetical: create_project builds its own
    context and never runs build_claude_code_context, so a key added only to
    the latter renders EMPTY on the build path — a hook config whose mixed
    read/write exemption covers nothing, beside a fully populated write-tool
    list, in a file nothing validates. The non-emptiness assertions below keep
    the equality from passing vacuously on two empty renders.
    """
    manager, project_dir = _create_project(tmp_path)
    built = (project_dir / ".claude" / "hooks" / "hook_config.json").read_text()
    _regenerate(manager, project_dir)
    rebuilt = (project_dir / ".claude" / "hooks" / "hook_config.json").read_text()

    assert built == rebuilt
    parsed = json.loads(built)
    assert parsed["write_tools"], "no write tools rendered — equality is vacuous"
    assert parsed["mixed_read_write_tools"], "no mixed tools rendered — equality is vacuous"
    assert parsed["server_prefixes"], "no server prefixes rendered — equality is vacuous"


# ---------------------------------------------------------------------------
# Unconditional: independent of the writes kill switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writes_enabled", [True, False])
def test_keys_are_unconditional_in_both_writes_states(tmp_path, writes_enabled):
    """Writes off must not empty any of the three keys.

    With writes disabled the renderer runs its kill-switch block, which
    classifies mixed vs pure-write tools for ``permissions``. These keys are
    deliberately NOT derived there: the hook and the middleware read this file
    under a readonly posture too, where the kill switch never ran.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, writes_enabled=writes_enabled, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    assert config["mixed_read_write_tools"] == ["mcp__python__execute", "mcp__python2__execute"]
    assert config["write_servers"] == ["sitectl"]
    assert "mcp__sitectl__" in config["server_prefixes"]


# ---------------------------------------------------------------------------
# server_prefixes
# ---------------------------------------------------------------------------


def test_server_prefixes_names_every_enabled_server(tmp_path):
    """Every enabled server, framework / clone / custom alike — one entry each.

    The middleware fails CLOSED when its own prefix is absent from this list,
    so a missing entry is not a cosmetic gap: it clamps that server to the
    framework floor.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    prefixes = config["server_prefixes"]
    for expected in (
        "mcp__controls__",
        "mcp__python__",
        "mcp__python2__",
        "mcp__bluesky__",
        "mcp__sitectl__",
    ):
        assert expected in prefixes
    assert len(prefixes) == len(set(prefixes)), "duplicate server prefix"


def test_server_prefixes_excludes_disabled_servers(tmp_path):
    """A server the project disables renders no prefix — it launches nothing."""
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers={"python": {"enabled": False}})
    config = _regenerate(manager, project_dir)

    assert "mcp__python__" not in config["server_prefixes"]


# ---------------------------------------------------------------------------
# write_servers
# ---------------------------------------------------------------------------


def test_write_servers_lists_only_regex_matched_write_servers(tmp_path):
    """Server-level coverage exists for exactly the servers exact names miss.

    The registry gives a custom server that opts into ``writes_check`` a single
    per-server regex rule (``mcp__<name>__.*``): its individual tool names are
    unknown at render time, so the hook's exact-name membership test can never
    match one. Framework servers and their clones carry exact matchers and are
    already covered tool-by-tool — listing them here would take their READ
    tools down with them under a readonly posture.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    assert config["write_servers"] == ["sitectl"]
    for exact_matcher_server in ("controls", "python", "python2", "bluesky"):
        assert exact_matcher_server not in config["write_servers"]


def test_write_servers_holds_bare_names_not_prefixes(tmp_path):
    """Names, not ``mcp__<name>__`` prefixes.

    The consuming hook builds the prefix itself. Emitting a prefix here would
    make its ``mcp__%s__`` composition produce ``mcp__mcp__sitectl____``, which
    matches no tool — a coverage hole that reads as coverage.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    assert config["write_servers"]
    for name in config["write_servers"]:
        assert not name.startswith("mcp__")
        assert f"mcp__{name}__" in config["server_prefixes"]


def test_write_servers_empty_without_a_regex_gated_server(tmp_path):
    """A default project has none — an empty list, not a missing key."""
    _, project_dir = _create_project(tmp_path)
    config = _hook_config(project_dir)

    assert config["write_servers"] == []


def test_write_servers_skips_a_custom_server_without_writes_check(tmp_path):
    """The regex is only interesting where a writes_check hook rides it.

    A custom server that declares only ``approval`` gets the same
    ``mcp__<name>__.*`` matcher shape; keying off the regex alone rather than
    off the writes_check hook would refuse every one of its read tools under a
    readonly posture.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(
        project_dir,
        claude_code_servers={
            "readonlyctl": {
                "command": "node",
                "args": ["readonlyctl.js"],
                "permissions": {"ask": ["look"]},
                "hooks": {"pre_tool_use": ["approval"]},
            }
        },
    )
    config = _regenerate(manager, project_dir)

    assert "mcp__readonlyctl__" in config["server_prefixes"]
    assert config["write_servers"] == []


def test_write_servers_lists_a_server_once(tmp_path):
    """One entry per server, however many write-gated rules it carries."""
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    names = config["write_servers"]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# mixed_read_write_tools
# ---------------------------------------------------------------------------


def test_mixed_read_write_tools_covers_clones_and_excludes_pure_writes(tmp_path):
    """The render's own mixed list, clone names included.

    This is the middleware's clamp exemption: ``write_tools`` MINUS this list.
    A pure-write tool leaking in would be exempted from the clamp; a missing
    clone would clamp a legitimate read-only compute path.
    """
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    mixed = config["mixed_read_write_tools"]
    assert mixed == ["mcp__python__execute", "mcp__python2__execute"]
    for pure_write in ("mcp__controls__channel_write", "mcp__bluesky__queue_add"):
        assert pure_write not in mixed
    assert set(mixed).issubset(set(config["write_tools"]))


def test_mixed_read_write_tools_empty_when_no_mixed_server_enabled(tmp_path):
    """Empty list, not a missing key, when the project ships no mixed server."""
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers={"python": {"enabled": False}})
    config = _regenerate(manager, project_dir)

    assert config["mixed_read_write_tools"] == []
    assert "mixed_read_write_tools" in config


# ---------------------------------------------------------------------------
# write_tools stays what it was
# ---------------------------------------------------------------------------


def test_write_tools_still_carries_exact_names_and_the_custom_regex(tmp_path):
    """The pre-existing key is unchanged by the three additions."""
    manager, project_dir = _create_project(tmp_path)
    _reconfigure(project_dir, claude_code_servers=_MULTI_SERVER)
    config = _regenerate(manager, project_dir)

    write_tools = config["write_tools"]
    assert "mcp__controls__channel_write" in write_tools
    assert "mcp__python__execute" in write_tools
    assert "mcp__python2__execute" in write_tools
    # The regex the exact-name membership test can never match — the hole
    # `write_servers` closes at server level. It stays in write_tools: removing
    # it would strip the entry read_only_disallowed_tools is built from.
    assert "mcp__sitectl__.*" in write_tools


# ---------------------------------------------------------------------------
# Absence semantics of the rendering context
# ---------------------------------------------------------------------------


def _render_direct(manager, ctx):
    """Render the template with a hand-built context (no project on disk)."""
    return manager.jinja_env.get_template(_HOOK_CONFIG_TEMPLATE).render(**ctx)


def test_absent_mixed_key_renders_an_empty_list(tmp_path):
    """A context missing the key still emits it.

    Jinja is not strict here: an absent key renders as nothing at all, which
    would produce ``"mixed_read_write_tools": ,`` — a hook config that fails to
    parse, taking the whole file's coverage with it. The template defaults it.
    """
    manager = TemplateManager()
    rendered = _render_direct(
        manager,
        {
            "servers": [{"name": "python", "enabled": True, "hooks_pre": []}],
            "control_system_write_tools": [],
        },
    )

    config = json.loads(rendered)
    assert config["mixed_read_write_tools"] == []
    assert set(config) == _EXPECTED_KEYS


def test_absent_servers_key_renders_empty_lists_silently(tmp_path):
    """Documents the trap the other tests in this file exist to guard.

    Iterating an undefined name is not an error in this non-strict
    environment — it yields nothing — so a context without ``servers`` renders
    a structurally valid hook config that covers no tool and no server, with
    no error anywhere. Nothing downstream can tell that file from a project
    that legitimately enables nothing.

    Both render paths assign ``servers`` unconditionally, which is what makes
    the file trustworthy; that is a property of the callers, not of the
    template, and it is pinned by the populated-render tests above rather than
    by anything here.
    """
    manager = TemplateManager()
    config = json.loads(_render_direct(manager, {"control_system_write_tools": []}))

    assert set(config) == _EXPECTED_KEYS
    assert all(value == [] for value in config.values())


def test_rendered_file_is_valid_json_on_a_bare_render(tmp_path):
    """Every key present and typed, even with nothing enabled."""
    manager = TemplateManager()
    config = json.loads(_render_direct(manager, {"servers": [], "control_system_write_tools": []}))

    assert set(config) == _EXPECTED_KEYS
    for value in config.values():
        assert isinstance(value, list)


# ---------------------------------------------------------------------------
# The build-path context fix this task depends on
# ---------------------------------------------------------------------------


def test_build_path_context_carries_the_mixed_list(tmp_path):
    """create_project's own context computes the mixed list, like the regen path.

    Pinned at the context level as well as through the rendered file, because
    this is a one-line assignment in a path that reads nothing from
    ``build_claude_code_context`` — the kind of omission the rendered file
    reports as an empty list rather than as an error.
    """
    manager, project_dir = _create_project(tmp_path)
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    regen_ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )

    built = _hook_config(project_dir)
    assert built["mixed_read_write_tools"] == regen_ctx["mixed_read_write_tools"]
    assert built["mixed_read_write_tools"], "default render should carry python execute"

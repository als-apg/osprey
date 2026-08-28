"""Render-context tests for the Claude Code artifact context builder.

Focus here: the ``mixed_read_write_tools`` context key — the fully-qualified
list of read/write-mixed MCP tools, computed Python-side at render time where
both the ``extends`` clone names and the clone-rewritten ``_WRITES_CHECK``
matchers are known.

Why Python-side rather than in Jinja: the templates that consume it
(``hook_config.json.j2`` today, the audit middleware's config downstream) would
otherwise each have to re-derive *which* write-gated tools are mixed, and each
would have to re-apply the ``mcp__<template>__`` → ``mcp__<clone>__`` splice.
Two derivations of a safety classification is one too many. The template
classifies nothing; it renders a list.

The key is written unconditionally — an empty list when no mixed server is
enabled — because the Jinja environment is not strict: a key that is merely
absent renders as nothing at all, which is exactly how a safety file can come
out looking complete while covering no tool.
"""

import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import framework_mixed_read_write_tools, writes_check_matchers

_PROJECT_COUNTER = 0


def _build_ctx(tmp_path, *, writes_enabled: bool = True, claude_code_overrides: dict | None = None):
    """Create a real project on disk, apply config overrides, return the ctx.

    Each call gets a unique project name (TemplateManager refuses to create a
    project in an already-existing directory), so a test may call it twice.
    """
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=f"cc-templates-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    if claude_code_overrides is not None:
        config["claude_code"] = claude_code_overrides
    (project_dir / "config.yml").write_text(yaml.dump(config))

    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )


# ---------------------------------------------------------------------------
# mixed_read_write_tools context key
# ---------------------------------------------------------------------------


def test_mixed_read_write_tools_key_is_always_present(tmp_path):
    """Present as a list in BOTH writes states.

    The kill-switch block that used to own the mixed/pure-write distinction
    runs only when writes are off; this key must not inherit that condition,
    because its consumers (the hook config, and the audit middleware's clamp
    set) are read at run time under either posture.
    """
    for writes_enabled in (True, False):
        ctx = _build_ctx(tmp_path, writes_enabled=writes_enabled)
        assert isinstance(ctx["mixed_read_write_tools"], list)


def test_mixed_read_write_tools_names_both_python_exec_tools(tmp_path):
    """The documented read/write-mixed tools in a default control project.

    Both python-executor tools: they run the same arbitrary Python through the
    same kernels, so a readonly posture has to keep both reachable and let the
    writes-check hook decide per call.
    """
    ctx = _build_ctx(tmp_path)
    assert ctx["mixed_read_write_tools"] == [
        "mcp__python__execute",
        "mcp__python__execute_file",
    ]


def test_mixed_read_write_tools_matches_the_registry_floor(tmp_path):
    """A default render agrees with the registry's own framework floor, so the
    degraded-render fallback and the rendered list cannot drift apart."""
    ctx = _build_ctx(tmp_path)
    assert ctx["mixed_read_write_tools"] == framework_mixed_read_write_tools()


def test_mixed_read_write_tools_excludes_pure_write_tools(tmp_path):
    """Pure-write tools are write-gated too, and are the ones the kill switch
    hard-denies. Leaking one in here would exempt it from the clamp."""
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    tools = ctx["mixed_read_write_tools"]
    pure_write = writes_check_matchers("controls") + writes_check_matchers("bluesky")
    assert pure_write, "no pure-write matcher found — test is vacuous"
    for tool in pure_write:
        assert tool not in tools


def test_mixed_read_write_tools_empty_when_no_mixed_server_enabled(tmp_path):
    """Empty list, not a missing key: the template must render ``[]`` rather
    than nothing when the project ships no mixed server."""
    ctx = _build_ctx(tmp_path, claude_code_overrides={"servers": {"python": {"enabled": False}}})
    assert ctx["mixed_read_write_tools"] == []


def test_mixed_read_write_tools_covers_extends_clones(tmp_path):
    """A clone of a mixed template is mixed too, under its OWN tool names.

    This is what the Python-side precompute buys: the clone's matchers exist
    only after the registry has rewritten them, so nothing upstream of
    resolve_servers can name them.
    """
    ctx = _build_ctx(
        tmp_path,
        claude_code_overrides={"servers": {"python2": {"extends": "python", "enabled": True}}},
    )
    assert "mcp__python__execute" in ctx["mixed_read_write_tools"]
    assert "mcp__python2__execute" in ctx["mixed_read_write_tools"]


def test_killswitch_still_pulls_mixed_tools_from_ask(tmp_path):
    """The behavior the moved constant used to drive stays put: with writes
    off, a mixed tool is pulled OUT of ask (its readonly path is legitimate)
    rather than hard-denied like a pure-write tool."""
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    assert "mcp__python__execute" in ctx["facility_permissions"]["remove_ask"]
    assert "mcp__python__execute" not in ctx["killswitch_deny"]

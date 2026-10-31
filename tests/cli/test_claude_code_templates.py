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

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import framework_mixed_read_write_tools, writes_check_matchers
from tests._preset_data import create_project_as_built

_PROJECT_COUNTER = 0


def _build_ctx(tmp_path, *, writes_enabled: bool = True, claude_code_overrides: dict | None = None):
    """Create a real project on disk, apply config overrides, return the ctx.

    Each call gets a unique project name (TemplateManager refuses to create a
    project in an already-existing directory), so a test may call it twice.
    """
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
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


# ---------------------------------------------------------------------------
# Agent-facing words for the three-position switch (D7, template half)
# ---------------------------------------------------------------------------

_SAFETY_RULE = "claude_code/claude/rules/control-system-safety.md.j2"
_SETUP_SKILL = "claude_code/claude/skills/setup-mode/SKILL.md.j2"
_DEMO_SKILL = "claude_code/claude/skills/demo-ui/SKILL.md"

#: The approval sentence every tier's rule carries, whatever its ceiling.
_APPROVAL_SENTENCE = (
    "a person approves each write while the machine is at asked; at allowed the "
    "login's switch position approves it, and limits still apply"
)

#: The one sentence each tier's render says about its own ceiling.
_TIER_SENTENCES = {
    "never": "This agent's tier is `never`: no switch position arms a write here, "
    "so every write is refused.",
    "asked": "This agent's tier is `asked`: the switch goes no further than asked, "
    "so a person approves every write.",
    "allowed": "This agent's tier is `allowed`: the login may set a machine to allowed, "
    "and then its writes run without a prompt.",
}


def _render_at_tier(tmp_path, tier: str, template: str, *, bluesky: bool = False) -> str:
    """Render one template from a real project whose tier key is ``tier``."""
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        project_name=f"cc-tier-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"].pop("writes_enabled", None)
    config["control_system"]["writes"] = tier
    if bluesky:
        config["claude_code"] = {"servers": {"bluesky": {"enabled": True}}}
    (project_dir / "config.yml").write_text(yaml.dump(config))
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    assert ctx["writes_tier"] == tier, "the render did not settle on the tier under test"
    return manager.jinja_env.get_template(template).render(**ctx)


def _flat(text: str) -> str:
    """Collapse line wrapping, so a pinned sentence matches however it is wrapped."""
    return " ".join(text.split())


@pytest.mark.parametrize("tier", sorted(_TIER_SENTENCES))
def test_safety_rule_names_who_approves_each_write(tmp_path, tier):
    """Human approval is the switch position's, not a per-tool policy's: a person at
    asked, the login's position at allowed, and the limits check either way."""
    rule = _flat(_render_at_tier(tmp_path, tier, _SAFETY_RULE))
    assert _APPROVAL_SENTENCE in rule
    assert "routes writes through the approval workflow" not in rule
    for position in ("**off**", "**asked**", "**allowed**"):
        assert position in rule


@pytest.mark.parametrize("tier", sorted(_TIER_SENTENCES))
def test_safety_rule_states_its_own_tier_and_no_other(tmp_path, tier):
    """Each tier's render says what its ceiling means, and only its own."""
    rule = _flat(_render_at_tier(tmp_path, tier, _SAFETY_RULE))
    assert _TIER_SENTENCES[tier] in rule
    for other, sentence in _TIER_SENTENCES.items():
        if other != tier:
            assert sentence not in rule, other


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_safety_rule_drops_one_approval_is_the_whole_gate(tmp_path, tier):
    """With a queue configured, ``queue_start`` follows the switch: the retired
    claim that one approval is the whole gate is gone, and both positions are named."""
    rule = _flat(_render_at_tier(tmp_path, tier, _SAFETY_RULE, bluesky=True))
    assert "## Starting a Bluesky Queue" in rule
    assert "one approval is the whole gate" not in rule
    assert "`queue_start` follows the switch position of the lane's machine" in rule
    assert "at allowed the login's switch position approves it" in rule


@pytest.mark.parametrize("tier", sorted(_TIER_SENTENCES))
def test_setup_mode_names_three_tier_values_and_three_positions(tmp_path, tier):
    skill = _flat(_render_at_tier(tmp_path, tier, _SETUP_SKILL))
    assert "The tier's write ceiling, `never`, `asked` or `allowed`" in skill
    assert "the furthest switch position (off, asked or allowed) a login may pick" in skill
    assert (
        "at off the write is refused, at asked a person approves it first, "
        "at allowed it runs without a prompt" in skill
    )
    assert "`allowed` or `never`" not in skill


def test_demo_ui_names_three_tier_values_and_three_positions():
    """``demo-ui`` is copied verbatim, so its text is pinned straight from the source."""
    skill = _flat((TemplateManager().template_root / _DEMO_SKILL).read_text(encoding="utf-8"))
    assert "The tier sets the ceiling — `never`, `asked` or `allowed`" in skill
    assert (
        "**off** refuses the write, **asked** puts the approval prompt in front of a "
        "person, and **allowed** runs the write without one" in skill
    )

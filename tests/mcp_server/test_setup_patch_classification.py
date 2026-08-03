"""`setup_patch` change classification must not oversell `writes_enabled`.

`control_system.writes_enabled` is enforced at three layers: the PreToolUse hook
(fresh subprocess per call, genuinely re-reads config), the connector (reads the
value through a process-cached config), and the rendered ``permissions.deny``
list in ``settings.json`` (regenerated only by ``osprey claude regen``). Calling
the key "hot" told the operator a patch alone would un-gate writes, when in fact
writes stay blocked until a regen and restart.
"""

from pathlib import Path

import pytest

from osprey.mcp_server.workspace.tools.setup import (
    _HOT_CHANGE_PATHS,
    _classify_change,
)

WRITES_ENABLED = "control_system.writes_enabled"

SKILL_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/claude_code/claude/skills/setup-mode/SKILL.md.j2"
)


def test_writes_enabled_is_not_a_hot_path():
    """The key must not be listed as hot — the connector caches it at launch."""
    assert WRITES_ENABLED not in _HOT_CHANGE_PATHS.get("config.yml", set())


def test_writes_enabled_classified_cold():
    note = _classify_change("config.yml", WRITES_ENABLED)
    assert note.startswith("cold")
    assert "hot" not in note.lower()


def test_writes_enabled_note_names_every_layer_and_the_remedy():
    """The note must explain why a patch alone is not enough."""
    note = _classify_change("config.yml", WRITES_ENABLED)
    assert "hook" in note.lower()
    assert "connector" in note.lower()
    assert "permissions.deny" in note
    assert "osprey claude regen" in note
    assert "restart" in note.lower()


@pytest.mark.parametrize(
    "key_path",
    [
        "control_system.limits_checking.enabled",
        "control_system.limits_checking.allow_unlisted_channels",
    ],
)
def test_hook_only_limits_keys_stay_hot(key_path):
    """The limits hook really is a per-call subprocess — don't over-correct."""
    assert key_path in _HOT_CHANGE_PATHS["config.yml"]
    assert _classify_change("config.yml", key_path).startswith("hot")


def test_unlisted_key_keeps_the_generic_cold_note():
    note = _classify_change("config.yml", "control_system.type")
    assert note.startswith("cold")
    assert "requires MCP server restart" in note


def test_mcp_json_patches_are_cold():
    note = _classify_change(".mcp.json", WRITES_ENABLED)
    assert note.startswith("cold")


def test_skill_table_no_longer_calls_writes_enabled_hot():
    """The rendered setup-mode skill tabulated the key as Hot."""
    text = SKILL_TEMPLATE.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith(f"| `{WRITES_ENABLED}` |")]
    assert len(rows) == 1, f"expected one hot/cold table row, got {rows}"
    assert "| Cold |" in rows[0]
    assert "| Hot |" not in rows[0]


def test_skill_writes_blocked_remedy_requires_regen_and_restart():
    """The 'Writes Blocked' fix used to prescribe a patch and call it hot."""
    text = SKILL_TEMPLATE.read_text(encoding="utf-8")
    fix_lines = [
        ln for ln in text.splitlines() if ln.startswith("**Fix**:") and WRITES_ENABLED in ln
    ]
    assert len(fix_lines) == 1, f"expected one writes_enabled fix line, got {fix_lines}"
    fix = fix_lines[0]
    assert "osprey claude regen" in fix
    assert "restart" in fix.lower()
    assert "(hot change)" not in fix

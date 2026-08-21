"""Pin the verification guidance in ``safety.md`` item 6.

Item 6 used to prescribe a level ("Default verification_level is
'callback.'"). The level is actually resolved per write by the deployment --
channel limits entry, then the limits database ``defaults.verification``, then
the connector config, then ``callback`` -- so a prompt-layer default is wrong
wherever a project configures readback. The rule now *describes* that
resolution and states the consequence the agent must respect: a
``callback``-level result carries no readback value and no alarm state, so the
agent must report ``write_state`` rather than the resulting machine state.

These tests exercise the real delivery path: ``safety.md`` is a non-Jinja
template copied into ``.claude/rules/safety.md`` by the Claude Code
integration (catalog entry ``rules/safety``), so the assertions run against a
scaffolded project rather than the template file.
"""

import pytest

from osprey.cli.templates.manager import TemplateManager


@pytest.fixture(scope="module")
def rendered_safety_rule(tmp_path_factory) -> str:
    """Scaffold a project and return its rendered ``.claude/rules/safety.md``."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="safety-verification-guidance",
        output_dir=tmp_path_factory.mktemp("safety-md"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    rule = project_dir / ".claude" / "rules" / "safety.md"
    assert rule.is_file(), "safety.md did not reach the generated project"
    return rule.read_text(encoding="utf-8")


def test_prescribed_default_level_is_gone(rendered_safety_rule: str) -> None:
    """The prompt layer must not name a default verification level."""
    assert "Default verification_level is" not in rendered_safety_rule


def test_rule_describes_the_resolution_order(rendered_safety_rule: str) -> None:
    """Item 6 names the four resolution steps, in order."""
    start = rendered_safety_rule.find("\n6. **")
    assert start != -1, f"item 6 headline not found in: {rendered_safety_rule}"
    end = rendered_safety_rule.find("## Data Integrity")
    assert end != -1, f"'## Data Integrity' section not found in: {rendered_safety_rule}"
    item6 = rendered_safety_rule[start:end]

    positions = [
        item6.index("limits entry"),
        item6.index("`defaults.verification`"),
        item6.index("connector config"),
        item6.index("`callback`"),
    ]
    assert positions == sorted(positions), f"resolution order out of sequence: {item6}"


def test_rule_states_callback_carries_no_readback_or_alarm(rendered_safety_rule: str) -> None:
    """A callback-level result has readback_value and alarm fields as None."""
    prose = " ".join(rendered_safety_rule.split())
    assert "no readback value and no alarm state" in prose
    assert "do NOT describe the machine state from it" in prose


def test_rule_points_the_agent_at_write_state(rendered_safety_rule: str) -> None:
    """The agent reports what the tool result says, not what it infers."""
    assert "`write_state`" in rendered_safety_rule


def test_rule_routes_legitimate_readback_to_rule_four(rendered_safety_rule: str) -> None:
    """The prohibition is scoped to the write result -- rule 4 still covers explicit readback."""
    prose = " ".join(rendered_safety_rule.split())
    assert "read the channel back (rule 4)" in prose

    start = rendered_safety_rule.find("\n4. **")
    assert start != -1, f"item 4 headline not found in: {rendered_safety_rule}"
    end = rendered_safety_rule.find("\n5. **")
    assert end != -1, f"item 5 headline not found in: {rendered_safety_rule}"
    item4 = rendered_safety_rule[start:end]
    assert "Read back channels after writing" in item4

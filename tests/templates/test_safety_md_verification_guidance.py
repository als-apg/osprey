"""Pin the verification guidance in ``safety.md`` item 6.

Item 6 used to prescribe a level ("Default verification_level is
'callback.'"). The level is actually resolved per write by the deployment --
channel limits entry, then the limits database ``defaults.verification``, then
the connector config, then ``callback`` -- so a prompt-layer default is wrong
wherever a project configures readback. The rule now *describes* that
resolution and tells the agent what to report: ``write_state``, plus the
``readback_value`` and alarm state the result carries. Every verifying level
carries the post-write readback, so the rule no longer declares it absent;
when a result has none, rule 4 (read back after writing) covers the gap, and
the two rules say the same thing.

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


def test_rule_no_longer_declares_the_callback_result_value_less(rendered_safety_rule: str) -> None:
    """The prompt patch is retired: callback-level results carry the readback now."""
    prose = " ".join(rendered_safety_rule.split())
    assert "no readback value" not in prose
    assert "do NOT describe the machine state" not in prose


def test_rule_points_the_agent_at_write_state_and_readback_value(
    rendered_safety_rule: str,
) -> None:
    """The agent reports what the tool result says -- state and carried value."""
    assert "`write_state`" in rendered_safety_rule
    assert "`readback_value`" in rendered_safety_rule


def test_rule_routes_a_value_less_result_to_rule_four(rendered_safety_rule: str) -> None:
    """Rule 6 and rule 4 agree: a result with no readback means read back after writing."""
    prose = " ".join(rendered_safety_rule.split())
    assert "If it carries none" in prose
    assert "read the channel back (rule 4)" in prose

    start = rendered_safety_rule.find("\n4. **")
    assert start != -1, f"item 4 headline not found in: {rendered_safety_rule}"
    end = rendered_safety_rule.find("\n5. **")
    assert end != -1, f"item 5 headline not found in: {rendered_safety_rule}"
    item4 = rendered_safety_rule[start:end]
    assert "Read back channels after writing" in item4

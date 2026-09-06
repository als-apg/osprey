"""Pin how ``safety.md`` item 6 teaches the agent to read a write result.

Item 6 used to prescribe a verification level, and then to describe how the
deployment resolved one. Both are gone. A write now carries a single ``outcome``
word, set by the connector and read by everyone, so the rule has exactly one job
left: make the agent repeat that word and stop there. An agent that narrates a
write the machine never checked as a confirmed one tells an operator the machine
is in a state nobody established.

The other half of what the agent must pass on is the evidence behind the word --
the observed value and the alarm state the result carries -- and, when the result
carries no observed value, item 4 (read back after writing) covers the gap.

One test here is the rules half of a parity pin: the key the ``channel_write``
tool emits and the key this prose tells the agent to read are one string. Its
other half lives beside the tool, in
``tests/mcp_server/test_channel_write_tool.py``.

These tests exercise the real delivery path: ``safety.md`` is a non-Jinja
template copied into ``.claude/rules/safety.md`` by the Claude Code
integration (catalog entry ``rules/safety``), so the assertions run against a
scaffolded project rather than the template file.
"""

from pathlib import Path

import pytest

from osprey.cli.templates.manager import TemplateManager


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


#: Every outcome word a write can carry. The rule lists them so the agent knows
#: the vocabulary is closed -- anything else it is tempted to say is invented.
OUTCOME_WORDS = ("confirmed", "mismatch", "unconfirmed", "unrequested", "refused", "failed")


@pytest.fixture(scope="module")
def rendered_safety_rule(tmp_path_factory) -> str:
    """Scaffold a project and return its rendered ``.claude/rules/safety.md``."""
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name="safety-verification-guidance",
        output_dir=tmp_path_factory.mktemp("safety-md"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    rule = project_dir / ".claude" / "rules" / "safety.md"
    assert rule.is_file(), "safety.md did not reach the generated project"
    return rule.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def item_six(rendered_safety_rule: str) -> str:
    """Item 6 of the channel-write list, as one whitespace-normalised string."""
    start = rendered_safety_rule.find("\n6. **")
    assert start != -1, f"item 6 headline not found in: {rendered_safety_rule}"
    end = rendered_safety_rule.find("## Data Integrity")
    assert end != -1, f"'## Data Integrity' section not found in: {rendered_safety_rule}"
    return " ".join(rendered_safety_rule[start:end].split())


def test_retired_write_vocabulary_is_gone(item_six: str) -> None:
    """No level to override, no resolution chain, no state key to read."""
    for retired in ("verification", "level", "readback", "write_state", "callback"):
        assert retired not in item_six.lower(), f"item 6 still names {retired!r}: {item_six}"


def test_rule_names_the_outcome_key_the_tool_emits(item_six: str) -> None:
    """Rules half of the parity pin: one word names the key on both sides.

    The tool half asserts the payload carries the key its constant names; this
    half asserts the prose sends the agent to that same key. Renaming either
    side alone tells the agent to read something the payload does not contain --
    silently, because the tool still returns a valid result.
    """
    from osprey.mcp_server.control_system.tools.channel_write import OUTCOME_KEY

    assert f"`{OUTCOME_KEY}`" in item_six, f"item 6 does not name the emitted key: {item_six}"


def test_rule_lists_every_outcome_word(item_six: str) -> None:
    """The vocabulary is closed, so the rule spells all six words out."""
    for word in OUTCOME_WORDS:
        assert f"`{word}`" in item_six, f"item 6 is missing outcome word {word!r}: {item_six}"


def test_rule_forbids_reporting_anything_stronger(item_six: str) -> None:
    """The one failure mode this rule exists to prevent, named in the headline."""
    assert "and nothing stronger" in item_six
    assert "Never upgrade it" in item_six
    assert "it is not a confirmed write" in item_six


def test_rule_asks_for_the_observed_value_and_alarm_state(item_six: str) -> None:
    """The word alone is not a report -- the evidence travels with it."""
    assert "`observed_value`" in item_six
    assert "alarm state" in item_six


def test_rule_routes_a_value_less_result_to_rule_four(
    rendered_safety_rule: str, item_six: str
) -> None:
    """Item 6 and item 4 agree: no observed value means read back after writing."""
    assert "If it carries no observed value" in item_six
    assert "read the channel back (rule 4)" in item_six

    start = rendered_safety_rule.find("\n4. **")
    assert start != -1, f"item 4 headline not found in: {rendered_safety_rule}"
    end = rendered_safety_rule.find("\n5. **")
    assert end != -1, f"item 5 headline not found in: {rendered_safety_rule}"
    item4 = rendered_safety_rule[start:end]
    assert "Read back channels after writing" in item4


def test_rule_leaves_the_confirm_decision_to_the_deployment(item_six: str) -> None:
    """What survives of "do not override the level": the agent does not decide
    whether a write is checked."""
    assert "the deployment's decision, not yours" in item_six

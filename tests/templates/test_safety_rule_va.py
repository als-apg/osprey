"""Render tests for the EPICS prohibitions of the agent's safety rules.

Every deployment whose connector speaks Channel Access -- an EPICS machine, the
served virtual accelerator, the live stand-in -- gets the EPICS prohibitions of
``control-system-safety.md.j2`` and the test-IOC rule ``test-ioc-safety.md.j2``,
because the same ``caget``/``caput`` bypass reaches each of them. The simulator
in process speaks no wire and gets neither.
"""

import pytest
import yaml
from jinja2 import TemplateRuntimeError
from tests._control_system import IN_PROCESS, section_for
from tests._preset_data import create_project_as_built

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey_connectors.types import connector_transport

#: The section words whose connector speaks Channel Access.
CA_TYPES = ("epics", "virtual_accelerator", "live_standin")

#: A connector type no transport row names: a facility's own dotted connector.
CUSTOM_TYPE = "my_facility.connectors.Custom"


def _render_safety_rule(tmp_path, project_name: str, control_system_type: str | None) -> str:
    """Scaffold a project, set control_system.type, render Claude Code
    integration files, and return the rendered safety-rule content."""
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text())
    if control_system_type is not None:
        section = config.setdefault("control_system", {})
        named = section_for(control_system_type)
        section["type"] = named["type"]
        for key, block in named.get("connector", {}).items():
            section.setdefault("connector", {}).setdefault(key, {}).update(block)
        (project_dir / "config.yml").write_text(yaml.dump(config))

    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    claude_code.create_claude_code_integration(
        manager.template_root, manager.jinja_env, project_dir, ctx
    )

    return (project_dir / ".claude" / "rules" / "control-system-safety.md").read_text()


def _assert_epics_prohibitions_present(content: str) -> None:
    assert "import aioca" in content
    assert "aioca.caget" in content
    assert "aioca.caput" in content
    assert "epics.caput" in content
    assert "Bypasses audit logging" in content
    assert (
        'await aioca.caput("SR:MAG:QF:01:CURRENT:SP", 150)  # Refused at runtime: RAW_CLIENT_WRITE'
        in content
    )


def test_epics_prohibitions_present(tmp_path):
    content = _render_safety_rule(tmp_path, "safety-epics", "epics")
    _assert_epics_prohibitions_present(content)


def test_virtual_accelerator_prohibitions_present(tmp_path):
    """The bug this task fixes: virtual_accelerator must get the same
    caget/caput prohibitions as epics, not the generic else-branch text."""
    content = _render_safety_rule(tmp_path, "safety-va", "virtual_accelerator")
    _assert_epics_prohibitions_present(content)


def test_the_simulator_in_process_renders_no_epics_family_lines(tmp_path):
    """The simulator in process speaks no Channel Access: no EPICS-family lines."""
    content = _render_safety_rule(tmp_path, "safety-in-process", IN_PROCESS)

    assert "osprey.runtime" in content
    assert "import aioca" not in content
    assert "import epics" not in content
    assert "aioca.caput" not in content
    assert "epics.caget" not in content
    assert "epics.caput" not in content
    assert "EPICS Channel Access" not in content


def _rendered_rules_dir(tmp_path, project_name: str, control_system_type: str | None):
    _render_safety_rule(tmp_path, project_name, control_system_type)
    return tmp_path / project_name / ".claude" / "rules"


@pytest.mark.parametrize("cs_type", CA_TYPES)
def test_test_ioc_rule_renders_for_channel_access(tmp_path, cs_type):
    """test-ioc-safety.md materializes wherever the connector speaks Channel Access."""
    rules_dir = _rendered_rules_dir(tmp_path, f"ioc-{cs_type[:8]}", cs_type)
    rule = rules_dir / "test-ioc-safety.md"
    assert rule.exists(), f"test-ioc-safety.md missing for {cs_type}"
    assert rule.read_text().strip()


@pytest.mark.parametrize("cs_type", [IN_PROCESS, "doocs", "tango"])
def test_test_ioc_rule_absent_off_channel_access(tmp_path, cs_type):
    """The template's Channel Access gate renders empty under any other
    connector, and the empty-file cleanup removes it rather than leave a blank
    rule."""
    rules_dir = _rendered_rules_dir(tmp_path, f"ioc-{cs_type[:8]}", cs_type)
    assert not (rules_dir / "test-ioc-safety.md").exists()


def _render_test_ioc_rule(**context) -> str:
    template = TemplateManager().jinja_env.get_template(
        "claude_code/claude/rules/test-ioc-safety.md.j2"
    )
    return template.render(**context)


@pytest.mark.parametrize("cs_type", [*CA_TYPES, IN_PROCESS, "doocs", "tango", CUSTOM_TYPE])
def test_test_ioc_rule_renders_wherever_channel_access_is_spoken(cs_type):
    """The rule is decided by the wire alone: present on every Channel Access
    connector, empty on every other one."""
    content = _render_test_ioc_rule(
        control_system_transport=connector_transport(section_for(cs_type))
    )

    if cs_type in CA_TYPES:
        assert "# Test IOC Safety" in content, f"{cs_type}: test-IOC rule missing"
    else:
        assert not content.strip(), f"{cs_type}: test-IOC rule rendered off Channel Access"


def test_test_ioc_rule_without_the_wire_stops():
    """A context that does not say which wire the connector speaks stops the
    render rather than dropping the rule."""
    with pytest.raises(TemplateRuntimeError, match="control_system_transport"):
        _render_test_ioc_rule()


def test_control_assistant_preset_selects_test_ioc_rule():
    """The DOA regression this pins: the rule existed in the artifact catalog
    but no preset selected it, so it rendered into no built project. The
    preset's rules list must resolve the rule into the allowed output set."""
    from importlib.resources import files

    from osprey.cli.templates.manifest import resolve_manifest_outputs

    preset = yaml.safe_load(
        files("osprey").joinpath("profiles/presets/control-assistant.yml").read_text()
    )
    assert "test-ioc-safety" in preset["rules"]
    outputs = resolve_manifest_outputs({"artifacts": {"rules": preset["rules"]}})
    assert ".claude/rules/test-ioc-safety.md" in outputs

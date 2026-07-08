"""Render tests for control-system-safety.md.j2 covering the
``virtual_accelerator`` control system type.

The Virtual Accelerator connector (``VirtualAcceleratorConnector``) is an
unmodified ``EPICSConnector`` subclass -- it talks real Channel Access to a
containerized PyAT soft-IOC, so the same ``caget``/``caput`` bypass hazard
that applies to ``epics`` applies identically to ``virtual_accelerator``.
Before this fix, ``virtual_accelerator`` fell into the generic ``else``
branch and lost those prohibitions exactly when a project first speaks real
CA. These tests assert the prohibitions render for both ``epics`` and
``virtual_accelerator``, and that ``mock`` is unaffected.
"""

import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager


def _render_safety_rule(tmp_path, project_name: str, control_system_type: str | None) -> str:
    """Scaffold a project, set control_system.type, render Claude Code
    integration files, and return the rendered safety-rule content."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text())
    if control_system_type is not None:
        config.setdefault("control_system", {})["type"] = control_system_type
        (project_dir / "config.yml").write_text(yaml.dump(config))

    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    claude_code.create_claude_code_integration(
        manager.template_root, manager.jinja_env, project_dir, ctx
    )

    return (project_dir / ".claude" / "rules" / "control-system-safety.md").read_text()


def _assert_epics_prohibitions_present(content: str) -> None:
    assert "import epics" in content
    assert "epics.caget" in content
    assert "epics.caput" in content
    assert "Bypasses audit logging" in content
    assert "Bypasses limits + approval" in content
    assert "Bypasses all safety layers" in content


def test_epics_prohibitions_present(tmp_path):
    content = _render_safety_rule(tmp_path, "safety-epics", "epics")
    _assert_epics_prohibitions_present(content)


def test_virtual_accelerator_prohibitions_present(tmp_path):
    """The bug this task fixes: virtual_accelerator must get the same
    caget/caput prohibitions as epics, not the generic else-branch text."""
    content = _render_safety_rule(tmp_path, "safety-va", "virtual_accelerator")
    _assert_epics_prohibitions_present(content)


def test_epics_and_virtual_accelerator_prohibited_sections_match(tmp_path):
    """Same underlying protocol (real Channel Access) -> identical code
    example, not a VA-specific rewrite."""
    epics_content = _render_safety_rule(tmp_path / "epics", "safety-epics", "epics")
    va_content = _render_safety_rule(tmp_path / "va", "safety-va", "virtual_accelerator")

    def _prohibited_section(content: str) -> str:
        start = content.index("### Prohibited")
        end = content.index("### Why This Matters")
        return content[start:end]

    assert _prohibited_section(epics_content) == _prohibited_section(va_content)


def test_mock_keeps_current_generic_text(tmp_path):
    """mock (the default) must be unaffected by this change."""
    content = _render_safety_rule(tmp_path, "safety-mock", None)

    assert "Control System" in content
    assert "direct hardware library calls" in content
    assert "osprey.runtime" in content
    assert "import epics" not in content
    assert "epics.caget" not in content
    assert "epics.caput" not in content
    assert "EPICS Channel Access" not in content

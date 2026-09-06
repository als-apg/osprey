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

from pathlib import Path

import yaml

from osprey.cli.templates import claude_code
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


def _render_safety_rule(tmp_path, project_name: str, control_system_type: str | None) -> str:
    """Scaffold a project, set control_system.type, render Claude Code
    integration files, and return the rendered safety-rule content."""
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
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


def _rendered_rules_dir(tmp_path, project_name: str, control_system_type: str | None):
    _render_safety_rule(tmp_path, project_name, control_system_type)
    return tmp_path / project_name / ".claude" / "rules"


def test_test_ioc_rule_renders_for_epics_family(tmp_path):
    """test-ioc-safety.md must materialize for both EPICS-family types."""
    for cs_type in ("epics", "virtual_accelerator"):
        rules_dir = _rendered_rules_dir(tmp_path / cs_type, f"ioc-{cs_type[:8]}", cs_type)
        rule = rules_dir / "test-ioc-safety.md"
        assert rule.exists(), f"test-ioc-safety.md missing for {cs_type}"
        assert rule.read_text().strip()


def test_test_ioc_rule_absent_for_mock(tmp_path):
    """The template's EPICS-family gate renders empty for mock, and the
    empty-file cleanup must remove it rather than leave a blank rule."""
    rules_dir = _rendered_rules_dir(tmp_path, "ioc-mock", None)
    assert not (rules_dir / "test-ioc-safety.md").exists()


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

"""Tests for the demo-ui skill and its 4-point framework wiring.

Mirrors ``test_operating_bluesky_plans_skill_install.py``'s registry/template/
preset/manifest pattern, plus a real ``TemplateManager.create_project`` build so
a missed registration cannot leave the SKILL.md in the template tree but absent
from a built project.

The content assertions pin the demo's run surface: the workspace panel tools it
choreographs must be the ones the workspace MCP server actually exposes, and the
two disciplines that make a demo repeatable in front of an audience -- reading
the live panel inventory instead of guessing IDs, and restoring the starting
layout -- must survive future edits.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import manifest
from osprey.cli.templates.manager import TemplateManager
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog


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


TEMPLATE_ROOT = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"
PRESETS_DIR = Path(__file__).parent.parent.parent / "src" / "osprey" / "profiles" / "presets"

SKILL_REL = "claude/skills/demo-ui/SKILL.md"
OUTPUT_REL = ".claude/skills/demo-ui/SKILL.md"
SKILL_PATH = TEMPLATE_ROOT / "claude" / "skills" / "demo-ui" / "SKILL.md"


class TestDemoUiRegistry:
    """Wiring point 1: BuildArtifactCatalog registration."""

    @pytest.fixture()
    def registry(self):
        return BuildArtifactCatalog.default()

    def test_registered(self, registry):
        art = registry.get("skills/demo-ui")
        assert art is not None
        assert art.output_path == OUTPUT_REL
        assert art.template_path == SKILL_REL


class TestDemoUiTemplateExists:
    """Wiring point 2: the skill bundle template file itself."""

    def test_skill_file_exists(self):
        assert SKILL_PATH.exists(), f"SKILL.md not found at {SKILL_PATH}"


class TestDemoUiPresetWiring:
    """Wiring point 3: the preset's ``skills:`` directive."""

    def test_control_assistant_lists_the_skill(self):
        profile = yaml.safe_load(
            (PRESETS_DIR / "control-assistant.yml").read_text(encoding="utf-8")
        )
        assert "demo-ui" in profile["skills"]


class TestDemoUiManifestWiring:
    """Wiring point 4: the regen-tracked-files fallback list."""

    def test_in_regen_tracked_files(self):
        assert OUTPUT_REL in manifest.REGEN_TRACKED_FILES


class TestDemoUiSkillStructure:
    """Content assertions: the tool surface and the demo disciplines."""

    @pytest.fixture()
    def skill_text(self):
        return SKILL_PATH.read_text(encoding="utf-8")

    def test_has_frontmatter(self, skill_text):
        assert skill_text.startswith("---")
        assert "name: demo-ui" in skill_text

    def test_choreographs_the_panel_tools(self, skill_text):
        """These are the workspace MCP tools the demo drives -- a rename in
        ``workspace/tools/panel_tools.py`` must fail here, not on stage."""
        for tool in ("list_panels", "open_panel", "add_panel_to_rail", "remove_panel_from_rail"):
            assert tool in skill_text, f"Missing panel tool: {tool}"

    def test_choreographs_the_artifact_tools(self, skill_text):
        for tool in (
            "create_interactive_plot",
            "artifact_register",
            "artifact_focus",
            "artifact_pin",
        ):
            assert tool in skill_text, f"Missing artifact tool: {tool}"

    def test_offers_the_workflow_menu(self, skill_text):
        """The skill's whole point is not having to spell out each demo."""
        lowered = skill_text.lower()
        for workflow in ("panel tour", "artifact drop", "layout switch", "grand tour"):
            assert workflow in lowered, f"Missing workflow: {workflow}"

    def test_reads_the_live_panel_inventory(self, skill_text):
        """Panel IDs vary per deployment; a hardcoded tab list breaks the demo
        on any deployment that does not enable that panel."""
        assert "list_panels" in skill_text
        assert "never guess" in skill_text.lower()

    def test_restores_the_starting_layout(self, skill_text):
        """Demos run twice, and often on someone's real working session."""
        lowered = skill_text.lower()
        assert "restore" in lowered
        assert "as you found it" in lowered

    def test_gates_the_write_beat_on_explicit_request(self, skill_text):
        """A live write is a real machine action -- it must never be an
        unprompted beat, and the approval prompt must never be bypassed."""
        lowered = skill_text.lower()
        assert "approval" in lowered
        assert "only when the operator explicitly asks" in lowered
        assert "never pre-approve" in lowered

    def test_defers_wide_artifact_spread_to_demo_gallery(self, skill_text):
        assert "demo-gallery" in skill_text


class TestDemoUiInstall:
    """End-to-end: the skill must actually land on disk via the standard build path."""

    def test_control_assistant_build_installs_the_skill(self, tmp_path):
        manager = TemplateManager()
        project_dir = _create_project(
            manager,
            project_name="demo-ui-install-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        installed = project_dir / ".claude" / "skills" / "demo-ui" / "SKILL.md"
        assert installed.exists(), f"Skill not installed at {installed}"
        assert installed.read_text(encoding="utf-8") == SKILL_PATH.read_text(encoding="utf-8")

    def test_resolve_manifest_outputs_includes_the_skill(self):
        mf = {"artifacts": {"skills": ["demo-ui"]}}
        outputs = manifest.resolve_manifest_outputs(mf)
        assert OUTPUT_REL in outputs

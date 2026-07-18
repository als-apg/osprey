"""Tests for the operating-bluesky-scans skill and its 4-point framework wiring.

Mirrors ``test_writing_bluesky_plans_skill_install.py``'s registry/template/
preset/manifest wiring pattern, plus a real ``TemplateManager.create_project``
build (the standard skill-install path for the
``templates/claude_code/claude/skills/`` family) to verify the skill actually
lands on disk end to end.

The content assertions pin the draft-first run surface: the skill choreographs
``set_draft`` -> human review -> ``launch_run(draft_revision)`` -> watch, and
must never name the deleted/renamed tools (``create_run_*``, ``run_status``,
``read_run_data``, ``*_plan_draft``).
"""

import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import manifest
from osprey.cli.templates.manager import TemplateManager
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

TEMPLATE_ROOT = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"
PRESETS_DIR = Path(__file__).parent.parent.parent / "src" / "osprey" / "profiles" / "presets"

SKILL_REL = "claude/skills/operating-bluesky-scans/SKILL.md"
OUTPUT_REL = ".claude/skills/operating-bluesky-scans/SKILL.md"


class TestOperatingBlueskyScansRegistry:
    """Wiring point 1: BuildArtifactCatalog registration."""

    @pytest.fixture()
    def registry(self):
        return BuildArtifactCatalog.default()

    def test_registered(self, registry):
        art = registry.get("skills/operating-bluesky-scans")
        assert art is not None
        assert art.output_path == OUTPUT_REL
        assert art.template_path == SKILL_REL


class TestOperatingBlueskyScansTemplateExists:
    """Wiring point 2: the skill bundle template file itself."""

    def test_skill_file_exists(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "operating-bluesky-scans" / "SKILL.md"
        assert path.exists(), f"SKILL.md not found at {path}"


class TestOperatingBlueskyScansPresetWiring:
    """Wiring point 3: the preset's ``skills:`` directive."""

    def test_control_assistant_lists_the_skill(self):
        profile_text = (PRESETS_DIR / "control-assistant.yml").read_text(encoding="utf-8")
        profile = yaml.safe_load(profile_text)
        assert "operating-bluesky-scans" in profile["skills"]


class TestOperatingBlueskyScansManifestWiring:
    """Wiring point 4: the regen-tracked-files fallback list."""

    def test_in_regen_tracked_files(self):
        assert OUTPUT_REL in manifest.REGEN_TRACKED_FILES


class TestOperatingBlueskyScansSkillStructure:
    """Content assertions: the draft-first run surface and its choreography."""

    @pytest.fixture()
    def skill_text(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "operating-bluesky-scans" / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def test_has_frontmatter(self, skill_text):
        assert skill_text.startswith("---")
        assert "name: operating-bluesky-scans" in skill_text

    # --- the shared-draft tool surface ---

    def test_documents_draft_tools(self, skill_text):
        for tool in ("get_draft", "set_draft", "clear_draft"):
            assert tool in skill_text, f"Missing draft tool: {tool}"

    def test_documents_launch_on_pinned_revision(self, skill_text):
        assert "launch_run" in skill_text
        assert "draft_revision" in skill_text

    def test_documents_watch_and_stop_tools(self, skill_text):
        for tool in ("get_run", "get_run_data", "list_runs", "list_plans", "stop_run"):
            assert tool in skill_text, f"Missing run tool: {tool}"

    # --- the choreography ---

    def test_stage_complete_config_in_one_call(self, skill_text):
        """A partial draft is a launchable hazard -- staging must be one call."""
        assert "set_draft" in skill_text
        lowered = skill_text.lower()
        assert "piecemeal" in lowered
        assert "revision" in lowered

    def test_documents_409_recovery_codes(self, skill_text):
        for code in ("stale_draft_revision", "draft_revision_already_launched"):
            assert code in skill_text, f"Missing 409 recovery code: {code}"

    def test_documents_launch_refusal_codes(self, skill_text):
        for code in ("run_launch_unarmed", "run_launch_forbidden", "run_launch_conflict"):
            assert code in skill_text, f"Missing refusal code: {code}"

    def test_points_at_authoring_skill(self, skill_text):
        assert "writing-bluesky-plans" in skill_text

    # --- must never name the deleted/renamed tools ---

    def test_no_stale_tool_names(self, skill_text):
        for stale in (
            "create_run_intent",
            "create_run_",
            "run_status",
            "read_run_data",
            "get_plan_draft",
            "set_plan_draft",
            "clear_plan_draft",
        ):
            assert stale not in skill_text, f"Stale tool name leaked into skill: {stale}"

    def test_no_purged_run_vocabulary(self, skill_text):
        """The run vocabulary settled on launch/contribute; the purged words --
        ``promote`` (tier vocabulary is now ``contribute``), run-state
        ``intent``, and ``execute``/``executes``/``executing`` as a
        launch-synonym verb -- must not reappear. Word-boundary-aware and
        case-sensitive on the ``execute`` check so the plan panel's actual
        ``Execute`` button label (a proper noun) and prose like ``executed``
        do not false-positive."""
        assert not re.search(r"(?i)\bpromot", skill_text), "purged tier word 'promote' leaked"
        assert not re.search(r"(?i)\bintent\b", skill_text), "purged run-state word 'intent' leaked"
        assert not re.search(r"\bexecut(?:e|es|ing)\b", skill_text), (
            "purged launch-synonym verb 'execute' leaked (the 'Execute' button label is fine)"
        )


class TestOperatingBlueskyScansInstall:
    """End-to-end: the skill must actually land on disk via the standard build path."""

    def test_control_assistant_build_installs_the_skill(self, tmp_path):
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="operating-bluesky-scans-install-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        installed = project_dir / ".claude" / "skills" / "operating-bluesky-scans" / "SKILL.md"
        assert installed.exists(), f"Skill not installed at {installed}"

        template_text = (
            TEMPLATE_ROOT / "claude" / "skills" / "operating-bluesky-scans" / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert installed.read_text(encoding="utf-8") == template_text

    def test_resolve_manifest_outputs_includes_the_skill(self):
        mf = {"artifacts": {"skills": ["operating-bluesky-scans"]}}
        outputs = manifest.resolve_manifest_outputs(mf)
        assert OUTPUT_REL in outputs

"""Tests for the writing-bluesky-plans skill and its 4-point framework wiring.

Mirrors ``test_session_report_skill.py``'s registry/template/content pattern,
plus a real ``TemplateManager.create_project`` build (the standard
skill-install path for the ``templates/claude_code/claude/skills/`` family --
these skills are NOT installed via the ``osprey skills install`` CLI, which
only serves the separate ``templates/skills/`` global-skill family) to verify
the skill actually lands on disk end to end.

The content assertions pin the session-plan flow as it actually runs: author
-> ``validate_plan`` (whose PASS also uploads the validated bytes for that
content hash) -> stage into the shared draft -> ``queue_add`` ->
``queue_start``. ``launch_run`` is asserted ABSENT — that tool and its route
were retired when plans moved into the queue server.
"""

import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import manifest
from osprey.cli.templates.manager import TemplateManager
from osprey.services.bluesky_bridge import plan_validation
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

TEMPLATE_ROOT = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"
PRESETS_DIR = Path(__file__).parent.parent.parent / "src" / "osprey" / "profiles" / "presets"


def _prose(text: str) -> str:
    """Lowercased skill text with markdown decoration and line wrapping removed.

    Same helper as in ``test_operating_bluesky_scans_skill_install.py``: the
    sentence-level pins assert what the prose SAYS, so a cosmetic re-wrap or an
    emphasis change must not fail them, while a weakened sentence still does.
    """
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text)).lower()


class TestWritingBlueskyPlansRegistry:
    """Wiring point 1: BuildArtifactCatalog registration."""

    @pytest.fixture()
    def registry(self):
        return BuildArtifactCatalog.default()

    def test_registered(self, registry):
        art = registry.get("skills/writing-bluesky-plans")
        assert art is not None
        assert art.output_path == ".claude/skills/writing-bluesky-plans/SKILL.md"
        assert art.template_path == "claude/skills/writing-bluesky-plans/SKILL.md"


class TestWritingBlueskyPlansTemplateExists:
    """Wiring point 2: the skill bundle template file itself."""

    def test_skill_file_exists(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "writing-bluesky-plans" / "SKILL.md"
        assert path.exists(), f"SKILL.md not found at {path}"


class TestWritingBlueskyPlansPresetWiring:
    """Wiring point 3: the preset's ``skills:`` directive."""

    def test_control_assistant_lists_the_skill(self):
        profile_text = (PRESETS_DIR / "control-assistant.yml").read_text(encoding="utf-8")
        profile = yaml.safe_load(profile_text)
        assert "writing-bluesky-plans" in profile["skills"]


class TestWritingBlueskyPlansManifestWiring:
    """Wiring point 4: the regen-tracked-files fallback list."""

    def test_in_regen_tracked_files(self):
        assert ".claude/skills/writing-bluesky-plans/SKILL.md" in manifest.REGEN_TRACKED_FILES


class TestWritingBlueskyPlansSkillStructure:
    """Content assertions: allowlist, validate-before-launch flow, bps.sleep guidance."""

    @pytest.fixture()
    def skill_text(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "writing-bluesky-plans" / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def test_has_frontmatter(self, skill_text):
        assert skill_text.startswith("---")
        assert "name: writing-bluesky-plans" in skill_text

    # --- plan-file format ---

    def test_documents_plan_metadata(self, skill_text):
        assert "PLAN_METADATA" in skill_text
        for field in ("name", "description", "category", "required_devices", "writes"):
            assert field in skill_text

    def test_documents_params_and_build_plan(self, skill_text):
        assert "PARAMS" in skill_text
        assert "build_plan" in skill_text
        assert "pydantic.BaseModel" in skill_text or "BaseModel" in skill_text

    def test_references_exemplars(self, skill_text):
        assert "orm" in skill_text
        assert "grid_scan" in skill_text
        assert "plans_core/orm.py" in skill_text
        assert "plans_core/grid_scan.py" in skill_text

    # --- the allowlist ---

    def test_documents_allowed_imports(self, skill_text):
        """The skill's enumeration is keyed to the validator's own constant so the
        prose cannot silently drift when ``_ALLOWED_TOP_LEVEL_MODULES`` changes.

        ``__future__`` is excluded: the validator allows it, but the skill
        deliberately tells authors never to write ``from __future__ import ...``
        in a plan body (the dry-run embeds the body mid-script, where a
        ``__future__`` import is a SyntaxError), so the prose must NOT list it.
        """
        for allowed in ("bluesky.plan_stubs", "bluesky.plans", "bluesky.preprocessors"):
            assert allowed in skill_text, f"Missing allowed import: {allowed}"
        for allowed in sorted(plan_validation._ALLOWED_TOP_LEVEL_MODULES - {"__future__"}):
            assert allowed in skill_text, f"Missing allowed import: {allowed}"

    def test_documents_forbidden_imports(self, skill_text):
        for forbidden in ("epics", "os", "subprocess", "ctypes", "importlib", "socket"):
            assert forbidden in skill_text, f"Missing forbidden import mention: {forbidden}"

    def test_documents_ca_pattern_scan(self, skill_text):
        for pattern in ("caput(", "caget(", "_osprey_connector", "write_channel("):
            assert pattern in skill_text

    # --- the foot-gun ---

    def test_documents_bps_sleep_guidance(self, skill_text):
        assert "bps.sleep" in skill_text
        assert "time.sleep" in skill_text
        assert "RunEngine" in skill_text

    # --- author -> validate -> run -> contribute workflow ---

    def test_documents_write_and_validate_tools(self, skill_text):
        assert "write_plan" in skill_text
        assert "validate_plan" in skill_text

    def test_documents_run_tools(self, skill_text):
        for tool in ("set_draft", "queue_add", "queue_start", "list_plans"):
            assert tool in skill_text, f"Missing tool reference: {tool}"

    def test_does_not_name_the_retired_launch_tool(self, skill_text):
        """``launch_run`` and its route were retired when plans moved into the
        queue server; prose naming it points at a dead surface."""
        assert "launch_run" not in skill_text

    def test_documents_the_upload_block_on_validate(self, skill_text):
        """A PASS triggers a script upload for that content hash, and the
        verdict is preserved regardless of how the upload went."""
        assert "upload" in skill_text
        assert "uploaded" in skill_text
        assert "content_hash" in skill_text

    def test_documents_revalidation_on_edit(self, skill_text):
        """The validation record is content-hash keyed, and the hash is
        re-checked at enqueue AND at queue start."""
        lowered = skill_text.lower()
        assert "content hash" in lowered
        assert "session_plan_" in skill_text, "the re-validate signal is a session_plan_* code"

    def test_documents_kwargs_are_params_unwrapped(self, skill_text):
        """Queue-item kwargs ARE the PARAMS fields; there is no envelope, and
        bridge-side pre-enqueue validation is the early gate."""
        lowered = skill_text.lower()
        assert "unwrapped" in lowered
        assert "envelope" in lowered
        assert "kwargs" in lowered

    def test_documents_leading_underscore_names_are_rejected(self, skill_text):
        assert "underscore" in skill_text.lower()

    def test_documents_contribute_to_permanent_pointer(self, skill_text):
        assert "contribute" in skill_text.lower()
        assert "promote" not in skill_text.lower()

    # --- explicit out-of-scope guidance ---

    def test_explicitly_rules_out_bba_and_tune_scan(self, skill_text):
        """BBA/tune-scan must appear only as a named anti-pattern, never as a
        plan option to author (memory: shipped scan plans are `orm` (ORM) +
        n-d `grid_scan` ONLY; bba/tune_scan "always creeps in")."""
        assert "propose a bba or tune-scan plan" in _prose(skill_text)
        assert "tune-scan" in skill_text
        assert "out of scope" in skill_text.lower()
        assert "only accelerator scan patterns" in _prose(skill_text)


class TestWritingBlueskyPlansInstall:
    """End-to-end: the skill must actually land on disk via the standard build path.

    ``templates/claude_code/claude/skills/`` artifacts are installed by
    ``osprey build`` / ``TemplateManager.create_project`` (BuildArtifactCatalog
    + preset artifact resolution), not by the ``osprey skills install`` CLI
    (that command only serves ``templates/skills/`` global skills).
    """

    def test_control_assistant_build_installs_the_skill(self, tmp_path):
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="writing-bluesky-plans-install-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        installed = project_dir / ".claude" / "skills" / "writing-bluesky-plans" / "SKILL.md"
        assert installed.exists(), f"Skill not installed at {installed}"

        template_text = (
            TEMPLATE_ROOT / "claude" / "skills" / "writing-bluesky-plans" / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert installed.read_text(encoding="utf-8") == template_text

    def test_resolve_manifest_outputs_includes_the_skill(self):
        mf = {"artifacts": {"skills": ["writing-bluesky-plans"]}}
        outputs = manifest.resolve_manifest_outputs(mf)
        assert ".claude/skills/writing-bluesky-plans/SKILL.md" in outputs

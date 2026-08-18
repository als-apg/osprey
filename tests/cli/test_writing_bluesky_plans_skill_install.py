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

import ast
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
AUTHORING_TOOLS = (
    Path(__file__).parent.parent.parent / "src" / "osprey" / "mcp_server" / "bluesky" / "tools"
)


def _prose(text: str) -> str:
    """Lowercased skill text with markdown decoration and line wrapping removed.

    Same helper as in ``test_operating_bluesky_plans_skill_install.py``: the
    sentence-level pins assert what the prose SAYS, so a cosmetic re-wrap or an
    emphasis change must not fail them, while a weakened sentence still does.
    """
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text)).lower()


def _metadata_section(text: str) -> str:
    """Just the ``PLAN_METADATA`` item of the plan-file-format list.

    The retired metadata keys are pinned absent from THIS span rather than
    from the whole skill, so unrelated prose elsewhere (a figure's "one value
    per named category") cannot make the pin either fail or rot.
    """
    start = text.index("**`PLAN_METADATA`**")
    end = text.index("2. **`PARAMS`**", start)
    return text[start:end]


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
        """``PLAN_METADATA`` is exactly three keys, and the section says so.

        The retired ``category``/``required_devices`` keys are asserted absent
        from that section rather than from the whole file: an author who reads
        them there writes them, and the loader now refuses the block outright
        (unknown keys forbidden), so the plan would never register.
        """
        assert "PLAN_METADATA" in skill_text
        section = _metadata_section(skill_text)
        for field in ("name", "description", "writes"):
            assert field in section, f"PLAN_METADATA section never names {field!r}"
        for retired in ("category", "required_devices"):
            assert retired not in section, (
                f"PLAN_METADATA section still offers the retired key {retired!r}"
            )

    def test_documents_the_role_typed_channel_declaration(self, skill_text):
        """Channels are declared on ``PARAMS`` fields, with all four helpers.

        The list pair and the nested-model pair are not interchangeable — an
        author with a nested shape (``grid_scan``'s axes) needs the singles —
        so the skill has to name all four and the module they come from.
        """
        for helper in (
            "MovableChannels",
            "ReadableChannels",
            "MovableChannel",
            "ReadableChannel",
        ):
            assert helper in skill_text, f"Missing role-typed field helper: {helper}"
        assert "osprey.services.bluesky_bridge.plan_fields" in skill_text
        prose = _prose(skill_text)
        assert "movable" in prose and "readable" in prose

    def test_documents_the_run_metadata_stamp(self, skill_text):
        """A self-opened run stamps its own metadata; a stock-plan wrapper does not.

        Both halves matter: an unstamped self-opened run fails the dry-run
        gate, and a second stamp over a stock plan's would overwrite a more
        accurate one.
        """
        assert "scan_metadata(" in skill_text
        for kwarg in ("movable=", "readable=", "points="):
            assert kwarg in skill_text, f"scan_metadata call never shows {kwarg}"
        prose = _prose(skill_text)
        assert "a plan that delegates to a stock bluesky plan needs no stamp at all" in prose

    def test_documents_both_gates_a_writing_plan_must_clear(self, skill_text):
        """The two refusals an author actually hits, with their own remedies.

        Quoted close enough to be recognizable when one lands: the load-gate
        quarantine (writes without a movable field) and the dry-run point-count
        rejection.
        """
        prose = _prose(skill_text)
        assert "declares writes: true but no movable channel field in params" in prose
        assert "dry-run gate:" in prose
        assert "opened a run that declares no point count" in prose
        assert "declares that it moves channels but opened no run" in prose

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

    def test_documents_the_current_write_plan_call(self, skill_text):
        """The call is four arguments, and channels are not among them.

        Pinned against the real tool signature rather than a literal, so a
        parameter added to or dropped from ``write_plan`` fails here instead of
        leaving the skill teaching a call the bridge would refuse. Read with
        ``ast`` rather than by import: the signature is a property of the
        source, and this test has no business starting an MCP server.
        """
        source = (AUTHORING_TOOLS / "authoring.py").read_text(encoding="utf-8")
        (defn,) = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "write_plan"
        ]
        params = [arg.arg for arg in defn.args.args]
        assert params == ["name", "writes", "body", "description"], (
            "write_plan's signature moved; the skill's documented call must move with it"
        )
        assert 'write_plan(name, writes, body, description="")' in skill_text

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

    # --- the plan's own view ---

    def test_documents_the_render_contract(self, skill_text):
        """``render`` is optional, but the three things an author gets wrong
        about it are not: its signature, that it must never raise, and that its
        labels come from the run rather than from a facility."""
        assert "render(window, params)" in skill_text
        prose = _prose(skill_text)
        assert "render() must never raise" in prose
        assert "render_failed" in skill_text
        assert "stay facility-neutral" in prose

    def test_documents_the_figure_vocabulary(self, skill_text):
        """One panel carries exactly one mark; an author has to know which
        three exist before choosing between them."""
        for name in ("Figure", "Panel", "LinesMark", "BarsMark", "HeatmapMark"):
            assert name in skill_text, f"Missing figure model: {name}"

    def test_documents_the_osprey_modules_a_plan_may_import(self, skill_text):
        """Keyed to the validator's own constant, and bidirectional.

        Set equality rather than a membership loop: a loop catches a WIDENED
        allowlist (a new module the prose never mentions) but not a narrowed
        one, which would leave the skill telling authors to import something
        the validator now rejects -- stale prose staying green. The dotted
        ``osprey.*`` strings in this skill are exactly the allow-listed modules
        (``_osprey_connector`` in the pattern-scan section has no dot and does
        not match), so equality holds today and fails on drift in either
        direction. It also pins the "Exactly two" claim in the prose without
        asserting on that wording.
        """
        named = set(re.findall(r"osprey\.[\w.]*\w", skill_text))
        assert named == set(plan_validation._ALLOWED_OSPREY_SUBMODULES)

    def test_documents_the_session_tier_render_exclusion(self, skill_text):
        """A session plan's ``render`` is never run -- and the reason an author
        needs is that nothing about running the plan changes, only the drawing,
        so this never reads as a plan that half-works."""
        assert "render_not_supported_for_session_plans" in skill_text
        prose = _prose(skill_text)
        assert "nothing about the execution surface changes" in prose

    def test_documents_the_figure_watch_tool(self, skill_text):
        """A plan that ships a view is exactly the case where the figure read
        beats the row read."""
        assert "get_run_figure" in skill_text

    # --- explicit out-of-scope guidance ---

    def test_explicitly_rules_out_bba_and_tune_scan(self, skill_text):
        """BBA/tune-scan must appear only as a named anti-pattern, never as a
        plan option to author (memory: shipped plans are `orm` (ORM) +
        n-d `grid_scan` ONLY; bba/tune_scan "always creeps in")."""
        assert "propose a bba or tune-scan plan" in _prose(skill_text)
        assert "tune-scan" in skill_text
        assert "out of scope" in skill_text.lower()
        assert "only accelerator plan patterns" in _prose(skill_text)


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

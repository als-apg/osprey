"""Tests for the operating-bluesky-scans skill and its 4-point framework wiring.

Mirrors ``test_writing_bluesky_plans_skill_install.py``'s registry/template/
preset/manifest wiring pattern, plus a real ``TemplateManager.create_project``
build (the standard skill-install path for the
``templates/claude_code/claude/skills/`` family) to verify the skill actually
lands on disk end to end.

The content assertions pin the draft-first run surface: the skill choreographs
``set_draft`` -> human review -> ``queue_add(draft_revision)`` ->
``queue_start`` -> watch, and must never name the deleted/renamed tools
(``create_run_*``, ``run_status``, ``read_run_data``, ``*_plan_draft``,
``launch_run``).

Two assertions here are safety pins rather than documentation checks, and
both fail in the direction that matters:

* ``test_no_stale_tool_names`` includes ``launch_run``. Neither that tool nor
  its route exists; prose naming it would send the agent at a dead surface.
* ``TestOperatingBlueskyScansStopHonesty`` pins that the skill tells the truth
  about halting — that a plain ``queue_stop`` halts the queue only after the
  running item finishes, that ``stop_run`` aborts the plan already in motion,
  what that abort costs, and that a failed abort is never reported as a halt.
  Its last test is the inverse pin: wording that denies any abort ("no OSPREY
  surface does", "out-of-band") must not sit next to the tool that performs
  one, because that combination tells an agent not to reach for a working halt.
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


def _prose(text: str) -> str:
    """Lowercased skill text with markdown decoration and line wrapping removed.

    The safety pins below assert on whole sentences, which in the source are
    line-wrapped and carry ``**bold**``/``*emphasis*``/``` `code` ``` markers.
    Matching the rendered prose instead of the raw bytes keeps a re-wrap or an
    emphasis change from failing a test whose subject is what the sentence
    SAYS -- while still failing loudly if the sentence itself is weakened.
    """
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text)).lower()


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

    # --- the queue tool surface ---

    def test_documents_the_queue_tools(self, skill_text):
        for tool in ("queue_add", "queue_list", "queue_start", "queue_stop", "queue_status"):
            assert tool in skill_text, f"Missing queue tool: {tool}"

    def test_documents_enqueue_on_pinned_revision(self, skill_text):
        assert "queue_add" in skill_text
        assert "draft_revision" in skill_text

    def test_documents_two_step_add_then_start(self, skill_text):
        """Execution is two steps by design; the arming gate sits on start."""
        assert skill_text.index("queue_add") < skill_text.index("queue_start")
        lowered = skill_text.lower()
        assert "two steps" in lowered

    def test_documents_watch_tools(self, skill_text):
        for tool in ("get_run", "get_run_data", "list_runs", "list_plans"):
            assert tool in skill_text, f"Missing run tool: {tool}"

    # --- the choreography ---

    def test_stage_complete_config_in_one_call(self, skill_text):
        """A partial draft is a launchable hazard -- staging must be one call."""
        assert "set_draft" in skill_text
        lowered = skill_text.lower()
        assert "piecemeal" in lowered
        assert "revision" in lowered

    def test_documents_revision_recovery_codes(self, skill_text):
        for code in ("stale_draft_revision", "draft_revision_already_launched"):
            assert code in skill_text, f"Missing revision recovery code: {code}"

    def test_documents_bridge_refusal_codes(self, skill_text):
        """Refusals are machine-readable and relayed verbatim, so the skill must
        name the codes the agent branches on -- see the bridge's queue wire
        contract."""
        for code in (
            "launch_token_required",
            "session_plan_unvalidated",
            "session_plan_not_in_namespace",
            "browse_only_connector",
            "manager_unreachable",
            "environment_unavailable",
            "queue_request_rejected",
        ):
            assert code in skill_text, f"Missing refusal code: {code}"
        assert "detail.code" in skill_text, "the skill must say to branch on detail.code"

    def test_documents_capability_handling(self, skill_text):
        """A browse-only deployment composes and validates but never runs."""
        assert "can_execute" in skill_text
        assert "queue_status" in skill_text
        lowered = skill_text.lower()
        assert "browse-only" in lowered
        assert "verbatim" in lowered, "the capability detail carries the flip command verbatim"

    def test_documents_arming_gates(self, skill_text):
        assert "control_system.writes_enabled" in skill_text
        assert "launch token" in skill_text

    def test_points_at_authoring_skill(self, skill_text):
        assert "writing-bluesky-plans" in skill_text

    # --- must never name the deleted/renamed tools ---

    def test_no_stale_tool_names(self, skill_text):
        """``launch_run`` is in this list deliberately: the tool and its route
        were retired when plans moved into the queue server, so naming it in
        agent-facing prose would point at a dead surface."""
        for stale in (
            "create_run_intent",
            "create_run_",
            "run_status",
            "read_run_data",
            "get_plan_draft",
            "set_plan_draft",
            "clear_plan_draft",
            "launch_run",
        ):
            assert stale not in skill_text, f"Stale tool name leaked into skill: {stale}"

    def test_no_purged_run_vocabulary(self, skill_text):
        """``promote`` (the tier vocabulary is ``contribute``) and run-state
        ``intent`` must not reappear.

        The former ban on ``execute``/``executes``/``executing`` as a
        launch-synonym verb is deliberately gone: with the queue, "execute" is
        the wire vocabulary itself -- ``can_execute``, ``executable``,
        ``executing_queue`` -- and the skill has to say plainly which
        deployments cannot run plans."""
        assert not re.search(r"(?i)\bpromot", skill_text), "purged tier word 'promote' leaked"
        assert not re.search(r"(?i)\bintent\b", skill_text), "purged run-state word 'intent' leaked"


class TestOperatingBlueskyScansStopHonesty:
    """Safety pins: the skill must describe each halt as exactly what it is.

    There are two, and they are not interchangeable. ``queue_stop`` halts
    the queue only AFTER the running item finishes; ``stop_run`` aborts the
    plan already moving hardware, via ``POST /queue/abort``. Prose that blurs
    them is read by whoever is deciding whether a queue-halt is enough, at the
    moment delay costs most.

    These pins replace the earlier set, which asserted that NO surface could
    abort and that ``stop_run`` was non-functional. Those were true of the
    retired ``POST /runs/{id}/stop`` and went false in the dangerous direction
    the moment a real abort was wired up -- an agent told not to reach for a
    tool that now works. The negative pin below is the same guard pointed the
    other way, so the retired wording cannot creep back.
    """

    @pytest.fixture()
    def skill_text(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "operating-bluesky-scans" / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def test_stop_is_after_the_running_item(self, skill_text):
        """``queue_stop``'s limit must be stated, because a tool exists
        which does not share it -- otherwise the two read as synonyms."""
        prose = _prose(skill_text)
        assert "queue_stop" in prose
        assert "stops the queue after the currently running item finishes" in prose
        assert "does not abort a plan that is already moving hardware" in prose

    def test_stop_run_is_described_as_the_working_abort(self, skill_text):
        """``stop_run`` aborts the RUNNING plan. The skill has to say so plainly
        and route the agent to it, since ``queue_stop`` is what an agent will
        otherwise reach for when someone says "stop"."""
        prose = _prose(skill_text)
        assert "stop_run" in prose
        assert "aborts the plan that is running right now" in prose

    def test_the_abort_states_what_it_costs(self, skill_text):
        """An abort is not a free halt: it discards the rest of the scan and
        leaves the machine wherever it stopped. An agent that proposes one
        without saying that has mis-sold it."""
        prose = _prose(skill_text)
        assert "the hardware is left wherever the scan had moved it" in prose
        assert "returns nothing to a starting position" in prose

    def test_a_failed_abort_is_never_reported_as_a_halt(self, skill_text):
        """The one lie this surface must not tell. ``abort_pause_timeout`` means
        the plan may still be running, and the skill must name the code and the
        consequence rather than leaving "the abort failed" to interpretation."""
        prose = _prose(skill_text)
        assert "abort_pause_timeout" in prose
        assert "nothing was aborted and the plan may still be running" in prose
        assert "nothing_running" in prose

    def test_both_halts_are_documented_as_ungated(self, skill_text):
        """Halting must never have a failure mode: no writes check, no token,
        on EITHER halt -- and the one arming exception stays distinguished."""
        prose = _prose(skill_text)
        assert "ungated" in prose
        assert "no writes_enabled check, no launch token, here or at the bridge" in prose, (
            "stop_run's ungated posture must be stated as plainly as queue_stop's"
        )
        assert "cancel=true" in prose, "the armed withdrawal case must be distinguished"

    def test_the_retired_no_abort_wording_is_gone(self, skill_text):
        """Inverse drift: the prose written when nothing could abort must not
        survive alongside the tool that now can. Each phrase below asserted, in
        agent-facing text, that no halt existed for a moving scan."""
        prose = _prose(skill_text)
        for retired in (
            "no osprey surface does",
            "fails and stops nothing",
            "out-of-band",
            "nothing here aborts",
        ):
            assert retired not in prose, (
                f"the skill still carries pre-abort wording {retired!r} -- it now "
                f"contradicts stop_run"
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

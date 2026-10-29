"""Tests for the session-report skill and its reference file."""

from pathlib import Path

import pytest

from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog

TEMPLATE_ROOT = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"


class TestSessionReportRegistry:
    """Verify both session-report artifacts are registered."""

    @pytest.fixture()
    def registry(self):
        return BuildArtifactCatalog.default()

    def test_session_report_registered(self, registry):
        art = registry.get("skills/session-report")
        assert art is not None
        assert art.output_path == ".claude/skills/session-report/SKILL.md"
        assert art.template_path == "claude/skills/session-report/SKILL.md"

    def test_session_report_reference_registered(self, registry):
        art = registry.get("skills/session-report/reference")
        assert art is not None
        assert art.output_path == ".claude/skills/session-report/reference.md"
        assert art.template_path == "claude/skills/session-report/reference.md"


class TestSessionReportTemplateExists:
    """Verify template files exist on disk."""

    def test_skill_file_exists(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "session-report" / "SKILL.md"
        assert path.exists(), f"SKILL.md not found at {path}"

    def test_reference_file_exists(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "session-report" / "reference.md"
        assert path.exists(), f"reference.md not found at {path}"


class TestSessionReportSkillStructure:
    """Verify the skill file contains required structural elements."""

    @pytest.fixture()
    def skill_text(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "session-report" / "SKILL.md"
        return path.read_text()

    # --- Phase structure ---

    def test_has_four_phases(self, skill_text):
        """Skill must define all 4 phases."""
        for phase in ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]:
            assert phase in skill_text, f"Missing {phase}"

    @pytest.mark.parametrize(
        "needle",
        [
            pytest.param("Intent", id="phase1_intent"),
            pytest.param("Inventory", id="phase2_inventory"),
            pytest.param("Delegate", id="phase3_delegate"),
            pytest.param("Register", id="phase4_register"),
            # The tools the phases name.
            pytest.param("session_summary", id="references_session_summary"),
            pytest.param("archiver_downsample", id="references_archiver_downsample"),
            pytest.param("artifact_register", id="references_artifact_register"),
            pytest.param("reference.md", id="references_reference_file"),
            # Mermaid diagrams, offered as block types.
            pytest.param("Mermaid event timeline", id="mentions_mermaid_event_timeline"),
            pytest.param("Mermaid state diagram", id="mentions_mermaid_state_diagram"),
            pytest.param("Anti-Pattern", id="has_anti_patterns_section"),
        ],
    )
    def test_names(self, skill_text, needle):
        """Each phase, tool, diagram type and section the skill depends on is named."""
        assert needle in skill_text

    # --- Intent modes ---

    @pytest.mark.parametrize(
        "intent",
        [
            "Session Log",
            "Analysis Report",
            "Executive Briefing",
        ],
    )
    def test_has_intent_mode(self, skill_text, intent):
        assert intent in skill_text, f"Missing intent mode: {intent}"

    # --- Safety rules ---

    def test_has_observation_only_rule(self, skill_text):
        """Must contain the safety rule about observation-only reporting."""
        assert (
            "never generate recommendations" in skill_text.lower()
            or "observation-only" in skill_text.lower()
        )

    def test_has_forbidden_examples(self, skill_text):
        """Must give examples of forbidden recommendation language."""
        assert "FORBIDDEN" in skill_text or "forbidden" in skill_text

    def test_has_allowed_examples(self, skill_text):
        """Must give examples of allowed observation language."""
        assert "ALLOWED" in skill_text or "allowed" in skill_text

    def test_has_attributed_quote_exception(self, skill_text):
        """Exception for operator-stated action items as attributed quotes."""
        assert "attributed quote" in skill_text.lower()

    # --- No hardcoded section list ---

    def test_no_fixed_eight_sections(self, skill_text):
        """Must NOT have the old 8-section hardcoded multi-select list."""
        # The old approach had numbered sections 1-8 with specific names
        assert "8 report sections" not in skill_text
        assert "Scope the Report" not in skill_text

    # --- Anti-patterns ---

    def test_anti_pattern_no_recommendations(self, skill_text):
        """Anti-patterns section must mention not generating recommendations."""
        lower = skill_text.lower()
        assert "recommend" in lower

    # --- Skill frontmatter ---

    def test_has_frontmatter(self, skill_text):
        """Skill must have YAML frontmatter with required fields."""
        assert skill_text.startswith("---")
        assert "name: session-report" in skill_text
        assert "disable-model-invocation: true" in skill_text


class TestSessionReportReferenceStructure:
    """Verify the reference file contains expected CSS/JS pattern sections."""

    @pytest.fixture()
    def reference_text(self):
        path = TEMPLATE_ROOT / "claude" / "skills" / "session-report" / "reference.md"
        return path.read_text()

    @pytest.mark.parametrize(
        "needles",
        [
            pytest.param(("--font-body", "--font-mono", "--accent"), id="custom_properties"),
            pytest.param(("node--hero", "node--elevated", "node--recessed"), id="depth_tiers"),
            pytest.param(("kpi-card", "countUp"), id="kpi_pattern"),
            pytest.param(("data-table", "sticky"), id="data_table_pattern"),
            pytest.param(("fadeUp", "fadeScale", "prefers-reduced-motion"), id="animations"),
            pytest.param(("min-width: 0", "overflow-wrap"), id="overflow_protection"),
            pytest.param(("collapsible",), id="collapsible_pattern"),
            pytest.param(("prefers-color-scheme: dark",), id="dark_mode"),
            pytest.param(("severity-card",), id="severity_cards"),
            pytest.param(("timeline",), id="timeline_pattern"),
            pytest.param(("diff-panel",), id="diff_panels"),
        ],
    )
    def test_has_pattern(self, reference_text, needles):
        """Each CSS/JS pattern the skill tells the agent to reach for is spelled out."""
        for needle in needles:
            assert needle in reference_text

    def test_has_mermaid_section(self, reference_text):
        assert "mermaid" in reference_text.lower()
        assert "zoom-controls" in reference_text

    def test_has_chartjs_section(self, reference_text):
        assert "Chart.js" in reference_text or "chart.js" in reference_text
        assert "chart.umd.min.js" in reference_text

    def test_chart_recipe_uses_per_dataset_x_values(self, reference_text):
        """The line-chart recipe must not teach a shared `labels` axis.

        ``archiver_downsample`` returns one ``timestamps`` array per channel,
        so each dataset needs its own ``{x, y}`` points.
        """
        assert "labels: timestamps" not in reference_text
        assert "x: Date.parse(" in reference_text
        assert "d.timestamps.map(" in reference_text
        # {x, y} point data requires parsing off and an explicit scale type.
        assert "parsing: false" in reference_text
        assert "type: 'linear'" in reference_text
        # Enum/status channels can't share the numeric y-axis.
        assert "d.numeric" in reference_text

    def test_has_font_pairing_table(self, reference_text):
        assert "Outfit" in reference_text
        assert "Space Mono" in reference_text
        assert "Inter" in reference_text  # mentioned as "never use"

    def test_has_responsive_nav(self, reference_text):
        assert "toc" in reference_text
        assert "scroll-spy" in reference_text.lower() or "Scroll-Spy" in reference_text

    def test_has_mermaid_event_timeline(self, reference_text):
        """Reference must include Mermaid timeline diagram syntax."""
        assert "Mermaid Event Timeline" in reference_text
        assert "timeline" in reference_text

    def test_has_mermaid_state_diagram(self, reference_text):
        """Reference must include Mermaid state diagram syntax."""
        assert "stateDiagram-v2" in reference_text
        assert "Mermaid State Diagram" in reference_text

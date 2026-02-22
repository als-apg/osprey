"""Tests for the session-report skill and its reference file."""

from pathlib import Path

import pytest

from osprey.cli.prompt_registry import PromptRegistry

TEMPLATE_ROOT = (
    Path(__file__).parent.parent.parent
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
)


class TestSessionReportRegistry:
    """Verify both session-report artifacts are registered."""

    @pytest.fixture()
    def registry(self):
        return PromptRegistry.default()

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

    def test_phase1_intent(self, skill_text):
        assert "Intent" in skill_text

    def test_phase2_inventory(self, skill_text):
        assert "Inventory" in skill_text

    def test_phase3_delegate(self, skill_text):
        assert "Delegate" in skill_text

    def test_phase4_register(self, skill_text):
        assert "Register" in skill_text

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
        assert "never generate recommendations" in skill_text.lower() or \
            "observation-only" in skill_text.lower()

    def test_has_forbidden_examples(self, skill_text):
        """Must give examples of forbidden recommendation language."""
        assert "FORBIDDEN" in skill_text or "forbidden" in skill_text

    def test_has_allowed_examples(self, skill_text):
        """Must give examples of allowed observation language."""
        assert "ALLOWED" in skill_text or "allowed" in skill_text

    def test_has_attributed_quote_exception(self, skill_text):
        """Exception for operator-stated action items as attributed quotes."""
        assert "attributed quote" in skill_text.lower()

    # --- Tool references ---

    def test_references_session_summary(self, skill_text):
        assert "session_summary" in skill_text

    def test_references_archiver_downsample(self, skill_text):
        assert "archiver_downsample" in skill_text

    def test_references_artifact_save(self, skill_text):
        assert "artifact_save" in skill_text

    def test_references_reference_file(self, skill_text):
        assert "reference.md" in skill_text

    # --- Mermaid diagram guidance ---

    def test_mentions_mermaid_event_timeline(self, skill_text):
        """Skill must mention Mermaid event timeline as a block type option."""
        assert "Mermaid event timeline" in skill_text

    def test_mentions_mermaid_state_diagram(self, skill_text):
        """Skill must mention Mermaid state diagram as a block type option."""
        assert "Mermaid state diagram" in skill_text

    # --- No hardcoded section list ---

    def test_no_fixed_eight_sections(self, skill_text):
        """Must NOT have the old 8-section hardcoded multi-select list."""
        # The old approach had numbered sections 1-8 with specific names
        assert "8 report sections" not in skill_text
        assert "Scope the Report" not in skill_text

    # --- Anti-patterns ---

    def test_has_anti_patterns_section(self, skill_text):
        assert "Anti-Pattern" in skill_text

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

    def test_has_custom_properties(self, reference_text):
        assert "--font-body" in reference_text
        assert "--font-mono" in reference_text
        assert "--accent" in reference_text

    def test_has_depth_tiers(self, reference_text):
        assert "node--hero" in reference_text
        assert "node--elevated" in reference_text
        assert "node--recessed" in reference_text

    def test_has_kpi_pattern(self, reference_text):
        assert "kpi-card" in reference_text
        assert "countUp" in reference_text

    def test_has_data_table_pattern(self, reference_text):
        assert "data-table" in reference_text
        assert "sticky" in reference_text

    def test_has_mermaid_section(self, reference_text):
        assert "mermaid" in reference_text.lower()
        assert "zoom-controls" in reference_text

    def test_has_chartjs_section(self, reference_text):
        assert "Chart.js" in reference_text or "chart.js" in reference_text
        assert "chart.umd.min.js" in reference_text

    def test_has_animations(self, reference_text):
        assert "fadeUp" in reference_text
        assert "fadeScale" in reference_text
        assert "prefers-reduced-motion" in reference_text

    def test_has_font_pairing_table(self, reference_text):
        assert "Outfit" in reference_text
        assert "Space Mono" in reference_text
        assert "Inter" in reference_text  # mentioned as "never use"

    def test_has_overflow_protection(self, reference_text):
        assert "min-width: 0" in reference_text
        assert "overflow-wrap" in reference_text

    def test_has_collapsible_pattern(self, reference_text):
        assert "collapsible" in reference_text

    def test_has_responsive_nav(self, reference_text):
        assert "toc" in reference_text
        assert "scroll-spy" in reference_text.lower() or "Scroll-Spy" in reference_text

    def test_has_dark_mode(self, reference_text):
        assert "prefers-color-scheme: dark" in reference_text

    def test_has_severity_cards(self, reference_text):
        assert "severity-card" in reference_text

    def test_has_timeline_pattern(self, reference_text):
        assert "timeline" in reference_text

    def test_has_mermaid_event_timeline(self, reference_text):
        """Reference must include Mermaid timeline diagram syntax."""
        assert "Mermaid Event Timeline" in reference_text
        assert "timeline" in reference_text

    def test_has_mermaid_state_diagram(self, reference_text):
        """Reference must include Mermaid state diagram syntax."""
        assert "stateDiagram-v2" in reference_text
        assert "Mermaid State Diagram" in reference_text

    def test_has_diff_panels(self, reference_text):
        assert "diff-panel" in reference_text

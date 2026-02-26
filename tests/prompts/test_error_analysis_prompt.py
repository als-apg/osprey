"""Tests for error analysis prompt builder — public API and system instructions."""

from types import SimpleNamespace

from osprey.prompts.defaults.error_analysis import DefaultErrorAnalysisPromptBuilder


class TestErrorAnalysisPromptAttributes:
    """Test error analysis prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        assert builder.PROMPT_TYPE == "error_analysis"

    def test_role_definition(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        role = builder.get_role()
        assert "error analysis" in role.lower()

    def test_task_definition_is_none(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        assert builder.get_task() is None

    def test_instructions_content(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        instructions = builder.get_instructions()
        assert "2-3 sentences" in instructions
        assert "why" in instructions.lower()
        assert "100 words" in instructions


class TestErrorAnalysisDynamicContext:
    """Test build_dynamic_context with various input combinations."""

    def test_capabilities_overview_included(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        ctx = builder.build_dynamic_context(capabilities_overview="Can read channels, plot data")
        assert "SYSTEM CAPABILITIES" in ctx
        assert "Can read channels, plot data" in ctx

    def test_error_context_included(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        error_ctx = SimpleNamespace(
            current_task="Fetch beam current",
            error_type=SimpleNamespace(value="TIMEOUT"),
            capability_name="channel_read",
            error_message="Connection timed out after 30s",
        )
        ctx = builder.build_dynamic_context(error_context=error_ctx)
        assert "ERROR CONTEXT" in ctx
        assert "Fetch beam current" in ctx
        assert "TIMEOUT" in ctx
        assert "channel_read" in ctx
        assert "Connection timed out" in ctx

    def test_empty_inputs_returns_empty(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        ctx = builder.build_dynamic_context()
        assert ctx == ""

    def test_both_inputs_combined(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        error_ctx = SimpleNamespace(
            current_task="Plot data",
            error_type=SimpleNamespace(value="RETRIABLE"),
            capability_name="python",
            error_message="Import error",
        )
        ctx = builder.build_dynamic_context(
            capabilities_overview="Python execution available",
            error_context=error_ctx,
        )
        assert "SYSTEM CAPABILITIES" in ctx
        assert "ERROR CONTEXT" in ctx

    def test_error_context_missing_attributes(self):
        """build_dynamic_context handles error context with missing attributes gracefully."""
        builder = DefaultErrorAnalysisPromptBuilder()
        error_ctx = SimpleNamespace()  # No attributes at all
        ctx = builder.build_dynamic_context(error_context=error_ctx)
        assert "ERROR CONTEXT" in ctx
        assert "Unknown" in ctx


class TestErrorAnalysisSystemInstructions:
    """Test the base class build_prompt composition."""

    def test_includes_role(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        prompt = builder.build_prompt()
        assert "error analysis" in prompt.lower()

    def test_includes_instructions(self):
        builder = DefaultErrorAnalysisPromptBuilder()
        prompt = builder.build_prompt()
        assert "2-3 sentences" in prompt

    def test_subclass_role_override(self):
        class CustomErrorBuilder(DefaultErrorAnalysisPromptBuilder):
            def get_role(self) -> str:
                return "You are a facility-specific error analyst."

        builder = CustomErrorBuilder()
        prompt = builder.build_prompt()
        assert "facility-specific error analyst" in prompt

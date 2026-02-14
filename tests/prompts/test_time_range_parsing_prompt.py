"""Tests for time range parsing prompt builder — public API, guides, and overrides."""

from osprey.prompts.defaults.time_range_parsing import DefaultTimeRangeParsingPromptBuilder


class TestTimeRangeParsingPromptAttributes:
    """Test time range parsing prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        assert builder.PROMPT_TYPE == "time_range_parsing"

    def test_role_definition(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        role = builder.get_role()
        assert "time range parser" in role.lower()

    def test_task_definition(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        task = builder.get_task()
        assert "Parse" in task
        assert "datetime" in task.lower()

    def test_instructions_content(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        instructions = builder.get_instructions()
        assert "CRITICAL" in instructions
        assert "YYYY-MM-DD" in instructions
        assert "STEP-BY-STEP" in instructions
        assert "Common patterns" in instructions


class TestTimeRangeParsingOrchestratorGuide:
    """Test orchestrator guide content."""

    def test_guide_exists(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None

    def test_guide_has_three_examples(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert len(guide.examples) == 3

    def test_guide_priority(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide.priority == 5

    def test_guide_instructions_content(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert "time_range_parsing" in guide.instructions
        assert "Context Key Format" in guide.instructions

    def test_examples_use_correct_capability(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "time_range_parsing"

    def test_relative_time_example(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        relative = guide.examples[0]
        assert "last week" in relative.step["task_objective"].lower()

    def test_absolute_time_example(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        absolute = guide.examples[1]
        assert "2024" in absolute.step["task_objective"]

    def test_implicit_time_example(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_orchestrator_guide()
        implicit = guide.examples[2]
        assert (
            "current" in implicit.step["task_objective"].lower()
            or "recent" in implicit.scenario_description.lower()
        )


class TestTimeRangeParsingClassifierGuide:
    """Test classifier guide content."""

    def test_guide_exists(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None

    def test_guide_has_examples(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_classifier_guide()
        assert len(guide.examples) == 8

    def test_guide_has_positive_examples(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 3

    def test_guide_has_negative_examples(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 3


class TestTimeRangeParsingOverrides:
    """Test that subclass overrides work."""

    def test_subclass_role_override(self):
        class CustomParser(DefaultTimeRangeParsingPromptBuilder):
            def get_role(self) -> str:
                return "You are a shift-schedule-aware time parser."

        builder = CustomParser()
        assert "shift-schedule" in builder.get_role()

    def test_subclass_instructions_override(self):
        class CustomParser(DefaultTimeRangeParsingPromptBuilder):
            def get_instructions(self) -> str:
                return "Parse accelerator run periods and shift boundaries."

        builder = CustomParser()
        prompt = builder.build_prompt()
        assert "accelerator run periods" in prompt


class TestTimeRangeParsingBuildPrompt:
    """Test build_prompt() composes static instructions with dynamic context."""

    def test_build_prompt_contains_critical_requirements(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt()
        assert "CRITICAL" in prompt
        assert "start_date MUST be BEFORE end_date" in prompt

    def test_build_prompt_contains_calculation_rules(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt()
        assert "STEP-BY-STEP" in prompt

    def test_build_prompt_contains_current_datetime(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt()
        assert "Current datetime:" in prompt

    def test_build_prompt_contains_user_query(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt(user_query="last 2 hours of beam current")
        assert "last 2 hours of beam current" in prompt

    def test_build_prompt_without_user_query(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt()
        assert "Current datetime:" in prompt
        assert "CRITICAL" in prompt
        # Should not contain "User query to parse:" when no query provided
        assert "User query to parse:" not in prompt

    def test_build_prompt_contains_common_patterns(self):
        builder = DefaultTimeRangeParsingPromptBuilder()
        prompt = builder.build_prompt()
        assert "yesterday" in prompt
        assert "last X hours" in prompt
        assert "this week" in prompt
